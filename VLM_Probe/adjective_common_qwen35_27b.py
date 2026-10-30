"""Shared, text-only experiment utilities; no inference imports at module load."""
from __future__ import annotations

import ast
import copy
import csv
import hashlib
import itertools
import json
import math
import os
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent
BENCHMARK_ROOT = ROOT.parent / 'SceneCritic/benchmark/benchmark_base_rooms'
OUTPUT_ROOT = ROOT / 'outputs/qwen35_27b_adjective_probe'
MODEL = 'Qwen/Qwen3.5-27B'
CATEGORIES = ('bedroom', 'bookstore', 'buffet_restaurant', 'classroom', 'computer_room', 'dining_room', 'living_room')
CONDITIONS = {'baseline': None, 'well_organized': 'well-organized', 'spacious': 'spacious', 'bright': 'bright', 'quiet': 'quiet', 'cluttered': 'cluttered', 'crowded': 'crowded'}
SEEDS = (1234, 11234, 21234, 31234, 41234)
STAGES = ('description', 'object_choice', 'planning', 'placement')
METRICS = ('object_set_diff', 'object_count_diff', 'occupancy_hamming', 'centroid_shift', 'legality_flip')
REFINEMENT = {'placement_calls': 1, 'best_of_n': False, 'heuristic_scoring': False, 'feedback': False, 'candidate_selection': False, 'orientation': False, 'scale': False, 'hidden_state_capture': False}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def modified_benchmark(original, condition):
    """The article change an -> a is the sole change besides adjective insertion."""
    result = copy.deepcopy(original)
    adjective = CONDITIONS[condition]
    task = original['task_description']
    if not re.fullmatch(r'This is an indoor [^\n]+\.', task):
        raise ValueError(f'Expected original sentence "This is an indoor <room>.": {task!r}')
    if adjective:
        result['task_description'] = task.replace('This is an indoor ', f'This is a {adjective} indoor ', 1)
    assert {k: v for k, v in result.items() if k != 'task_description'} == {k: v for k, v in original.items() if k != 'task_description'}
    return result


def select_benchmarks(root, categories):
    selected = []
    for category in categories:
        directory = Path(root) / category
        preferred = directory / f'{category}_1.json'
        candidates = sorted(directory.glob('*.json'))
        if not candidates:
            raise ValueError(f'No benchmark JSON in {directory}')
        path = preferred if preferred.is_file() else candidates[0]
        original = read_json(path)
        modified_benchmark(original, 'baseline')
        room = original['room_dimension']
        for key in ('room_dimension_x', 'room_dimension_y'):
            if not isinstance(room[key], (int, float)) or room[key] <= 0:
                raise ValueError(f'Invalid room dimension: {path}')
        if not isinstance(original['assets'], dict) or any(not isinstance(n, int) or n < 0 for n in original['assets'].values()):
            raise ValueError(f'Invalid asset counts: {path}')
        selected.append({'category': category, 'benchmark': path.stem, 'path': str(path.resolve()), 'sha256': digest(path), 'original': original, 'selection': 'preferred' if path == preferred else 'lexicographically first fallback'})
    return selected


def source_hashes():
    paths = [ROOT / 'scenegen/scenegen/Heuristic_wo_dspy.py', ROOT / 'probe_metrics/utils_copy_img.py', ROOT / 'probe_metrics/probe_capture.py', ROOT / 'probe_metrics/adv_l1_variance.py']
    paths += sorted(ROOT.glob('*qwen35_27b.py'))
    return {str(p.relative_to(ROOT)): digest(p) for p in paths}


def load_definitions(path, names, namespace=None):
    """Reuse selected local definitions without importing model/probing side effects.

    This compiles an explicit allowlist of AST definitions from trusted repository
    source, unchanged. No imports, main blocks, placement managers or probe classes
    are executed. Source hashes are captured in every experiment manifest.
    """
    tree = ast.parse(Path(path).read_text(), filename=str(path))
    nodes = []
    found = set()
    for node in tree.body:
        name = getattr(node, 'name', None)
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
        if name in names:
            nodes.append(node)
            found.add(name)
    if found != set(names):
        raise RuntimeError(f'Repository API changed: missing {set(names) - found} in {path}')
    scope = {'__builtins__': __builtins__, 'os': os, 're': re, 'json': json, 'Counter': Counter, 'Any': Any, 'Dict': Dict, 'List': List, 'Optional': Optional, 'Tuple': Tuple}
    scope.update(namespace or {})
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), scope)
    return SimpleNamespace(**{name: scope[name] for name in names}), scope


def pipeline_definitions():
    utils, _ = load_definitions(ROOT / 'probe_metrics/utils_copy_img.py', {'_parse_allowed_objects'})
    names = {'JSON_SYSTEM', 'SCENE_DESC_SCHEMA', 'OBJECT_CHOICE_SCHEMA', 'SCENE_OBJECT_SCHEMA', 'ATOMIC_TASK_SCHEMA', '_strip_fences', '_extract_balanced', 'parse_llm_json', 'coerce_obj_list', 'build_planning_prompt', 'build_object_placement_prompt', 'SceneDescription', 'ObjectChoice', 'PlanAtomicTasks'}
    return load_definitions(ROOT / 'scenegen/scenegen/Heuristic_wo_dspy.py', names, {'utils_copy_img': utils})


def geometry_definitions():
    names = {'EMPTY_SYMBOL', 'EGO_LABEL', '_CANON', '_ORD', '_corners', '_inter_area', '_real_objs', '_norm_label', 'room_extent', 'build_grid_strings', 'legality_overlap', 'out_of_bounds'}
    return load_definitions(ROOT / 'probe_metrics/probe_capture.py', names)[0]


def check_objects(objects):
    """Validate representation, without repairing, rejecting overlap or rescoring."""
    if not isinstance(objects, list):
        raise ValueError('Placement must contain an objects list')
    for obj in objects:
        if not isinstance(obj, dict) or not isinstance(obj.get('label'), str) or not obj['label']:
            raise ValueError('Object requires a nonempty label')
        for key, size in [('bbox', 4), ('position', 2)]:
            values = obj.get(key)
            if not isinstance(values, list) or len(values) != size or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
                raise ValueError(f'Invalid {key} for {obj["label"]}')
        if obj['bbox'][2] <= 0 or obj['bbox'][3] <= 0:
            raise ValueError(f'Nonpositive footprint for {obj["label"]}')
        if obj['label'] == 'ego_person':
            raise ValueError('Model must not emit ego_person')


def scene_labels(objects, room):
    geometry = geometry_definitions()
    extent = geometry.room_extent(room['room_dimension_x'], room['room_dimension_y'])
    occupancy, contested = geometry.build_grid_strings(objects, 5, 5, extent, 0.1, 'category')
    overlap, within, per_object = geometry.legality_overlap([], objects)
    return {'occupancy_str': occupancy, 'contested': contested, 'legality': int(not overlap), 'overlap_any': overlap, 'within_overlap': within, 'per_new_overlap': per_object, 'oob': geometry.out_of_bounds(objects, extent)}


def object_positions(objects):
    """Stable occurrence keys avoid collapsing duplicate unsuffixed model labels.

    The first occurrence uses the exact label; subsequent duplicate labels receive
    an analysis-only occurrence key. The saved model objects are never renamed.
    """
    counts = Counter()
    positions = {}
    for obj in objects:
        label = obj['label']
        counts[label] += 1
        positions[(label, counts[label])] = obj['position']
    return positions


def pair_metrics(a, b):
    A, B = a['scene']['objects'], b['scene']['objects']
    canon = lambda label: re.sub(r'_\d+$', '', label.lower().strip())
    ka, kb = {canon(o['label']) for o in A}, {canon(o['label']) for o in B}
    union = ka | kb
    pa, pb = object_positions(A), object_positions(B)
    shared = pa.keys() & pb.keys()
    shifts = [math.dist(pa[k], pb[k]) for k in shared]
    ga, gb = a['labels']['occupancy_str'], b['labels']['occupancy_str']
    if len(ga) != 25 or len(gb) != 25:
        raise ValueError('Expected repository 5x5 occupancy grid')
    return {'object_set_diff': 1 - len(ka & kb) / len(union) if union else 0.0, 'object_count_diff': abs(len(A) - len(B)), 'occupancy_hamming': sum(x != y for x, y in zip(ga, gb)) / len(ga), 'centroid_shift': statistics.mean(shifts) if shifts else None, 'legality_flip': int(a['labels']['legality'] != b['labels']['legality'])}


def load_runs(input_dir):
    root = Path(input_dir).resolve()
    manifest = read_json(root / 'results/experiment_manifest.json')
    if manifest['status'] not in ('complete', 'complete_with_failures'):
        raise ValueError('Generation manifest is not complete; refusing partial analysis')
    groups = defaultdict(dict)
    for entry in manifest['runs']:
        if entry.get('status') == 'failed':
            continue
        path = root / entry['output_path']
        if root not in path.resolve().parents:
            raise ValueError('Run output escapes input directory')
        result = read_json(path / 'run_qwen35_27b.json')
        for key in ('benchmark', 'category', 'condition', 'seed', 'model', 'task_description'):
            if result[key] != entry[key]:
                raise ValueError(f'Manifest mismatch: {path}: {key}')
        if result['refinement'] != REFINEMENT or tuple(c['stage'] for c in result['calls']) != STAGES:
            raise ValueError(f'Unexpected generation procedure: {path}')
        check_objects(result['scene']['objects'])
        if result['labels'] != scene_labels(result['scene']['objects'], result['benchmark_copy']['room_dimension']):
            raise ValueError(f'Geometry labels mismatch: {path}')
        key = (entry['category'], entry['condition'])
        if entry['seed'] in groups[key]:
            raise ValueError(f'Duplicate seed in {key}')
        groups[key][entry['seed']] = result
    for category in manifest['categories']:
        baseline = groups[(category, 'baseline')]
        if manifest['status'] == 'complete' and set(baseline) != set(manifest['seeds']):
            raise ValueError(f'Missing baseline seeds for {category}')
        for condition in manifest['conditions']:
            if manifest['status'] == 'complete' and set(groups[(category, condition)]) != set(baseline):
                raise ValueError(f'Missing condition seeds for {category}/{condition}')
            for seed, b in groups[(category, condition)].items():
                if seed not in baseline:
                    continue
                a = baseline[seed]
                for key in ('model', 'provider', 'generation_config', 'benchmark_sha256', 'runtime'):
                    if a[key] != b[key]:
                        raise ValueError(f'Uncontrolled paired configuration: {key}')
                if b['benchmark_copy'] != modified_benchmark(a['benchmark_copy'], condition):
                    raise ValueError('Intervention changed more than task_description')
                if [c['sampling_seed'] for c in a['calls']] != [c['sampling_seed'] for c in b['calls']] or [c['prompt_seed'] for c in a['calls']] != [c['prompt_seed'] for c in b['calls']]:
                    raise ValueError('Paired call seeds differ')
    return manifest, groups


def write_summary(output_dir, stem, rows, metadata, fieldnames=None):
    output_dir = Path(output_dir)
    write_json(output_dir / f'{stem}.json', {'metadata': metadata, 'rows': rows})
    with (output_dir / f'{stem}.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]) if rows else (fieldnames or []))
        writer.writeheader()
        writer.writerows(rows)


def finite_mean(values):
    values = [v for v in values if v is not None and math.isfinite(v)]
    return statistics.mean(values) if values else None


def distance_statistic(matrix, labels):
    between, within = [], []
    for i, j in itertools.combinations(range(len(labels)), 2):
        value = matrix[i][j]
        if value is not None:
            (between if labels[i] != labels[j] else within).append(value)
    b, w = finite_mean(between), finite_mean(within)
    return b - w if b is not None and w is not None else None


def permutation_test(matrix, n_pairs, paired=True):
    """Repository T and exact upper-tail enumeration, with optional seed blocks."""
    labels = [0] * n_pairs + [1] * n_pairs
    observed = distance_statistic(matrix, labels)
    if observed is None:
        return {'test_statistic': None, 'p_value': None, 'number_of_permutations': 0, 'min_achievable_p': None}
    stats = []
    if paired:
        assignments = ([*bits, *(1 - bit for bit in bits)] for bits in itertools.product((0, 1), repeat=n_pairs))
    else:
        assignments = ([int(i in indices) for i in range(2 * n_pairs)] for indices in itertools.combinations(range(2 * n_pairs), n_pairs))
    for assignment in assignments:
        value = distance_statistic(matrix, assignment)
        if value is not None:
            stats.append(value)
    return {'test_statistic': observed, 'p_value': sum(s >= observed - 1e-12 for s in stats) / len(stats), 'number_of_permutations': len(stats), 'min_achievable_p': min(1.0, 2 / len(stats))}
