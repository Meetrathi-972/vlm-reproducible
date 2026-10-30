"""Exactly three bedroom benchmarks x seven conditions, unseeded and initial-only."""
import argparse
import contextlib
import copy
import io
import json
import re
from datetime import datetime, timezone
from types import SimpleNamespace
from pathlib import Path
import adjective_common_qwen35_27b as C
from api_backend_qwen35_27b import ApiBackend, API_BASE
from adv_l1_behavioral_qwen35_27b import compare

class UnseededApi(ApiBackend):
    def payload(self, prompt, seed, system):
        result = super().payload(prompt, None, system)
        result.pop('seed')
        return result

class Backend:
    def __init__(self):
        self.args = SimpleNamespace(mock=False, model_path=C.MODEL, api_base=API_BASE,
            api_provider='deepinfra', api_timeout=180, api_retries=0,
            decode='sample', temperature=.5, top_p=.9, top_k=20, max_tokens=8192)
        self.api = UnseededApi(self.args)
        self.runtime = self.api.runtime
        self.last_output = None
    def complete(self, prompt, stage, unused, benchmark, system):
        try:
            return self.api.complete(prompt, None, system)
        finally:
            self.last_output = self.api.last_output

def run_unseeded_initial(backend, spec, condition, directory, config):
    directory.mkdir(parents=True, exist_ok=False)
    benchmark = C.modified_benchmark(spec['original'], condition)
    definitions, scope = C.pipeline_definitions()
    calls = []
    prompt_counter = 0
    stage = 'description'

    def prompt_seed():
        nonlocal prompt_counter
        prompt_counter += 1
        return 0  # Internal adapter placeholder; removed from every transmitted prompt.

    scope['utils_copy_img'].generate_random_seed = prompt_seed

    def complete(prompt, images=None, system=None, temperature=0.5, max_tokens=None, force_json=True):
        if images:
            raise ValueError('Images are forbidden in this experiment')
        if len(calls) >= 4:
            raise RuntimeError('More than four initial-generation calls attempted')
        number = len(calls) + 1
        prompt = re.sub(r"(?m)^Random seed:[ \t]*\d+[ \t]*\n", "", prompt)
        record = {'index': number - 1, 'stage': stage, 'prompt': prompt, 'system': system or definitions.JSON_SYSTEM}
        calls.append(record)
        try:
            record['output_text'] = backend.complete(prompt, stage, None, benchmark, record['system'])
        except Exception as error:
            record['error'] = f'{type(error).__name__}: {error}'
            if getattr(backend, 'last_output', None) is not None:
                record['output_text'] = backend.last_output
            C.write_json(directory / 'calls_qwen35_27b.json', calls)
            raise
        finally:
            if getattr(backend, 'api', None) is not None and backend.api.last_response is not None:
                record['api_response'] = backend.api.last_response
                C.write_json(directory / 'calls_qwen35_27b.json', calls)
        C.write_json(directory / 'calls_qwen35_27b.json', calls)
        return record['output_text']

    scope['llm_complete'] = complete
    C.write_json(directory / 'benchmark_copy_qwen35_27b.json', benchmark)
    log = io.StringIO()
    try:
        with contextlib.redirect_stdout(log):
            env_note = f'Instruction: {benchmark["task_description"]}'
            description, name, area = definitions.SceneDescription()(env_note, 'No previous scene description!')
            stage = 'object_choice'
            # Preserve this existing (unused) stage and RNG/call ordering.
            definitions.ObjectChoice().suggest_objects(env_note, description)
            stage = 'planning'
            assets = ', '.join(f'{count} {label}' for label, count in benchmark['assets'].items())
            room = benchmark['room_dimension']
            plan, region, consistency, movement, _ = definitions.PlanAtomicTasks(str(directory)).forward((0, 0), env_note, description, assets, [], room['room_dimension_x'], room['room_dimension_y'], 0, 0)
            stage = 'placement'
            plan_text = '\n'.join(task.get('execution_prompt', '') for task in plan['atomic_tasks'] if task.get('execution_prompt'))
            instruction = definitions.build_object_placement_prompt(plan_text, (0, 0), room['room_dimension_x'], room['room_dimension_y'], consistency, movement, env_note, region)
            # Exactly the initial _generate_placement prompt, with empty feedback.
            prompt = '\n\n'.join([f'Random seed: {prompt_seed()}', instruction, definitions.SCENE_OBJECT_SCHEMA])
            raw = complete(prompt)
            parsed = definitions.parse_llm_json(raw)
            (directory / 'placement_raw_qwen35_27b.txt').write_text(raw)
            C.write_json(directory / 'initial_model_output_qwen35_27b.json', parsed)
            objects = definitions.coerce_obj_list(parsed)
            if not (isinstance(parsed, list) or isinstance(parsed, dict) and any(isinstance(parsed.get(k), list) for k in ('objects', 'placement_json', 'updated_scene_json')) or isinstance(parsed, dict) and isinstance(parsed.get('scene'), dict) and isinstance(parsed['scene'].get('objects'), list)):
                raise ValueError('No recognized placement objects array; refusing silent empty scene')
            C.check_objects(objects)
            scene = {'objects': objects, 'Environmental_note': env_note, 'scene_description': description, 'Environment Name': name, 'Scene Area': area}
            labels = C.scene_labels(objects, room)  # analysis labels only, never scoring
            result = {'category': spec['category'], 'benchmark': spec['benchmark'], 'benchmark_path': spec['path'], 'benchmark_sha256': spec['sha256'], 'benchmark_copy': benchmark, 'condition': condition, 'adjective': C.CONDITIONS[condition], 'model': backend.args.model_path, 'provider': getattr(backend.args, 'api_provider', 'fixture'), 'task_description': benchmark['task_description'], 'generation_config': config, 'refinement': C.REFINEMENT, 'runtime': backend.runtime, 'scene': scene, 'labels': labels, 'calls': calls, 'initial_generation': True}
            if tuple(c['stage'] for c in calls) != C.STAGES or prompt_counter != 4:
                raise RuntimeError('Unexpected stage or prompt seed sequence')
            C.write_json(directory / 'run_qwen35_27b.json', result)
            # Compatibility envelope adds only the ego metadata; model objects
            # remain byte-for-value identical to the direct parsed placement.
            compatible = copy.deepcopy(scene)
            compatible['objects'].append({'label': 'ego_person', 'object_type': 'bulk', 'anchor': None, 'label_index': 'ego_person', 'position': [0, 0], 'bbox': [0, 0, 0, 0], 'description': 'Ego agent position', 'rotation_angle': 0.0})
            C.write_json(directory / 'scene_pos_0_0.json', compatible)
        return result
    finally:
        (directory / 'stdout_qwen35_27b.log').write_text(log.getvalue())
        if C.digest(spec['path']) != spec['sha256']:
            raise RuntimeError(f'Original benchmark changed during run: {spec["path"]}')


def now():
    return datetime.now(timezone.utc).isoformat()

def analyze(root, manifest):
    rows = []
    for spec in manifest['benchmarks']:
        paths = {condition: root / spec['benchmark'] / condition / 'run_qwen35_27b.json' for condition in C.CONDITIONS}
        for condition in list(C.CONDITIONS)[1:]:
            row = dict(benchmark=spec['benchmark'], condition=condition, baseline_condition='baseline')
            if paths['baseline'].exists() and paths[condition].exists():
                row.update(status='success', **compare(C.read_json(paths['baseline']), C.read_json(paths[condition])))
            else:
                row.update(status='unavailable_failed_generation', **{k: None for k in (*C.METRICS, 'first_divergent_call', 'structural_divergence', 'placement_changed')})
            rows.append(row)
    metadata = dict(design='One output per benchmark and condition; no seeds or replicates', scientific_results=any(r['status']=='success' for r in rows), model=C.MODEL, provider='deepinfra', inference_limitation='Previously observed nondeterministic API; single outputs cannot isolate adjective effects from generation randomness', first_divergent_call_indexing='zero-based')
    C.write_summary(root / 'results', 'behavioral_summary', rows, metadata)
    aggregate = []
    for condition in list(C.CONDITIONS)[1:]:
        available = [r for r in rows if r['condition']==condition and r['status']=='success']
        aggregate.append(dict(condition=condition, compared_benchmarks=len(available),
            **{metric+'_mean': C.finite_mean([r[metric] for r in available]) for metric in C.METRICS},
            placement_changed_count=sum(r['placement_changed'] for r in available),
            structural_divergence_count=sum(r['structural_divergence'] for r in available),
            first_divergent_calls={r['benchmark']: r['first_divergent_call'] for r in available}))
    C.write_summary(root / 'results', 'behavioral_aggregate', aggregate, metadata)
    return rows

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark', choices=('bedroom_1.json', 'bedroom_2.json', 'bedroom_3.json'))
    parser.add_argument('--output-root', type=Path, default=C.OUTPUT_ROOT / 'bedroom_only')
    args = parser.parse_args()
    paths = [C.BENCHMARK_ROOT / 'bedroom' / args.benchmark] if args.benchmark else sorted((C.BENCHMARK_ROOT / 'bedroom').glob('*.json'))
    if len(paths)!=(1 if args.benchmark else 3) or not all(path.is_file() for path in paths):
        raise RuntimeError('Requested bedroom benchmarks are missing')
    selected = []
    for path in paths:
        print('Selected:', path.resolve(), flush=True)
        selected.append(dict(category='bedroom', benchmark=path.stem, path=str(path.resolve()), sha256=C.digest(path), original=C.read_json(path)))
    root = args.output_root
    if root.exists():
        raise RuntimeError('Refusing to overwrite existing bedroom_only results')
    root.mkdir(parents=True)
    entries = [dict(benchmark=spec['benchmark'], benchmark_path=spec['path'], condition=condition,
        adjective=C.CONDITIONS[condition], original_task_description=spec['original']['task_description'],
        modified_task_description=C.modified_benchmark(spec['original'],condition)['task_description'],
        output_path=f"{spec['benchmark']}/{condition}", status='not_attempted') for spec in selected for condition in C.CONDITIONS]
    manifest = dict(status='running', model=C.MODEL, provider='deepinfra', authentication='Existing HF_TOKEN, never saved',
        benchmarks=selected, conditions=list(C.CONDITIONS), seeds=None, replicates=0,
        requested_generations=len(entries), successful_generations=0, failed_generations=0, attempted_generations=0,
        refinement=C.REFINEMENT, placement_call_count=1, started_at=now(), runs=entries,
        source_hashes=C.source_hashes(), retries=0)
    manifest_path=root/'results/experiment_manifest.json'
    C.write_json(manifest_path,manifest)
    backend = Backend()
    config = {k:v for k,v in backend.api.payload('',None,'').items() if k!='messages'}
    manifest['generation_parameters']=config
    C.write_json(root/'results/provider_catalog_qwen35_27b.json',backend.api.catalog)
    by_name={spec['benchmark']:spec for spec in selected}
    for entry in entries:
        print('Generating:',entry['benchmark'],entry['condition'],flush=True)
        entry['started_at']=now()
        manifest['attempted_generations']+=1
        try:
            result=run_unseeded_initial(backend,by_name[entry['benchmark']],entry['condition'],root/entry['output_path'],config)
            result.update(original_task_description=entry['original_task_description'], modified_task_description=entry['modified_task_description'], timestamp=now(), status='success')
            C.write_json(root/entry['output_path']/'run_qwen35_27b.json',result)
            entry['status']='success'
            manifest['successful_generations']+=1
        except Exception as error:
            entry.update(status='failed',error=f'{type(error).__name__}: {error}')
            manifest['failed_generations']+=1
            C.write_json(root/entry['output_path']/'failure_qwen35_27b.json',entry)
            print('Failed:',entry['error'],flush=True)
        entry['finished_at']=now()
        manifest['api_attempt_count']=backend.api.attempt_count
        C.write_json(manifest_path,manifest)
    manifest.update(status='complete_with_failures' if manifest['failed_generations'] else 'complete',finished_at=now(),
        benchmarks_unchanged=all(C.digest(spec['path'])==spec['sha256'] for spec in selected))
    rows=analyze(root,manifest)
    manifest['behavioral_comparisons']=len(rows)
    C.write_json(manifest_path,manifest)
    print('Completed:',manifest['successful_generations'],f'/{len(entries)}; failed:',manifest['failed_generations'],flush=True)
    print('Results:',root/'results',flush=True)

if __name__=='__main__':
    main()
