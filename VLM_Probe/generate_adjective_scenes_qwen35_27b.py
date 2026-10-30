"""Isolated initial-generation adjective experiment. Default mode only prints plan."""
from __future__ import annotations

import argparse
import contextlib
import copy
import importlib.metadata
import io
import json
import os
import platform
import random
import re
import sys
import tempfile
from pathlib import Path

import adjective_common_qwen35_27b as C


class Backend:
    def __init__(self, args):
        self.args = args
        self.runtime = {'python': platform.python_version(), 'mock': args.mock}
        if args.mock:
            self.runtime['backend'] = 'synthetic fixture, NOT Qwen inference'
            return
        from api_backend_qwen35_27b import ApiBackend
        self.api = ApiBackend(args)
        self.runtime.update(self.api.runtime)

    def seed(self, seed):
        random.seed(seed)

    def complete(self, prompt, stage, seed, benchmark, system):
        self.seed(seed)
        self.last_output = None
        if self.args.mock:
            # Intentionally synthetic. Stable fixtures exercise the four-stage
            # pipeline and paired analyses; never used as scientific results.
            if stage == 'description':
                return json.dumps({'new_scene_description': 'Synthetic fixture scene.', 'concise_scene_name': 'fixture', 'scene_area': 'indoor'})
            if stage == 'object_choice':
                return json.dumps({'bulk_objects': list(benchmark['assets']), 'ornament_objects': []})
            if stage == 'planning':
                return json.dumps({'atomic_tasks': [{'id': '1', 'execution_prompt': 'Place the requested assets.'}]})
            objects = []
            for label, count in benchmark['assets'].items():
                for _ in range(count):
                    x, y = random.uniform(-1, 1), random.uniform(0, 2)
                    objects.append({'label': label, 'object_type': 'bulk', 'anchor': None, 'bbox': [x, y, 0.4, 0.4], 'position': [x + 0.2, y + 0.2], 'rotation_angle': 0.0, 'description': 'Synthetic fixture', 'concise_description': 'Synthetic fixture'})
            return json.dumps({'objects': objects})
        if self.api is not None:
            try:
                return self.api.complete(prompt, seed, system)
            finally:
                self.last_output = self.api.last_output


def run_initial(backend, spec, condition, seed, directory, config):
    directory.mkdir(parents=True, exist_ok=False)
    benchmark = C.modified_benchmark(spec['original'], condition)
    definitions, scope = C.pipeline_definitions()
    calls = []
    prompt_counter = 0
    stage = 'description'

    def prompt_seed():
        nonlocal prompt_counter
        prompt_counter += 1
        return (seed * 7919 + prompt_counter * 104729) % 10000001

    scope['utils_copy_img'].generate_random_seed = prompt_seed

    def complete(prompt, images=None, system=None, temperature=0.5, max_tokens=None, force_json=True):
        if images:
            raise ValueError('Images are forbidden in this experiment')
        if len(calls) >= 4:
            raise RuntimeError('More than four initial-generation calls attempted')
        number = len(calls) + 1
        sampling_seed = seed + number
        record = {'index': number - 1, 'stage': stage, 'prompt': prompt, 'system': system or definitions.JSON_SYSTEM, 'prompt_seed': int(re.search(r'Random seed:\s*(\d+)', prompt).group(1)), 'sampling_seed': sampling_seed}
        calls.append(record)
        try:
            record['output_text'] = backend.complete(prompt, stage, sampling_seed, benchmark, record['system'])
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
    backend.seed(seed)
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
            result = {'category': spec['category'], 'benchmark': spec['benchmark'], 'benchmark_path': spec['path'], 'benchmark_sha256': spec['sha256'], 'benchmark_copy': benchmark, 'condition': condition, 'adjective': C.CONDITIONS[condition], 'seed': seed, 'model': backend.args.model_path, 'provider': getattr(backend.args, 'api_provider', 'fixture'), 'task_description': benchmark['task_description'], 'generation_config': config, 'refinement': C.REFINEMENT, 'runtime': backend.runtime, 'scene': scene, 'labels': labels, 'calls': calls, 'initial_generation': True}
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


def run_payload(result):
    payload = {key: result[key] for key in ('scene', 'labels', 'benchmark_copy', 'generation_config', 'refinement', 'runtime')}
    # Request IDs, timing and usage are not generated scene content. Keep request
    # parameters + server identity in replay equality, excluding volatile IDs.
    payload['calls'] = [{key: value for key, value in call.items() if key != 'api_response'} for call in result['calls']]
    payload['api_controls'] = [{key: call.get('api_response', {}).get(key) for key in ('request', 'response_model', 'response_provider', 'system_fingerprint')} for call in result['calls']]
    return payload


def validate(backend, selected, seeds, conditions, root, config, identity):
    """One uncached identical-request pair, then seven classroom initial scenes."""
    spec = next((s for s in selected if s['category'] == 'classroom'), selected[0])
    validation = root / 'validation_qwen35_27b'
    validation.mkdir(parents=True, exist_ok=True)
    definition, _ = C.pipeline_definitions()
    benchmark = spec['original']
    prompt = (f'Instruction: {benchmark["task_description"]}\n'
              f'Room dimensions: {json.dumps(benchmark["room_dimension"], sort_keys=True)}\n'
              f'Required assets: {json.dumps(benchmark["assets"], sort_keys=True)}\n'
              'Return one initial layout with all requested assets and no ego_person.\n'
              + definition.SCENE_OBJECT_SCHEMA)
    report = {'passed': False, 'mock': backend.args.mock, 'identity': identity,
              'category': spec['category'], 'seed': seeds[0], 'pairs': [],
              'replay_passed': False, 'classroom_validation_passed': False,
              'runtime': backend.runtime, 'refinement': C.REFINEMENT,
              'scope': 'One actual uncached identical-request pair; exact final content equality. No cross-hardware/time claim.'}
    C.write_json(validation / 'validation_qwen35_27b.json', report)
    try:
        for pair in range(1):
            pair_records = []
            for repeat in range(2):
                record = {'prompt': prompt, 'system': definition.JSON_SYSTEM, 'seed': seeds[0] + 4}
                try:
                    record['content'] = backend.complete(prompt, 'placement', seeds[0] + 4, benchmark, definition.JSON_SYSTEM)
                finally:
                    if getattr(backend, 'api', None) is not None:
                        record['api_response'] = backend.api.last_response
                    C.write_json(validation / f'api_replay_pair_{pair}_request_{repeat}_qwen35_27b.json', record)
                pair_records.append(record)
            A, B = pair_records
            identical_requests = (A.get('api_response', {}).get('request') == B.get('api_response', {}).get('request')
                                  and A['prompt'] == B['prompt'] and A['system'] == B['system'] and A['seed'] == B['seed'])
            passed = identical_requests and A['content'] == B['content']
            report['pairs'].append({'pair': pair, 'identical_requests': identical_requests,
                                    'content_identical': A['content'] == B['content'], 'passed': passed,
                                    'first_content_sha256': C.fingerprint(A['content']), 'second_content_sha256': C.fingerprint(B['content'])})
            report['api_seed_replay_deterministic'] = all(p['passed'] for p in report['pairs'])
            report['replay_passed'] = report['api_seed_replay_deterministic']
            C.write_json(validation / 'validation_qwen35_27b.json', report)
            if not identical_requests:
                raise RuntimeError('Replay requests differed; configuration invalid')
        report['api_seed_replay_deterministic'] = all(p['passed'] for p in report['pairs'])
        report['replay_passed'] = report['api_seed_replay_deterministic']
        C.write_json(validation / 'validation_qwen35_27b.json', report)
        for condition in conditions:
            result = run_initial(backend, spec, condition, seeds[0], validation / 'classroom' / condition / f'seed_{seeds[0]}', config)
            if result['benchmark_copy'] != C.modified_benchmark(spec['original'], condition):
                raise RuntimeError('Classroom intervention altered original data')
            if tuple(call['stage'] for call in result['calls']) != C.STAGES or result['refinement'] != C.REFINEMENT:
                raise RuntimeError('Classroom stages/refinement differ from intended procedure')
        report['classroom_validation_passed'] = report['passed'] = True
        C.write_json(validation / 'validation_qwen35_27b.json', report)
        print(f'API replay and classroom validation passed; fixture={backend.args.mock}', flush=True)
        return report
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
        C.write_json(validation / 'validation_qwen35_27b.json', report)
        raise


def cleanup_outputs(root):
    """Delete only named adjective outputs inside this repository's output parent."""
    import shutil
    parent = C.OUTPUT_ROOT.parent.resolve()
    root = Path(root).absolute()
    if root.parent.resolve() != parent or root.is_symlink() or not root.name.startswith('qwen35_27b_adjective_probe'):
        raise ValueError('Cleanup is restricted to non-linked qwen35_27b_adjective_probe* directories in VLM_Probe/outputs')
    for path in parent.glob('qwen35_27b_adjective_probe*'):
        if path.is_symlink():
            raise ValueError(f'Refusing linked experiment output: {path}')
        if path.is_dir():
            shutil.rmtree(path)
            print(f'Cleared adjective output: {path}', flush=True)


def record_unavailable_analysis(root, manifest):
    """Explicitly empty, non-scientific reports when API availability or classroom validation blocks execution."""
    reason = manifest.get('error', 'Validation did not pass')
    metadata = {'status': 'not_run', 'reason': reason, 'mock': manifest['mock'], 'experiment_identity': manifest['identity'], 'scientific_results': False}
    C.write_summary(root / 'results', 'behavioral_summary', [], metadata,
                    ['benchmark', 'condition', 'seed', *C.METRICS, 'first_divergent_call', 'structural_divergence', 'placement_changed'])
    C.write_summary(root / 'results', 'variance_summary', [], metadata,
                    ['benchmark', 'condition', 'metric', 'within_baseline', 'within_adjective', 'between', 'test_statistic', 'p_value', 'min_achievable_p', 'test_type', 'verdict'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=('api',), default='api')
    parser.add_argument('--api-base', default='https://router.huggingface.co/v1')
    parser.add_argument('--api-provider', default='deepinfra')
    parser.add_argument('--api-timeout', type=float, default=180)
    parser.add_argument('--api-retries', type=int, default=2)
    parser.add_argument('--model-path', default=C.MODEL)
    parser.add_argument('--benchmark-root', type=Path, default=C.BENCHMARK_ROOT)
    parser.add_argument('--output-root', type=Path, default=C.OUTPUT_ROOT)
    parser.add_argument('--seeds', type=int, nargs='+')
    parser.add_argument('--num-seeds', type=int, default=5)
    parser.add_argument('--categories', nargs='+', choices=C.CATEGORIES, default=list(C.CATEGORIES))
    parser.add_argument('--conditions', nargs='+', choices=C.CONDITIONS, default=list(C.CONDITIONS))
    parser.add_argument('--mode', choices=('dry-run', 'validate', 'full'), default='dry-run')
    parser.add_argument('--cleanup', action='store_true', help="Delete only this repository's old adjective outputs before starting")
    parser.add_argument('--mock', action='store_true', help='Tests only; never a final experiment')
    parser.add_argument('--decode', choices=('sample', 'greedy'), default='sample')
    parser.add_argument('--temperature', type=float, default=0.5)
    parser.add_argument('--top-p', type=float, default=0.9)
    parser.add_argument('--top-k', type=int, default=20)
    parser.add_argument('--max-tokens', type=int, default=8192)
    args = parser.parse_args(argv)
    seeds = args.seeds if args.seeds is not None else [1234 + i * 10000 for i in range(args.num_seeds)]
    if args.model_path != C.MODEL or args.api_provider != 'deepinfra' or args.api_base.rstrip('/') != 'https://router.huggingface.co/v1':
        parser.error('Exactly Qwen/Qwen3.5-27B through Hugging Face pinned to DeepInfra is required')
    if not seeds or any(s < 0 or s >= 2**63 - 4 for s in seeds) or len(set(seeds)) != len(seeds):
        parser.error('Seeds must be unique and nonnegative')
    if len(set(args.categories)) != len(args.categories) or len(set(args.conditions)) != len(args.conditions) or 'baseline' not in args.conditions:
        parser.error('Categories/conditions must be unique and baseline is required')
    if args.temperature <= 0 or not 0 < args.top_p <= 1 or args.top_k <= 0 or args.max_tokens <= 0 or args.api_timeout <= 0 or not 0 <= args.api_retries <= 5:
        parser.error('Invalid generation or retry configuration')
    if args.mode == 'full' and not args.mock and (args.categories != list(C.CATEGORIES) or args.conditions != list(C.CONDITIONS) or seeds != list(C.SEEDS)):
        parser.error('The final real experiment requires all seven categories/conditions and the five fixed seeds')
    if 'qwen35_27b' not in args.output_root.name or args.mock and 'mock' not in str(args.output_root).lower():
        parser.error('Output name must contain qwen35_27b; fixtures must also contain mock')
    if args.cleanup:
        cleanup_outputs(args.output_root)
    selected = C.select_benchmarks(args.benchmark_root, args.categories)
    config = {'model': C.MODEL, 'routed_model': 'Qwen/Qwen3.5-27B:deepinfra', 'backend': 'huggingface',
              'api_base': args.api_base.rstrip('/'), 'api_method': 'POST /chat/completions', 'provider': args.api_provider,
              'decode': args.decode, 'temperature': args.temperature if args.decode == 'sample' else 0.0,
              'top_p': args.top_p, 'top_k': args.top_k, 'max_tokens': args.max_tokens,
              'frequency_penalty': 0.0, 'presence_penalty': 0.0, 'repetition_penalty': 1.0,
              'stop': [], 'reasoning_effort': 'none', 'response_format': {'type': 'json_object'},
              'n': 1, 'stream': False, 'provider_fallbacks': False, 
              'prompt_seed_rule': '(seed*7919 + call_number*104729) % 10000001',
              'sampling_seed_rule': 'seed + call_number', 'maximum_retries': args.api_retries,
              'grid': {'rows': 5, 'cols': 5, 'min_area_frac': 0.1, 'label_mode': 'category'},
              'parameter_basis': 'Preserve existing SceneCritic/scenegen temperature .5 and top_p .9; explicit checkpoint top_k=20; neutral penalties'}
    plan = {'model': C.MODEL, 'provider': args.api_provider, 'api_endpoint': config['api_base'] + '/chat/completions', 'api_method': 'POST', 'generation_config': config, 'refinement': C.REFINEMENT, 'placement_call_count': 1, 'mock': args.mock, 'categories': args.categories, 'conditions': args.conditions, 'seeds': seeds, 'benchmarks': selected, 'sources': C.source_hashes()}
    identity = C.fingerprint(plan)
    for spec in selected:
        print(f'\nSelected: {spec["path"]} ({spec["selection"]})', flush=True)
        for condition in args.conditions:
            print(f'  {condition}: {C.modified_benchmark(spec["original"], condition)["task_description"]}')
    print('Seeds:', seeds, '\nGeneration:', json.dumps(config, sort_keys=True), '\nRefinement:', C.REFINEMENT, flush=True)
    root = args.output_root.resolve()
    manifest_path = root / 'results/experiment_manifest.json'
    old = C.read_json(manifest_path) if manifest_path.exists() else {}
    if old and (old.get('identity') != identity or old.get('status') not in ('planned', 'validated', 'blocked_missing_credentials')):
        raise ValueError('Existing results will not be overwritten; use --cleanup for the authorized final experiment')
    runs = [{'category': spec['category'], 'benchmark': spec['benchmark'], 'benchmark_path': spec['path'], 'benchmark_sha256': spec['sha256'], 'condition': condition, 'adjective': C.CONDITIONS[condition], 'seed': seed, 'model': C.MODEL, 'provider': args.api_provider, 'task_description': C.modified_benchmark(spec['original'], condition)['task_description'], 'output_path': f'{spec["category"]}/{condition}/seed_{seed}', 'status': 'not_attempted'} for spec in selected for seed in seeds for condition in args.conditions]
    manifest = dict(plan, authentication='HF_TOKEN (never recorded)', output_locations={'behavioral': 'results/behavioral_summary.csv', 'variance': 'results/variance_summary.csv', 'validation': 'validation_qwen35_27b'}, identity=identity, status='planned', runs=runs, total_requested_generations=len(runs), successful_generations=0, failed_generations=0, attempted_generations=0, not_attempted_generations=len(runs), retry_count=0, api_attempt_count=0, deterministic_replay={'status': 'not_run', 'passed': False})
    C.write_json(manifest_path, manifest)
    if args.mode == 'dry-run':
        print(f'Dry plan: {len(runs)} configurations; no inference', flush=True)
        return
    backend = None
    try:
        if not args.mock:
            from api_backend_qwen35_27b import inspect_provider, read_token
            read_token()
            endpoint, catalog = inspect_provider(args.api_provider, args.api_timeout)
            manifest['provider_endpoint'] = endpoint
            C.write_json(root / 'results/provider_catalog_qwen35_27b.json', catalog)
            C.write_json(manifest_path, manifest)
        backend = Backend(args)
        manifest['runtime'] = backend.runtime
        report_path = root / 'validation_qwen35_27b/validation_qwen35_27b.json'
        report = C.read_json(report_path) if report_path.exists() else {}
        if args.mode == 'validate' or not (report.get('passed') and report.get('identity') == identity and report.get('runtime') == backend.runtime):
            report = validate(backend, selected, seeds, args.conditions, root, config, identity)
        manifest['deterministic_replay'] = report
        manifest['api_seed_replay_deterministic'] = report['api_seed_replay_deterministic']
        if args.mode == 'validate':
            manifest['status'] = 'validated'
            C.write_json(manifest_path, manifest)
            return
        manifest['status'] = 'running'
        by_category = {spec['category']: spec for spec in selected}
        for entry in runs:
            print(f'Generating {entry["category"]}/{entry["condition"]}/seed_{entry["seed"]}', flush=True)
            manifest['attempted_generations'] += 1
            try:
                result = run_initial(backend, by_category[entry['category']], entry['condition'], entry['seed'], root / entry['output_path'], config)
                entry['status'] = 'success'
                manifest['successful_generations'] += 1
            except Exception as error:
                entry['status'] = 'failed'
                entry['error'] = f'{type(error).__name__}: {error}'
                manifest['failed_generations'] += 1
                C.write_json(root / entry['output_path'] / 'failure_qwen35_27b.json', entry)
                print(f'Run failed: {entry["error"]}', flush=True)
            manifest['not_attempted_generations'] = len(runs) - manifest['attempted_generations']
            if getattr(backend, 'api', None):
                manifest['retry_count'] = backend.api.retry_count
                manifest['api_attempt_count'] = backend.api.attempt_count
            C.write_json(manifest_path, manifest)
        for spec in selected:
            if C.digest(spec['path']) != spec['sha256']:
                raise RuntimeError('Original benchmark changed during experiment')
        manifest['status'] = 'complete' if not manifest['failed_generations'] else 'complete_with_failures'
        C.write_json(manifest_path, manifest)
        print(f'Completed: {manifest["successful_generations"]}/{len(runs)} scenes, failures={manifest["failed_generations"]}, retries={manifest["retry_count"]}', flush=True)
    except Exception as error:
        manifest['status'] = 'blocked_missing_credentials' if 'Missing HF_TOKEN' in str(error) else 'blocked_api_credits' if getattr(error, 'status', None) == 402 else 'validation_failed' if manifest['status'] == 'planned' else 'failed'
        manifest['error'] = f'{type(error).__name__}: {error}'
        report_path = root / 'validation_qwen35_27b/validation_qwen35_27b.json'
        if report_path.exists():
            manifest['deterministic_replay'] = C.read_json(report_path)
            manifest['api_seed_replay_deterministic'] = manifest['deterministic_replay'].get('api_seed_replay_deterministic')
        if backend is not None and getattr(backend, 'api', None):
            manifest['retry_count'] = backend.api.retry_count
            manifest['api_attempt_count'] = backend.api.attempt_count
        C.write_json(manifest_path, manifest)
        record_unavailable_analysis(root, manifest)
        raise


if __name__ == '__main__':
    try:
        main()
    except (ValueError, RuntimeError) as error:
        print(f'ERROR: {error}', file=sys.stderr)
        sys.exit(1)
