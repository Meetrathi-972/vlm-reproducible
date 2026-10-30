"""Classroom-only experiment, gated by ten byte-identical greedy API replays."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import adjective_common_qwen35_27b as C
import generate_adjective_scenes_qwen35_27b as G
from api_backend_qwen35_27b import API_BASE


def main():
    root = C.OUTPUT_ROOT.parent / 'qwen35_27b_classroom_greedy'
    if root.exists():
        raise RuntimeError(f'Refusing to overwrite existing test: {root}')
    root.mkdir(parents=True)
    args = SimpleNamespace(mock=False, model_path=C.MODEL, api_base=API_BASE,
        api_provider='deepinfra', api_timeout=180, api_retries=2,
        decode='greedy', temperature=0.0, top_p=1.0, top_k=20, max_tokens=8192)
    spec = C.select_benchmarks(C.BENCHMARK_ROOT, ['classroom'])[0]
    definition, _ = C.pipeline_definitions()
    original = spec['original']
    prompt = (f'Instruction: {original["task_description"]}\n'
        f'Room dimensions: {json.dumps(original["room_dimension"], sort_keys=True)}\n'
        f'Required assets: {json.dumps(original["assets"], sort_keys=True)}\n'
        'Return one initial layout with all requested assets and no ego_person.\n'
        + definition.SCENE_OBJECT_SCHEMA)
    report = {'status': 'replay_pending', 'model': C.MODEL, 'provider': 'deepinfra',
        'benchmark': spec['benchmark'], 'benchmark_sha256': spec['sha256'],
        'replicate_seed': 1234, 'replay_seed': 1234, 'requested_replays': 10,
        'completed_replays': 0, 'all_ten_outputs_identical': None,
        'identical_requests': None, 'runs_differing_from_first': [],
        'output_groups': [], 'replays': [], 'intended_generations': 35,
        'attempted_generations': 0, 'successful_generations': 0, 'failed_generations': 0,
        'authentication': 'Existing HF_TOKEN, never saved', 'refinement': C.REFINEMENT,
        'replay_scope': 'Ten identical baseline classroom initial-layout requests; compare full content UTF-8 bytes and complete message including any reasoning. Response IDs, timing and usage are metadata, not generation output.'}
    report_path = root / 'replay_report_qwen35_27b.json'
    C.write_json(report_path, report)
    backend = None
    try:
        backend = G.Backend(args)
        api = backend.api
        payload = api.payload(prompt, 1234, definition.JSON_SYSTEM)
        report['api_endpoint'] = API_BASE + '/chat/completions'
        report['exact_request'] = payload
        C.write_json(root / 'provider_catalog_qwen35_27b.json', api.catalog)
        C.write_json(report_path, report)
        groups = {}
        message_groups = {}
        request_bytes = json.dumps(payload, sort_keys=True, allow_nan=False).encode('utf-8')
        for number in range(1, 11):
            print(f'Greedy baseline replay {number}/10', flush=True)
            record = {'run': number}
            try:
                api.complete(prompt, 1234, definition.JSON_SYSTEM)
                message = api.last_response['raw_response']['choices'][0]['message']
                content = message['content']
                encoded = content.encode('utf-8')
                # Store returned content exactly, before any thinking-tag removal.
                (root / f'replay_{number:02d}_content.txt').write_bytes(encoded)
                record['content_sha256'] = hashlib.sha256(encoded).hexdigest()
                record['complete_message_sha256'] = C.fingerprint(message)
                sent = json.dumps(api.last_response['request'], sort_keys=True, allow_nan=False).encode('utf-8')
                record['request_identical'] = sent == request_bytes
                groups.setdefault(record['content_sha256'], []).append(number)
                message_groups.setdefault(record['complete_message_sha256'], []).append(number)
                report['completed_replays'] += 1
                report['replays'].append(record)
                report['observed_outputs_identical'] = len(groups) == 1 and len(message_groups) == 1
                if not report['observed_outputs_identical']:
                    report['difference_origin'] = 'Identical serialized requests; differences arose on the remote model/API side' if all(r['request_identical'] for r in report['replays']) else 'Request changed; output differences cannot be attributed to server nondeterminism'
                report['output_groups'] = list(groups.values())
                report['complete_message_groups'] = list(message_groups.values())
                report['identical_requests'] = all(r['request_identical'] for r in report['replays'])
                first_hash = report['replays'][0]['content_sha256']
                report['runs_differing_from_first'] = [r['run'] for r in report['replays'] if r['content_sha256'] != first_hash]
                C.write_json(report_path, report)
            finally:
                record['api_response'] = copy.deepcopy(api.last_response)
                C.write_json(root / f'replay_{number:02d}_api.json', record)
        report['all_ten_outputs_identical'] = len(groups) == 1 and len(message_groups) == 1
        if not report['identical_requests']:
            report['status'] = 'stopped_request_mismatch'
            report['difference_origin'] = 'Request changed; cannot attribute output changes to server nondeterminism'
        elif not report['all_ten_outputs_identical']:
            report['status'] = 'stopped_nondeterministic_replay'
            report['difference_origin'] = 'Identical serialized requests; differences arose on the remote model/API side'
        else:
            report['status'] = 'replay_deterministic'
        C.write_json(report_path, report)
        if report['status'] != 'replay_deterministic':
            print(json.dumps({k: report[k] for k in ('status', 'all_ten_outputs_identical', 'runs_differing_from_first', 'identical_requests')}, indent=2), flush=True)
            return
        config = {key: value for key, value in payload.items() if key != 'messages'}
        config.update(model=C.MODEL, routed_model=payload['model'], backend='huggingface',
            api_base=API_BASE, provider='deepinfra', decode='greedy',
            prompt_seed_rule='(seed*7919 + call_number*104729) % 10000001',
            sampling_seed_rule='seed + call_number', maximum_retries=args.api_retries)
        plan = dict(model=C.MODEL, provider='deepinfra', generation_config=config,
            categories=['classroom'], conditions=list(C.CONDITIONS), seeds=list(C.SEEDS),
            benchmarks=[spec], mock=False, refinement=C.REFINEMENT, placement_call_count=1,
            sources=C.source_hashes())
        entries = [dict(category='classroom', benchmark=spec['benchmark'], benchmark_path=spec['path'],
            benchmark_sha256=spec['sha256'], model=C.MODEL, provider='deepinfra',
            condition=condition, seed=seed, adjective=C.CONDITIONS[condition],
            task_description=C.modified_benchmark(original, condition)['task_description'],
            output_path=f'classroom/{condition}/seed_{seed}', status='not_attempted')
            for seed in C.SEEDS for condition in C.CONDITIONS]
        manifest = dict(plan, identity=C.fingerprint(plan), status='running', runtime=backend.runtime,
            runs=entries, total_requested_generations=35, successful_generations=0,
            failed_generations=0, attempted_generations=0, not_attempted_generations=35,
            api_seed_replay_deterministic=True, deterministic_replay=copy.deepcopy(report))
        manifest_path = root / 'results/experiment_manifest.json'
        for entry in entries:
            print(f'Generating classroom/{entry["condition"]}/seed_{entry["seed"]}', flush=True)
            manifest['attempted_generations'] += 1
            try:
                G.run_initial(backend, spec, entry['condition'], entry['seed'], root / entry['output_path'], config)
                entry['status'] = 'success'
                manifest['successful_generations'] += 1
            except Exception as error:
                entry.update(status='failed', error=str(error))
                manifest['failed_generations'] += 1
                C.write_json(root / entry['output_path'] / 'failure_qwen35_27b.json', entry)
            manifest.update(not_attempted_generations=35-manifest['attempted_generations'],
                retry_count=api.retry_count, api_attempt_count=api.attempt_count)
            C.write_json(manifest_path, manifest)
        manifest['status'] = 'complete_with_failures' if manifest['failed_generations'] else 'complete'
        C.write_json(manifest_path, manifest)
        from adv_l1_behavioral_qwen35_27b import main as behavioral
        from adv_l1_variance_qwen35_27b import main as variance
        behavioral(['--input-dir', str(root)])
        variance(['--input-dir', str(root)])
        report.update(status=manifest['status'], attempted_generations=35,
            successful_generations=manifest['successful_generations'], failed_generations=manifest['failed_generations'])
        C.write_json(report_path, report)
    except Exception as error:
        report.update(status='blocked_api' if getattr(error, 'status', None) else 'failed',
            error=f'{type(error).__name__}: {error}', http_status=getattr(error, 'status', None))
        C.write_json(report_path, report)
        print(report['error'], flush=True)
    finally:
        if backend:
            report.update(api_attempt_count=backend.api.attempt_count, retry_count=backend.api.retry_count)
        report['benchmark_unchanged'] = C.digest(spec['path']) == spec['sha256']
        C.write_json(report_path, report)
        print('Report:', report_path, flush=True)


if __name__ == '__main__':
    main()
