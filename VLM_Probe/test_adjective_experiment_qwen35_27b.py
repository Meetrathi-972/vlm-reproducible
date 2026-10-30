"""Regression tests of scientific controls, repository equivalence and CLI flow."""
import contextlib
import io
import copy
import itertools
import json
import math
import os
from unittest.mock import patch
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace

import adjective_common_qwen35_27b as C
import generate_adjective_scenes_qwen35_27b as G
from adv_l1_behavioral_qwen35_27b import compare
from adv_l1_variance_qwen35_27b import summarize


class ExperimentTests(unittest.TestCase):
    def test_interventions_all_categories(self):
        selected = C.select_benchmarks(C.BENCHMARK_ROOT, C.CATEGORIES)
        self.assertEqual(len(selected), 7)
        for spec in selected:
            self.assertEqual(spec['benchmark'], spec['category'] + '_1')
            original = copy.deepcopy(spec['original'])
            for condition, adjective in C.CONDITIONS.items():
                result = C.modified_benchmark(original, condition)
                expected = original['task_description'] if adjective is None else original['task_description'].replace('an indoor', 'a ' + adjective + ' indoor', 1)
                self.assertEqual(result['task_description'], expected)
                self.assertEqual(result['assets'], original['assets'])
                self.assertEqual(result['room_dimension'], original['room_dimension'])
            self.assertEqual(original, spec['original'])
            self.assertEqual(C.digest(spec['path']), spec['sha256'])
        with self.assertRaises(ValueError):
            C.modified_benchmark({'task_description': 'A room'}, 'bright')

    def test_geometry_repository_semantics(self):
        room = {'room_dimension_x': 10, 'room_dimension_y': 10}
        objects = [{'label': 'chair_1', 'bbox': [-5, 0, 2, 2], 'position': [-4, 1]}]
        labels = C.scene_labels(objects, room)
        self.assertEqual(labels['occupancy_str'][0], 'chair')
        self.assertEqual(labels['occupancy_str'].count('chair'), 1)
        self.assertEqual(labels['legality'], 1)
        self.assertFalse(labels['oob'])
        overlapping = objects + [{'label': 'table', 'bbox': [-5, 0, 2, 2], 'position': [-4, 1]}]
        self.assertEqual(C.scene_labels(overlapping, room)['legality'], 0)
        outside = [{'label': 'chair', 'bbox': [10, 10, 2, 2], 'position': [11, 11]}]
        self.assertEqual(C.scene_labels(outside, room)['legality'], 1)
        self.assertTrue(C.scene_labels(outside, room)['oob'])

    def test_metric_values_duplicate_objects(self):
        def result(objects):
            return {'scene': {'objects': objects}, 'labels': C.scene_labels(objects, {'room_dimension_x': 10, 'room_dimension_y': 10})}
        a = result([{'label': 'chair', 'bbox': [-5, 0, 1, 1], 'position': [-4.5, .5]}, {'label': 'chair', 'bbox': [-3, 0, 1, 1], 'position': [-2.5, .5]}])
        b = result([{'label': 'chair', 'bbox': [-4, 0, 1, 1], 'position': [-3.5, .5]}, {'label': 'chair', 'bbox': [-2, 0, 1, 1], 'position': [-1.5, .5]}, {'label': 'table', 'bbox': [0, 0, 1, 1], 'position': [.5, .5]}])
        metrics = C.pair_metrics(a, b)
        self.assertEqual(metrics['object_count_diff'], 1)
        self.assertEqual(metrics['object_set_diff'], .5)
        self.assertEqual(metrics['centroid_shift'], 1)
        self.assertEqual(C.pair_metrics(a, a)['occupancy_hamming'], 0)

    def test_permutation_equivalence_and_pairing(self):
        # Execute original repository test with a minimal NumPy mean shim. This
        # verifies equivalence without importing its inference dependencies.
        original, _ = C.load_definitions(C.ROOT / 'probe_metrics/adv_l1_variance.py', {'_stat', 'permutation_test'}, {'np': SimpleNamespace(mean=statistics.mean), 'itertools': itertools})
        n = 3
        matrix = [[None if i == j else (0.0 if (i < n) == (j < n) else 1.0) for j in range(2*n)] for i in range(2*n)]
        converted = [[float('nan') if v is None else v for v in row] for row in matrix]
        expected = original.permutation_test(converted, [0]*n + [1]*n)
        actual = C.permutation_test(matrix, n, paired=False)
        self.assertEqual((actual['test_statistic'], actual['p_value'], actual['number_of_permutations'], actual['min_achievable_p']), expected)
        n = 5
        matrix = [[None if i == j else (0.0 if (i < n) == (j < n) else 1.0) for j in range(2*n)] for i in range(2*n)]
        paired = C.permutation_test(matrix, n)
        self.assertEqual(paired['number_of_permutations'], 32)
        self.assertEqual(paired['p_value'], .0625)
        self.assertEqual(paired['min_achievable_p'], .0625)

    def test_cli_validation_generation_analysis(self):
        with tempfile.TemporaryDirectory(prefix='experiment_mock_qwen35_27b_') as temporary:
            root = Path(temporary)
            def command(script, *arguments):
                result = subprocess.run([sys.executable, str(C.ROOT / script), *map(str, arguments)], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            command('generate_adjective_scenes_qwen35_27b.py', '--mode', 'full', '--mock', '--categories', 'classroom', '--output-root', root)
            manifest, groups = C.load_runs(root)
            self.assertEqual(len(manifest['runs']), 35)
            self.assertTrue(C.read_json(root / 'validation_qwen35_27b/validation_qwen35_27b.json')['passed'])
            self.assertTrue(manifest['mock'])
            for runs in groups.values():
                for result in runs.values():
                    self.assertEqual(len(result['calls']), 4)
                    self.assertEqual(result['refinement'], C.REFINEMENT)
                    self.assertEqual(result['scene']['objects'], json.loads(result['calls'][-1]['output_text'])['objects'])
                    self.assertNotIn('activations', result)
                    self.assertNotIn('Feedback based', result['calls'][-1]['prompt'])
            self.assertFalse(list(root.rglob('*refine_logs*')))
            command('adv_l1_behavioral_qwen35_27b.py', '--input-dir', root)
            command('adv_l1_variance_qwen35_27b.py', '--input-dir', root)
            behavior = C.read_json(root / 'results/behavioral_summary.json')
            variance = C.read_json(root / 'results/variance_summary.json')
            self.assertEqual(len(behavior['rows']), 30)
            self.assertEqual(len(variance['rows']), 30)
            self.assertTrue(all(r['verdict'] == 'underpowered by design' for r in variance['rows']))
            # Corruption and a changed paired configuration must not be analyzed.
            path = root / manifest['runs'][1]['output_path'] / 'run_qwen35_27b.json'
            corrupted = C.read_json(path)
            corrupted['generation_config']['temperature'] = .8
            C.write_json(path, corrupted)
            with self.assertRaises(ValueError):
                C.load_runs(root)

    def test_single_call_no_scoring_or_refinement(self):
        spec = C.select_benchmarks(C.BENCHMARK_ROOT, ['classroom'])[0]
        args = SimpleNamespace(mock=True, model_path=C.MODEL)
        with tempfile.TemporaryDirectory() as temporary:
            backend = G.Backend(args)
            first = G.run_initial(backend, spec, 'baseline', 1234, Path(temporary)/'first', {})
            second = G.run_initial(backend, spec, 'baseline', 1234, Path(temporary)/'second', {})
            self.assertEqual(G.run_payload(first), G.run_payload(second))
            self.assertEqual(len(first['calls']), 4)
            self.assertEqual(compare(first, second)['first_divergent_call'], None)
            self.assertEqual(compare(first, second)['structural_divergence'], False)


    def test_api_payload_and_replay_ignore_request_ids(self):
        from api_backend_qwen35_27b import ApiBackend
        args = SimpleNamespace(mock=False, backend='api', model_path=C.MODEL,
            api_env_file='/nonexistent_qwen35_27b.env', api_base='https://router.huggingface.co/v1',
            api_provider='deepinfra', api_timeout=180, api_retries=2, top_k=20, temperature=.5, decode='sample', top_p=.9, max_tokens=8192)
        requests = []
        def transport(url, payload=None, token=None, timeout=180):
            if payload is None:
                return self.provider_catalog()
            requests.append(payload)
            self.assertEqual(token, 'fixture_token')
            return {'id': f'volatile-{len(requests)}', 'model': 'Qwen/Qwen3.5-27B', 'provider': 'DeepInfra', 'system_fingerprint': 'fixed', 'usage': {'total_tokens': 20}, 'choices': [{'message': {'content': '{"objects": []}'}, 'finish_reason': 'stop'}]}
        with patch.dict(os.environ, {'HF_TOKEN': 'fixture_token'}), patch('api_backend_qwen35_27b.request_json', side_effect=transport):
            backend = G.Backend(args)
            self.assertEqual(backend.complete('Random seed: 9', 'placement', 1238, {}, 'JSON only'), '{"objects": []}')
            first_response = copy.deepcopy(backend.api.last_response)
            backend.complete('Random seed: 9', 'placement', 1238, {}, 'JSON only')
            self.assertEqual(requests[0], requests[1])
            self.assertEqual(requests[0]['model'], 'Qwen/Qwen3.5-27B:deepinfra')
            self.assertNotIn('provider', requests[0])
            self.assertEqual(requests[0]['seed'], 1238)
            self.assertEqual(requests[0]['reasoning_effort'], 'none')
            self.assertEqual(requests[0]['temperature'], .5)
            self.assertEqual(requests[0]['top_p'], .9)
            self.assertNotIn('fixture_token', json.dumps(backend.runtime))
            first = {'scene': {}, 'labels': {}, 'benchmark_copy': {}, 'generation_config': {}, 'refinement': {}, 'runtime': {}, 'calls': [{'stage': 'placement', 'output_text': '{}', 'api_response': first_response}]}
            second = copy.deepcopy(first)
            second['calls'][0]['api_response'] = backend.api.last_response
            self.assertEqual(G.run_payload(first), G.run_payload(second))
            second['calls'][0]['output_text'] = '{"different": true}'
            self.assertNotEqual(G.run_payload(first), G.run_payload(second))

    def test_api_rejects_truncation_and_provider_changes(self):
        from api_backend_qwen35_27b import ApiBackend
        args = SimpleNamespace(model_path=C.MODEL, api_env_file='/nonexistent_qwen35_27b.env', api_base='https://router.huggingface.co/v1', api_provider='deepinfra', api_timeout=180, api_retries=2, top_k=20, temperature=.5, decode='sample', top_p=.9, max_tokens=8192)
        mapping = self.provider_catalog()
        response = {'model': 'Qwen/Qwen3.5-27B', 'provider': 'DeepInfra', 'choices': [{'message': {'content': '{"objects": []}'}, 'finish_reason': 'length'}]}
        with patch.dict(os.environ, {'HF_TOKEN': 'fixture_token'}), patch('api_backend_qwen35_27b.request_json', side_effect=[mapping, response]) as transport:
            backend = ApiBackend(args)
            with self.assertRaises(ValueError):
                backend.complete('text', 1234, 'JSON')
            self.assertEqual(transport.call_count, 2)  # mapping + one attempt; no retry
            self.assertEqual(backend.last_output, '{"objects": []}')
        response['choices'][0]['finish_reason'] = 'stop'
        changed = copy.deepcopy(response)
        changed['model'] = 'another model'
        with patch.dict(os.environ, {'HF_TOKEN': 'fixture_token'}), patch('api_backend_qwen35_27b.request_json', side_effect=[mapping, response, changed]):
            backend = ApiBackend(args)
            backend.complete('text', 1234, 'JSON')
            with self.assertRaises(RuntimeError):
                backend.complete('text', 1234, 'JSON')

    def test_api_credentials_and_redaction(self):
        from api_backend_qwen35_27b import read_token, redact
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'credentials_qwen35_27b.env'
            path.write_text('HF_TOKEN="fixture_token"\n')
            with patch.dict(os.environ, {}, clear=True):
                self.assertEqual(read_token(path), 'fixture_token')
            with patch.dict(os.environ, {}, clear=True):
                with self.assertRaises(RuntimeError):
                    read_token(Path(directory) / 'absent.env')
            with patch.dict(os.environ, {'HF_TOKEN': 'fixture_token'}):
                self.assertEqual(read_token(), 'fixture_token')
        self.assertEqual(redact('bad fixture_token hf_example123', 'fixture_token'), 'bad [REDACTED] [REDACTED]')



    @staticmethod
    def provider_catalog():
        return {'inferenceProviderMapping': {'deepinfra': {'status': 'live', 'providerId': C.MODEL, 'task': 'conversational'}}}

    def test_api_transient_retry_preserves_request(self):
        from api_backend_qwen35_27b import ApiBackend, ApiError
        args = SimpleNamespace(model_path=C.MODEL, api_base='https://router.huggingface.co/v1',
            api_provider='deepinfra', api_timeout=180, api_retries=2, top_k=20,
            temperature=.5, decode='sample', top_p=.9, max_tokens=8192)
        payloads = []
        def transport(url, payload=None, token=None, timeout=180):
            if payload is None:
                return self.provider_catalog()
            payloads.append(copy.deepcopy(payload))
            if len(payloads) == 1:
                raise ApiError('temporary', True, 503)
            return {'model': 'Qwen/Qwen3.5-27B', 'provider': 'DeepInfra',
                    'choices': [{'message': {'content': '{"objects": []}'}, 'finish_reason': 'stop'}]}
        with patch.dict(os.environ, {'HF_TOKEN': 'fixture_token'}), patch('api_backend_qwen35_27b.request_json', side_effect=transport), patch('api_backend_qwen35_27b.time.sleep'):
            backend = ApiBackend(args)
            backend.complete('same text', 1238, 'JSON')
            self.assertEqual(payloads[0], payloads[1])
            self.assertEqual(backend.retry_count, 1)
            self.assertEqual(backend.attempt_count, 2)
            self.assertEqual(len(backend.last_response['attempts']), 2)
            self.assertIsNotNone(backend.last_response['raw_response'])

    def test_stochastic_replay_continues_classroom_validation(self):
        spec = C.select_benchmarks(C.BENCHMARK_ROOT, ['classroom'])[0]
        args = SimpleNamespace(mock=True, model_path=C.MODEL)
        backend = G.Backend(args)
        original = backend.complete
        replay_outputs = iter(['{"objects": []}', '{"objects": [{"label": "different"}]}'])
        counter = [0]
        def complete(*arguments):
            counter[0] += 1
            if counter[0] <= 2:
                return next(replay_outputs)
            return original(*arguments)
        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(backend, 'complete', side_effect=complete):
                G.validate(backend, [spec], list(C.SEEDS), list(C.CONDITIONS), Path(temporary), {}, 'test')
            report = C.read_json(Path(temporary) / 'validation_qwen35_27b/validation_qwen35_27b.json')
            self.assertTrue(report['passed'])
            self.assertFalse(report['api_seed_replay_deterministic'])
            self.assertTrue(report['pairs'][0]['identical_requests'])
            self.assertEqual(len(list((Path(temporary) / 'validation_qwen35_27b/classroom').glob('*/seed_1234/run_qwen35_27b.json'))), 7)

    def test_provider_pin_and_credit_failure(self):
        from api_backend_qwen35_27b import ApiBackend, ApiError, inspect_provider
        with self.assertRaises(ValueError):
            inspect_provider('other-provider')
        args = SimpleNamespace(model_path=C.MODEL, api_base='https://router.huggingface.co/v1', api_provider='deepinfra', api_timeout=180, api_retries=2, top_k=20, temperature=.5, decode='sample', top_p=.9, max_tokens=8192)
        with patch.dict(os.environ, {'HF_TOKEN': 'fixture_token'}), patch('api_backend_qwen35_27b.request_json', side_effect=[self.provider_catalog(), ApiError('credits depleted', False, 402)]) as transport:
            backend = ApiBackend(args)
            with self.assertRaises(ApiError):
                backend.complete('same text', 1234, 'JSON')
            self.assertEqual(transport.call_count, 2)
            self.assertEqual(backend.retry_count, 0)
            self.assertEqual(backend.last_response['attempts'][0]['http_status'], 402)

    def test_full_run_records_credit_block_after_stochastic_replay(self):
        from api_backend_qwen35_27b import ApiError
        response = {'model': C.MODEL, 'choices': [{'message': {'content': '{"objects": []}'}, 'finish_reason': 'stop'}]}
        changed = copy.deepcopy(response)
        changed['choices'][0]['message']['content'] = '{"objects": [], "different": true}'
        with tempfile.TemporaryDirectory(prefix='qwen35_27b_') as directory:
            root = Path(directory) / 'qwen35_27b_results'
            with patch.dict(os.environ, {'HF_TOKEN': 'fixture_token'}), patch('api_backend_qwen35_27b.request_json', side_effect=[self.provider_catalog(), self.provider_catalog(), response, changed, ApiError('credits depleted', False, 402)]), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(ApiError):
                    G.main(['--mode', 'full', '--output-root', str(root)])
            manifest = C.read_json(root / 'results/experiment_manifest.json')
            self.assertEqual(manifest['status'], 'blocked_api_credits')
            self.assertFalse(manifest['api_seed_replay_deterministic'])
            self.assertEqual(manifest['not_attempted_generations'], 245)
            self.assertEqual(manifest['successful_generations'], 0)
            self.assertEqual(manifest['api_attempt_count'], 3)
            for stem in ('behavioral_summary', 'variance_summary'):
                summary = C.read_json(root / 'results' / (stem + '.json'))
                self.assertFalse(summary['metadata']['scientific_results'])


if __name__ == '__main__':
    unittest.main()
