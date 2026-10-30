"""Replay only: preserve prior baseline request, change top_k to one, never generate an experiment."""
import copy
import hashlib
import json
from datetime import datetime, timezone
from types import SimpleNamespace
import adjective_common_qwen35_27b as C
from api_backend_qwen35_27b import ApiBackend, API_BASE


def main():
    previous = C.OUTPUT_ROOT.parent / 'qwen35_27b_classroom_greedy/replay_report_qwen35_27b.json'
    old = C.read_json(previous)
    payload = copy.deepcopy(old['exact_request'])
    payload.update(temperature=0.0, top_p=1.0, top_k=1, seed=1234)
    assert payload['model'] == 'Qwen/Qwen3.5-27B:deepinfra'
    assert payload['messages'] == old['exact_request']['messages']
    directory = C.OUTPUT_ROOT.parent / ('qwen35_27b_classroom_strict_greedy_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    directory.mkdir(parents=True, exist_ok=False)
    report = dict(status='running', model=C.MODEL, provider='deepinfra',
        source_baseline_report=str(previous), benchmark=old['benchmark'],
        exact_request=payload, requested_replays=10, completed_replays=0,
        all_outputs_identical=None, identical_requests=None, output_groups=[],
        runs_differing_from_first=[], experiment_generations=0,
        do_sample='Omitted: not documented in the DeepInfra chat-completion request schema',
        parameter_reference='https://docs.deepinfra.com/api-reference/chat-completions/openai-chat-completions',
        comparison='Complete returned content encoded UTF-8, without trimming, JSON normalization or thinking removal; response IDs/usage excluded')
    path = directory / 'replay_report_qwen35_27b.json'
    C.write_json(path, report)
    backend = None
    try:
        args = SimpleNamespace(model_path=C.MODEL, api_base=API_BASE, api_provider='deepinfra',
            api_timeout=180, api_retries=2, decode='greedy', temperature=0., top_p=1.,
            top_k=1, max_tokens=payload['max_tokens'])
        backend = ApiBackend(args)
        C.write_json(directory / 'provider_catalog_qwen35_27b.json', backend.catalog)
        system, prompt = (item['content'] for item in payload['messages'])
        assert backend.payload(prompt, 1234, system) == payload
        expected = json.dumps(payload, sort_keys=True, allow_nan=False).encode()
        contents, messages, requests = [], [], []
        for number in range(1, 11):
            print(f'Strict greedy replay {number}/10', flush=True)
            try:
                backend.complete(prompt, 1234, system)
                message = backend.last_response['raw_response']['choices'][0]['message']
                encoded = message['content'].encode('utf-8')
                (directory / f'replay_{number:02d}_content.txt').write_bytes(encoded)
                contents.append(encoded)
                messages.append(json.dumps(message, sort_keys=True, ensure_ascii=False).encode())
                requests.append(json.dumps(backend.last_response['request'], sort_keys=True, allow_nan=False).encode())
                report['completed_replays'] = len(contents)
                report['identical_requests'] = all(req == expected for req in requests)
                report['all_outputs_identical'] = all(content == contents[0] for content in contents)
                report['complete_messages_identical'] = all(msg == messages[0] for msg in messages)
                report['runs_differing_from_first'] = [i + 1 for i, content in enumerate(contents) if content != contents[0]]
                groups = {}
                for i, content in enumerate(contents):
                    groups.setdefault(hashlib.sha256(content).hexdigest(), []).append(i + 1)
                report['output_groups'] = list(groups.values())
                report['content_sha256'] = list(groups)
                C.write_json(path, report)
            finally:
                C.write_json(directory / f'replay_{number:02d}_api.json', backend.last_response)
        report['status'] = 'complete'
        report['difference_origin'] = ('No content differences observed' if report['all_outputs_identical'] else 'Remote model/API nondeterminism with identical serialized requests') if report['identical_requests'] else 'Requests differed; inspect saved payloads'
    except Exception as error:
        report.update(status='blocked_api' if getattr(error, 'status', None) else 'failed', error=str(error), http_status=getattr(error, 'status', None))
    finally:
        if backend:
            report.update(api_attempt_count=backend.attempt_count, retry_count=backend.retry_count)
        report['ten_replay_test_completed'] = report['completed_replays'] == 10
        C.write_json(path, report)
        print(json.dumps({key: report.get(key) for key in ('status', 'completed_replays', 'all_outputs_identical', 'identical_requests', 'output_groups', 'runs_differing_from_first', 'error')}, indent=2), flush=True)
        print('Report:', path, flush=True)


if __name__ == '__main__':
    main()
