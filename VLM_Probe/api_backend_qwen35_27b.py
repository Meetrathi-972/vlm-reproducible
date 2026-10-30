"""Hugging Face-only Qwen3.5-27B chat; explicit provider, seeds and bounded retries."""
from __future__ import annotations
import copy
import json
import os
import re
import time
import urllib.error
import urllib.request

API_BASE = 'https://router.huggingface.co/v1'
MODEL_ID = 'Qwen/Qwen3.5-27B'
DEFAULT_PROVIDER = 'deepinfra'


def read_token(env_file=None):
    # Reuse SceneCritic's existing .env setup without exposing its contents.
    if not os.environ.get('HF_TOKEN'):
        from pathlib import Path
        path = Path(env_file) if env_file else Path(__file__).resolve().parents[1] / 'SceneCritic/.env'
        if path.is_file():
            for line in path.read_text().splitlines():
                match = re.match(r"^\s*(?:export\s+)?HF_TOKEN\s*=\s*(.*?)\s*$", line)
                if match:
                    os.environ['HF_TOKEN'] = match.group(1).strip().strip('\"\'')
                    break
    token = os.environ.get('HF_TOKEN')
    if not token:
        raise RuntimeError('Missing HF_TOKEN in environment or existing SceneCritic/.env; no inference attempted')
    return token


def redact(message, token):
    message = str(message)
    if token:
        message = message.replace(token, '[REDACTED]')
    return re.sub(r'(?:hf_[A-Za-z0-9]+|sk-or-[A-Za-z0-9_-]+|Bearer\s+\S+)', '[REDACTED]', message)


class ApiError(RuntimeError):
    def __init__(self, message, retryable=False, status=None):
        super().__init__(message)
        self.retryable, self.status = retryable, status


def request_json(url, payload=None, token=None, timeout=180):
    headers = {'Accept': 'application/json', 'User-Agent': 'qwen35_27b-adjective-experiment'}
    if token:
        headers['Authorization'] = f'Bearer {token}'
    body = None
    if payload is not None:
        body = json.dumps(payload, allow_nan=False, sort_keys=True).encode()
        headers['Content-Type'] = 'application/json'
    request = urllib.request.Request(url, data=body, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors='replace')[:2000]
        raise ApiError(f'Hugging Face HTTP {error.code}: {redact(detail, token)}', error.code in (408, 429) or error.code >= 500, error.code) from None
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        raise ApiError(f'Hugging Face transport error: {redact(error, token)}', True) from None
    except ValueError as error:
        raise ApiError(f'Hugging Face invalid JSON response: {redact(error, token)}') from None


def inspect_provider(provider, timeout=180):
    if provider != DEFAULT_PROVIDER:
        raise ValueError('Only the pinned DeepInfra provider is allowed')
    catalog = request_json('https://huggingface.co/api/models/' + MODEL_ID + '?expand=inferenceProviderMapping', timeout=timeout)
    endpoint = catalog.get('inferenceProviderMapping', {}).get(provider)
    if not endpoint or endpoint.get('status') != 'live' or endpoint.get('providerId') != MODEL_ID:
        raise RuntimeError('Exact Qwen/Qwen3.5-27B is unavailable on DeepInfra; no fallback')
    return dict(endpoint, provider_name='DeepInfra', name=MODEL_ID), catalog


class ApiBackend:
    def __init__(self, args):
        self.args = args
        if args.model_path != 'Qwen/Qwen3.5-27B' or args.api_base.rstrip('/') != API_BASE:
            raise ValueError('This experiment requires exactly Qwen/Qwen3.5-27B through https://router.huggingface.co/v1')
        self.base, self.provider, self.routed_model = API_BASE, args.api_provider, MODEL_ID + ':deepinfra'
        self.endpoint, self.catalog = inspect_provider(self.provider, args.api_timeout)
        self.token = read_token(getattr(args, 'api_env_file', None))
        self.runtime = {'backend': 'huggingface_chat_api', 'api_base': self.base, 'api_method': 'POST /chat/completions', 'provider': self.provider, 'provider_name': self.endpoint['provider_name'], 'routed_model': self.routed_model, 'quantization': self.endpoint.get('quantization'), 'endpoint_name': self.endpoint['name'], 'server_determinism': 'Measured by uncached replay; stochastic replay does not block generation', 'local_model_loading': False, 'response_format': 'json_object', 'maximum_retries_per_request': args.api_retries}
        self.last_response = self.last_output = self.server_signature = None
        self.retry_count = 0
        self.attempt_count = 0

    def payload(self, prompt, seed, system):
        return {'model': self.routed_model, 'messages': [{'role': 'system', 'content': system}, {'role': 'user', 'content': prompt}], 'seed': seed, 'temperature': self.args.temperature if self.args.decode == 'sample' else 0.0, 'top_p': self.args.top_p, 'top_k': self.args.top_k, 'max_tokens': self.args.max_tokens, 'frequency_penalty': 0.0, 'presence_penalty': 0.0, 'repetition_penalty': 1.0, 'stop': [], 'reasoning_effort': 'none', 'response_format': {'type': 'json_object'}, 'stream': False, 'n': 1}

    def complete(self, prompt, seed, system):
        self.last_output = None
        payload = self.payload(prompt, seed, system)
        self.last_response = {'request': copy.deepcopy(payload), 'attempts': [], 'raw_response': None}
        for attempt in range(self.args.api_retries + 1):
            if attempt:
                self.retry_count += 1
                time.sleep(min(2 ** (attempt - 1), 4))
            self.attempt_count += 1
            try:
                response = request_json(self.base + '/chat/completions', copy.deepcopy(payload), self.token, self.args.api_timeout)
                if response.get('error'):
                    error = response['error']
                    code = error.get('code') if isinstance(error, dict) else None
                    raise ApiError(redact(json.dumps(error), self.token), code in (408, 429) or isinstance(code, int) and code >= 500, code)
                self.last_response['attempts'].append({'attempt': attempt + 1, 'status': 'success'})
                break
            except ApiError as error:
                self.last_response['attempts'].append({'attempt': attempt + 1, 'status': 'failed', 'error': str(error), 'http_status': error.status})
                if not error.retryable or attempt >= self.args.api_retries:
                    raise
        # Save full body, without ever copying authorization headers or credentials.
        response = json.loads(redact(json.dumps(response), self.token))
        self.last_response['raw_response'] = response
        try:
            choice = response['choices'][0]
            message = choice['message']
        except (KeyError, IndexError, TypeError):
            raise RuntimeError('Hugging Face response has no completion choice') from None
        self.last_output = message.get('content')
        self.last_response.update(id=response.get('id'), response_model=response.get('model'), response_provider=response.get('provider'), system_fingerprint=response.get('system_fingerprint'), usage=response.get('usage'), finish_reason=choice.get('finish_reason'))
        if response.get('model') != MODEL_ID:
            raise RuntimeError('Hugging Face response model differs from exact requested model')
        returned_provider = response.get('provider')
        if returned_provider and str(returned_provider).casefold() != 'deepinfra':
            raise RuntimeError('Returned provider differs from pinned DeepInfra')
        if choice.get('finish_reason') != 'stop':
            raise ValueError(f'Generation did not finish normally ({choice.get("finish_reason")}); raw response retained')
        if not isinstance(self.last_output, str) or not self.last_output.strip():
            raise ValueError('No visible scene-generation content returned')
        if '<think>' in self.last_output or '</think>' in self.last_output:
            self.last_output = re.sub(r'<think>.*?</think>', '', self.last_output, flags=re.S).strip()
        return self.last_output
