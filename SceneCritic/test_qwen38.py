"""One short HF-routed API request; success does not establish free billing."""
import os
from pathlib import Path
import re
import sys

from huggingface_hub import InferenceClient


def main():
    token = os.environ.get('HF_TOKEN')
    env_file = Path(__file__).resolve().parent / '.env'
    if not token and env_file.is_file():
        for line in env_file.read_text().splitlines():
            key, separator, value = line.partition('=')
            if separator and key.strip() == 'HF_TOKEN':
                token = value.strip().strip('\"\'')
                break
    if not token:
        print('Missing HF_TOKEN: set it in the environment or local .env.', file=sys.stderr)
        return 1
    client = InferenceClient(model='Qwen/Qwen3.8-27B', provider='auto', token=token, timeout=60)
    print('Testing Qwen/Qwen3.8-27B through Hugging Face (provider=auto).', flush=True)
    try:
        response = client.chat.completions.create(
            messages=[{'role': 'user', 'content': 'Say hello in one sentence.'}],
            max_tokens=512,
        )
    except Exception as exc:
        message = str(exc).replace(token, '[REDACTED]')
        message = re.sub(r'hf_[A-Za-z0-9]+', '[REDACTED]', message)
        print(f'API test failed: {type(exc).__name__}: {message}', file=sys.stderr)
        return 1
    print('Response:', response.choices[0].message.content)
    print('Finish reason:', response.choices[0].finish_reason)
    print('Usage:', response.usage)
    if not response.choices[0].message.content:
        print('No visible answer received; the text-generation test is incomplete.', file=sys.stderr)
        return 2
    print('API request succeeded. Check HF billing to determine whether free credits covered it.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
