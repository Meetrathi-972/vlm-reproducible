"""Shared configuration for reproducible Qwen runs."""
import os
from pathlib import Path
import random


def load_env():
    path = Path(__file__).resolve().parents[1] / '.env'
    if path.exists():
        for line in path.read_text().splitlines():
            key, sep, value = line.partition('=')
            key = key.strip()
            if sep and key and not key.startswith('#'):
                os.environ.setdefault(key, value.strip().strip("\"'"))


def configure_lm():
    import dspy
    import numpy as np

    load_env()
    seed = int(os.getenv('SCENECRITIC_SEED', '42'))
    random.seed(seed)
    np.random.seed(seed)
    backend = os.getenv('SCENECRITIC_BACKEND', 'hosted')
    if backend not in ('local', 'hosted'):
        raise ValueError('SCENECRITIC_BACKEND must be local or hosted')
    local = backend == 'local'
    model = os.getenv('SCENECRITIC_MODEL') or (
        'Qwen/Qwen3-4B-Instruct-2507' if local else 'Qwen/Qwen3.8-27B'
    )
    token = 'local' if local else os.getenv('HF_TOKEN')
    if not token:
        raise RuntimeError('Set HF_TOKEN in your environment or .env for hosted inference')
    return dspy.LM(
        'openai/' + model,
        api_base=os.getenv('SCENECRITIC_API_BASE') or (
            'http://127.0.0.1:7501/v1' if local else 'https://router.huggingface.co/v1'
        ),
        api_key=token, model_type='chat', temperature=0.0, seed=seed,
        max_tokens=int(os.getenv('SCENECRITIC_MAX_TOKENS', '8192')),
        timeout=1800 if local else 180, num_retries=0, cache=False,
    )
