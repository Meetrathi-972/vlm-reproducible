# VLM_Probe

Local scene generation and activation probe code, plus the existing hosted Qwen adjective experiments.

Install from the combined repository root using Python 3.11:

```bash
python -m pip install -r VLM_Probe/requirements.txt
```

For GPU scene generation use `scenegen/scenegen/generate_scenes.py`; the combined repository notebook configures its immutable model revision, greedy decoding and fresh output folders. `GEN_MODEL_REVISION` is required for real local model loading. `GEN_DTYPE=float16` supports GPUs without bfloat16.

The existing `_qwen35_27b` adjective experiments and bedroom replay demonstration remain hosted API experiments pinned to Qwen3.5-27B/DeepInfra. They need `HF_TOKEN` and may incur charges. See [their documentation](README_qwen35_27b_adjective_probe.md). The Colab notebook uses the separate local generation path; outputs from different model sizes or pipelines are different experiments.

Original activation capture code is in `probe_metrics/`; the notebook runs scene generation without activation capture.
