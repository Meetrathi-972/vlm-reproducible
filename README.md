# SceneCritic + VLM_Probe: reproducible runs

Contains both complete working source trees, benchmark inputs, ontology configuration and existing experiment outputs. Upstream projects: [SceneCritic](https://github.com/DIASENGUPTA/SceneCritic) and [VLM_Probe](https://github.com/DIASENGUPTA/VLM_Probe). Existing output folders are historical evidence, not newly reproduced results.

## Colab

1. Open `Colab.ipynb` in Colab and select a GPU runtime.
2. Clone this repository into `/content/vlm-reproducible` before running the notebook cells.
3. Run the setup, model server, SceneCritic comparison and VLM_Probe comparison cells.
4. Save the model revision SHA and result archive for future replay.

The notebook creates an isolated Python 3.11 environment and installs pinned dependencies. It runs **Qwen3-4B-Instruct-2507 locally**, using greedy decoding, eager attention, strict deterministic PyTorch operations, fixed seeds and a pinned model revision. This smaller model is a Colab example; it does not reproduce the existing hosted Qwen3.5/3.8-27B outputs. Use the exact original model and adequate GPU memory for that comparison. Long prompts may exceed 16 GB GPU memory.

The notebook performs two fresh runs per project and compares the generated scene JSONs. GPU execution has not been validated in this preparation environment. Matching outputs across different GPUs, CUDA versions or library versions is not guaranteed; see [PyTorch reproducibility](https://pytorch.org/docs/stable/notes/randomness.html).

## Dependencies and secrets

- `requirements-colab.lock.txt`: resolved transitive package pins for the combined notebook.
- `requirements-colab.txt`: local inference dependencies for both projects.
- `SceneCritic/requirements.txt`: direct dependencies pinned from the existing local environment.
- `SceneCritic/reproducibility/requirements-local-py310.txt`: full original Python 3.10 environment snapshot; CUDA 12.1 torch wheels require the PyTorch cu121 index.
- `SceneCritic/.env.example`: optional hosted API settings. Supply a token privately through environment variables or Colab Secrets. Real `.env` files are excluded.

The hosted `_qwen35_27b` experiments remain explicitly pinned to Hugging Face/DeepInfra and may incur API charges. The Colab examples use local model weights instead. Rendering needs Blender and external meshes/datasets, which are not bundled.

## Validation

Run the existing offline probe tests with:

```bash
python -m unittest discover -s VLM_Probe -p 'test_*qwen35_27b.py'
```

See each project's README for its original commands and experiment details.
