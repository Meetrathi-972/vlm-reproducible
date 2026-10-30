# Qwen3.5-27B adjective experiment

This separate experiment measures how inserting one adjective changes the initial scene layout. Original SceneCritic/scenegen/probe_metrics source and benchmark JSONs are preserved. Only the new `_qwen35_27b` files run this experiment.

## Inference and current execution

The only permitted route is **Hugging Face Inference Providers → DeepInfra → `Qwen/Qwen3.5-27B`**. Requests go to `https://router.huggingface.co/v1/chat/completions` with model **`Qwen/Qwen3.5-27B:deepinfra`**. The suffix pins DeepInfra; alternative providers, models, API bases and local inference are rejected. A public Hugging Face mapping preflight verifies that DeepInfra lists this exact checkpoint as live.

Authentication uses `HF_TOKEN` from the environment. To reuse the existing SceneCritic setup, the backend securely loads that variable from `SceneCritic/.env` if the process environment lacks it. Tokens and authorization headers are never printed or saved. No additional Python packages are required.

**Actual execution on 2026-10-05:** the first two real identical requests returned different final content, so `api_seed_replay_deterministic=false`. A third request returned **HTTP 402: included Inference Providers credits depleted**. Classroom validation and all 245 benchmark generations consequently remain unattempted. The manifest records `blocked_api_credits`, 0 successful, 0 failed benchmark generations, 245 unattempted, 3 API attempts and 0 retries. Raw successful replay responses and the failed request are retained. The analysis CSVs contain headers only; their JSON reports explicitly say `not_run` and `scientific_results=false`. They are not scientific results.

The runner now uses exactly two requests for the replay check. A stochastic result is recorded and **does not stop the experiment**. An unavailable API or failed classroom pipeline validation does stop execution. Refill the existing Hugging Face account's inference credits before reproducing; no provider switch is implemented.

## Conditions and inputs

Conditions are exactly `baseline`, `well_organized`, `spacious`, `bright`, `quiet`, `cluttered`, `crowded`. The corresponding inserted words are `well-organized`, `spacious`, `bright`, `quiet`, `cluttered`, `crowded`.

Baseline remains byte-for-byte the original task-description string:

```text
This is an indoor classroom.
```

An intervention replaces only the article and inserts the adjective immediately before `indoor`:

```text
This is a spacious indoor classroom.
```

The JSON is deep-copied in memory; only `task_description` changes. Dimensions, assets, punctuation and original room wording are preserved. In particular the buffet benchmark says **restaurant**. Benchmark SHA-256 values are captured and checked; originals are never written.

All seven preferred files exist under `SceneCritic/benchmark/benchmark_base_rooms/`:

| Category | Selected file |
|---|---|
| bedroom | `bedroom/bedroom_1.json` |
| bookstore | `bookstore/bookstore_1.json` |
| buffet_restaurant | `buffet_restaurant/buffet_restaurant_1.json` |
| classroom | `classroom/classroom_1.json` |
| computer_room | `computer_room/computer_room_1.json` |
| dining_room | `dining_room/dining_room_1.json` |
| living_room | `living_room/living_room_1.json` |

The five fixed seeds are `1234`, `11234`, `21234`, `31234`, `41234`. Each category/seed has all seven conditions: **245 intended initial scenes**. Per-call sampling seed is `replicate_seed + call_number`; all seven conditions use the same four seeds and generation parameters within each replicate. Prompt-builder seeds follow the existing deterministic counter rule, recorded in the manifest.

Each request sets temperature `.5`, top-p `.9`, top-k `20`, max tokens `8192`, neutral penalties, `reasoning_effort="none"`, JSON-object format, one completion and no streaming. Request bodies and full response bodies are retained, including returned model and finish reason. The server's implementation of individual sampling controls cannot be guaranteed by a successful HTTP response. Exact final-content replay tests the observed behavior; it does not establish reproducibility across servers or time.

## Initial pipeline and source reuse

The adapter extracts explicitly named schemas, JSON parsers, prompt builders and three initial stage classes from the existing scenegen source using Python AST. It also extracts pure geometry functions from the repository. Original imports, model loading, probe capture and executable main blocks are not run.

The four logical generation calls are:

1. Scene description from the modified benchmark task.
2. Object choice from that description. As in the original pipeline, planning still uses the benchmark assets.
3. Atomic task planning from description, dimensions and assets.
4. **One placement call** from the atomic plan and original scene-object schema, with empty feedback.

Final visible content is parsed as scene JSON; separate reasoning is retained only in the raw API response. Completed `<think>...</think>` blocks in visible content are removed before parsing. Truncation, empty content and malformed scenes fail explicitly. Parsed initial objects are saved without correction or candidate selection. Compatibility `scene_pos_0_0.json` adds the usual ego marker; the initial-model scene excludes that marker.

There is no best-of-N, heuristic candidate scoring, feedback, placement refinement, orientation refinement, scale refinement, hidden-state capture or activation analysis. Geometry labels are computed for subsequent analysis and never fed back into generation.

Transient HTTP 408/429/5xx and transport failures receive at most two retries with identical payload, seed, model and provider. Retry attempts are logged separately; retries do not select among generated candidates. Permanent individual experiment failures are recorded and independent runs continue. A permanent pre-experiment API failure prevents spending further calls on an unavailable API.

## Analyses

Behavioral analysis compares each baseline with each adjective at the same category and seed, yielding 210 intended comparisons. It records zero-based first divergent call, structural divergence, placement change, normalized category-set Jaccard difference `1-J`, absolute raw object-list count difference, 5×5 occupancy Hamming fraction, mean Euclidean displacement of objects matched by exact label plus occurrence, and overlap-legality flip. Categories remove trailing numeric label suffixes. Occupancy and overlap use existing repository geometry; bounds violations are reported separately and do not redefine legality. No shared centroids yields a null centroid metric.

Variance analysis yields 210 intended category/adjective/metric rows for the five distance metrics. It computes within-baseline, within-adjective and between-condition distances, with statistic `mean_between - mean_within`. The primary exact test swaps condition labels **within each paired seed**, enumerating all 32 assignments. The symmetric statistic gives a minimum attainable p-value of **2/32 = 0.0625**. Five seeds cannot establish p<0.05 with this test. Original unrestricted permutations are retained only as exploratory results. Missing failed runs reduce the available paired sample; unavailable comparisons are marked explicitly.

The cloud API was observed to be stochastic despite matching seeds. Seed pairing controls the requested parameters, but does not isolate adjective effects from server randomness. More replicates and an appropriate expanded design would be needed for stronger statistical claims. Multiple comparisons are unadjusted.

## Files, outputs and reproduction

`api_backend_qwen35_27b.py` handles the pinned API, credentials and transport retries. `generate_adjective_scenes_qwen35_27b.py` handles replay, classroom validation, initial scenes and manifest. `adjective_common_qwen35_27b.py` supplies source adapters, interventions, geometry and statistics. The two `adv_l1_*_qwen35_27b.py` files generate the analyses. `test_adjective_experiment_qwen35_27b.py` checks the pipeline and analyses using temporary fixtures; fixtures never become final scientific results.

From the repository root:

```bash
bash VLM_Probe/run_adjective_experiment_qwen35_27b.sh
```

Or from `VLM_Probe/`:

```bash
bash run_adjective_experiment_qwen35_27b.sh
```

The default shell clears only `VLM_Probe/outputs/qwen35_27b_adjective_probe*` directories, verifies credentials and provider availability, runs replay then all seven classroom validation conditions at seed 1234, automatically generates all 245 scenes, runs both analyses and prints counts. Cleanup rejects linked/outside directories. It never removes original outputs, benchmarks, caches or unrelated experiments. Modes `dry-run`, `validate`, `analysis` are also available. A rerun clears the previous adjective evidence deliberately.

Outputs use `outputs/qwen35_27b_adjective_probe/<category>/<condition>/seed_<seed>/`, with copied benchmark, raw request/response for each call, initial parsed model scene, compatibility scene, prompts and stdout. `validation_qwen35_27b/` holds replay/classroom evidence. `results/` holds `experiment_manifest.json`, provider mapping and both `behavioral_summary` and `variance_summary` in CSV/JSON. Blocked execution produces explicit unavailable-analysis reports instead of synthetic results.

Regression command:

```bash
python -m unittest discover -s VLM_Probe -p test_adjective_experiment_qwen35_27b.py
```

Thirteen tests passed for exact interventions, category selection, unchanged assets/dimensions/benchmarks, initial call count, disabled refinement, provider/model pinning, HF credentials, seed payload, retry identity, permanent credit errors, stochastic replay continuation, behavioral geometry and paired permutation calculations. The shell also passes `bash -n`.
