#!/usr/bin/env bash
# Hugging Face -> seed replay check -> classroom -> 245 scenes -> both analyses.
# Cleanup is confined by Python to named adjective outputs in VLM_Probe/outputs.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
API_PROVIDER="deepinfra"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRIPT_DIR/outputs/qwen35_27b_adjective_probe}"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-$SCRIPT_DIR/../SceneCritic/benchmark/benchmark_base_rooms}"
MODE="${1:-all}"
if [[ $# -gt 0 ]]; then shift; fi
case "$MODE" in
    -h|--help)
        cat <<'HELP'
Usage: bash run_adjective_experiment_qwen35_27b.sh [all|dry-run|validate|analysis]
Default all: clear old adjective outputs, Hugging Face replay measurement, classroom
validation, full 245 scenes, behavioral and paired variance analyses, summary.
Required: HF_TOKEN and available inference credits (token never printed or saved).
Overrides: PYTHON_BIN, BENCHMARK_ROOT.
HF_TOKEN may also be loaded securely from existing SceneCritic/.env.
HELP
        exit 0 ;;
    all|dry-run|validate|analysis) ;;
    *) echo 'Unknown mode. Use --help.' >&2; exit 2 ;;
esac
if [[ $# -gt 0 ]]; then echo 'Use the Python CLI for diagnostic options; final shell design is fixed.' >&2; exit 2; fi
GENERATION_ARGS=(--model-path Qwen/Qwen3.5-27B --api-base https://router.huggingface.co/v1
    --api-provider "$API_PROVIDER" --benchmark-root "$BENCHMARK_ROOT" --output-root "$OUTPUT_ROOT"
    --seeds 1234 11234 21234 31234 41234)
summarize() {
    "$PYTHON_BIN" - "$OUTPUT_ROOT" <<'PY'
import json,sys
from pathlib import Path
path=Path(sys.argv[1])/'results/experiment_manifest.json'
if path.exists():
    data=json.loads(path.read_text())
    print('Final experiment status:', data['status'])
    print('Model:', data['model'], '| Provider:', data['provider'])
    print('Success:',data['successful_generations'],'/',data['total_requested_generations'],
          '| Failed:',data['failed_generations'],'| Not attempted:',data['not_attempted_generations'],
          '| Retries:',data['retry_count'])
    print('Deterministic replay:',data['deterministic_replay'].get('api_seed_replay_deterministic',False))
    print('Results:',path.parent)
PY
}
trap summarize EXIT
case "$MODE" in
    dry-run) "$PYTHON_BIN" "$SCRIPT_DIR/generate_adjective_scenes_qwen35_27b.py" --mode dry-run "${GENERATION_ARGS[@]}" ;;
    validate) "$PYTHON_BIN" "$SCRIPT_DIR/generate_adjective_scenes_qwen35_27b.py" --mode validate --cleanup "${GENERATION_ARGS[@]}" ;;
    all) "$PYTHON_BIN" "$SCRIPT_DIR/generate_adjective_scenes_qwen35_27b.py" --mode full --cleanup "${GENERATION_ARGS[@]}" ;;
esac
if [[ "$MODE" == all || "$MODE" == analysis ]]; then
    "$PYTHON_BIN" "$SCRIPT_DIR/adv_l1_behavioral_qwen35_27b.py" --input-dir "$OUTPUT_ROOT" --output-dir "$OUTPUT_ROOT/results"
    "$PYTHON_BIN" "$SCRIPT_DIR/adv_l1_variance_qwen35_27b.py" --input-dir "$OUTPUT_ROOT" --output-dir "$OUTPUT_ROOT/results"
fi
