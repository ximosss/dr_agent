#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PRIMARY_PORT=8000
EVAL_MODEL_NAME="${EVAL_MODEL_NAME:-qwen3-8b}"
FRAMES_EVAL_N="${FRAMES_EVAL_N:-20}"
SIMPLEQA_EVAL_N="${SIMPLEQA_EVAL_N:-20}"
GAIA_EVAL_N="${GAIA_EVAL_N:-10}"

echo "=== Step 8: Evaluation ==="

cd "${REPO_ROOT}"

# Verify model is running
if ! curl -s "http://localhost:${PRIMARY_PORT}/v1/models" | grep -q "${EVAL_MODEL_NAME}"; then
    echo "ERROR: vLLM not responding on port ${PRIMARY_PORT}"
    exit 1
fi

echo "Model is ready. Starting evaluation on the reserved benchmark test splits..."
export BASE_URL="http://localhost:${PRIMARY_PORT}/v1"
export MODEL_NAME_AT_ENDPOINT="${EVAL_MODEL_NAME}"

# Run each benchmark
echo "[1/3] Evaluating on FRAMES (${FRAMES_EVAL_N} examples)..."
uv run run_agent.py --eval -b frames -n "${FRAMES_EVAL_N}"

echo "[2/3] Evaluating on SimpleQA (${SIMPLEQA_EVAL_N} examples)..."
uv run run_agent.py --eval -b simpleqa -n "${SIMPLEQA_EVAL_N}"

echo "[3/3] Evaluating on GAIA (${GAIA_EVAL_N} examples)..."
uv run run_agent.py --eval -b gaia -n "${GAIA_EVAL_N}"

# Print summary of all results
echo ""
echo "=== Evaluation Results ==="
uv run python -c "
import json
from pathlib import Path

results_dir = Path('eval_outputs')
# Get the latest results for each benchmark
for bench in ['frames', 'simpleqa', 'gaia']:
    files = sorted(results_dir.glob(f'{bench}_*_results.json'), reverse=True)
    if files:
        with files[0].open() as f:
            data = json.load(f)
        acc = data.get('accuracy')
        acc_str = f'{acc:.1%}' if acc is not None else 'N/A'
        print(f'{bench:>10s}: {data.get(\"correct\", 0)}/{data.get(\"scored_examples\", 0)} correct ({acc_str}) | errors={data.get(\"errors\", 0)}')
"

echo "=== Evaluation complete ==="
