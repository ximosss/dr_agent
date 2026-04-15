#!/usr/bin/env bash
set -euo pipefail

echo "=== Step 8: Evaluation ==="

cd /home/ubuntu/workspace/dr_agent

# Verify model is running
if ! curl -s http://localhost:8000/v1/models | grep -q "qwen3-8b"; then
    echo "ERROR: vLLM not responding on port 8000"
    exit 1
fi

echo "Model is ready. Starting evaluation..."

# Run each benchmark
echo "[1/3] Evaluating on FRAMES (20 examples)..."
uv run run_agent.py --eval -b frames -n 20

echo "[2/3] Evaluating on SimpleQA (20 examples)..."
uv run run_agent.py --eval -b simpleqa -n 20

echo "[3/3] Evaluating on GAIA (10 examples)..."
uv run run_agent.py --eval -b gaia -n 10

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
