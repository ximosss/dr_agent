#!/usr/bin/env bash
set -euo pipefail

echo "=== Step 6: LoRA SFT Training ==="

cd /home/ubuntu/workspace/dr_agent

RUN_ID="${SFT_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
ADAPTER_DIR="/data/sft_output/qwen3-8b-dr-agent-${RUN_ID}"
RUN_NAME="qwen3-8b-dr-agent-sft-${RUN_ID}"
LATEST_ADAPTER_FILE="/home/ubuntu/workspace/dr_agent/training/data/latest_adapter_dir.txt"

# 1. Convert and merge data
echo "[1/3] Converting to ShareGPT format..."
uv run python training/convert_sharegpt.py

echo "[1/3] Merging and splitting..."
uv run python training/merge_and_split.py

# 2. Verify data
echo "[2/3] Verifying dataset..."
uv run python -c "
import json
with open('training/data/sft_ready/train.json') as f:
    train_data = json.load(f)
with open('training/data/sft_ready/eval.json') as f:
    eval_data = json.load(f)
print(f'Training examples: {len(train_data)}')
print(f'Eval examples: {len(eval_data)}')
# Check first example
ex = train_data[0]
print(f'First example keys: {list(ex.keys())}')
print(f'Conversation turns: {len(ex[\"conversations\"])}')
for turn in ex['conversations'][:3]:
    print(f'  {turn[\"from\"]}: {turn[\"value\"][:80]}...')
"

# 3. Run training on GPU 1
echo "[3/3] Starting SFT training on GPU 1..."
echo "  Run ID: ${RUN_ID}"
echo "  Adapter output: ${ADAPTER_DIR}"
WANDB_PROJECT=deep_research_agent CUDA_VISIBLE_DEVICES=1 uv run --group training \
    llamafactory-cli train \
    /home/ubuntu/workspace/dr_agent/training/configs/qwen3_8b_sft.yaml \
    output_dir="${ADAPTER_DIR}" \
    run_name="${RUN_NAME}"

printf '%s\n' "${ADAPTER_DIR}" > "${LATEST_ADAPTER_FILE}"

echo "=== SFT Training complete ==="
echo "Output: ${ADAPTER_DIR}"
