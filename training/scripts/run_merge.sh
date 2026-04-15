#!/usr/bin/env bash
set -euo pipefail

echo "=== Step 7: LoRA Merge + Deploy ==="

cd /home/ubuntu/workspace/dr_agent

LATEST_ADAPTER_FILE="/home/ubuntu/workspace/dr_agent/training/data/latest_adapter_dir.txt"
ADAPTER_DIR="${SFT_ADAPTER_DIR:-}"

if [ -z "${ADAPTER_DIR}" ] && [ -f "${LATEST_ADAPTER_FILE}" ]; then
    ADAPTER_DIR="$(cat "${LATEST_ADAPTER_FILE}")"
fi

if [ -z "${ADAPTER_DIR}" ]; then
    echo "ERROR: No adapter directory provided."
    echo "Run training first, or set SFT_ADAPTER_DIR=/data/sft_output/qwen3-8b-dr-agent-<run_id>"
    exit 1
fi

# 1. Find best checkpoint
BEST_CKPT=$(ls -d "${ADAPTER_DIR}"/checkpoint-* 2>/dev/null | sort -V | tail -1)

if [ -z "$BEST_CKPT" ]; then
    echo "Using adapter dir directly: ${ADAPTER_DIR}"
    BEST_CKPT="${ADAPTER_DIR}"
else
    echo "Using best checkpoint: ${BEST_CKPT}"
fi

# 2. Merge LoRA
echo "[1/3] Merging LoRA weights..."
CUDA_VISIBLE_DEVICES=1 uv run --group training \
    llamafactory-cli export \
    --model_name_or_path /data/qwen3-8b \
    --adapter_name_or_path "${BEST_CKPT}" \
    --template qwen3 \
    --finetuning_type lora \
    --export_dir /data/qwen3-8b-sft-merged \
    --export_size 5 \
    --export_legacy_format false

echo "[2/3] Merged model saved to /data/qwen3-8b-sft-merged/"

# 3. Deploy merged model
echo "[3/3] Deploying merged model via vLLM..."
docker stop interesting_kepler || true

docker run -d \
    --name sft-vllm \
    --runtime nvidia \
    -e NVIDIA_VISIBLE_DEVICES=0 \
    -v /data:/data \
    -p 8000:8000 \
    vllm/vllm-openai:latest \
    --model /data/qwen3-8b-sft-merged \
    --served-model-name qwen3-8b \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --tensor-parallel-size 1

echo "Waiting for SFT model to load..."
for i in $(seq 1 30); do
    if curl -s http://localhost:8000/v1/models | grep -q "qwen3-8b"; then
        echo "SFT model deployed and ready!"
        break
    fi
    sleep 5
done

echo "=== Merge + Deploy complete ==="
