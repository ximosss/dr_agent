#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

echo "=== Step 7: LoRA Merge + Deploy ==="

echo "[0/3] Syncing training environment..."
uv sync --group training --inexact

ADAPTER_DIR="${SFT_ADAPTER_DIR:-}"
MERGED_MODEL_DIR="${MERGED_MODEL_DIR:-/data/qwen3-8b-sft-merged}"
MERGED_MODEL_NAME="${MERGED_MODEL_NAME:-qwen3-8b}"
DOCKER_MERGED_MODEL_DIR="${DOCKER_MERGED_MODEL_DIR:-${MERGED_MODEL_DIR/#\/data/\/datas}}"
if [ -z "${ADAPTER_DIR}" ] && [ -f training/data/latest_adapter_dir.txt ]; then
    ADAPTER_DIR="$(cat training/data/latest_adapter_dir.txt)"
fi

if [ -z "${ADAPTER_DIR}" ]; then
    echo "ERROR: set SFT_ADAPTER_DIR or run training first"
    exit 1
fi

BEST_CKPT="$(ls -d "${ADAPTER_DIR}"/checkpoint-* 2>/dev/null | sort -V | tail -1)"
if [ -n "${BEST_CKPT}" ]; then
    ADAPTER_DIR="${BEST_CKPT}"
fi

echo "[1/3] Merging LoRA weights..."
CUDA_VISIBLE_DEVICES=1 uv run --group training \
  llamafactory-cli export \
  --model_name_or_path /data/qwen3-8b \
  --adapter_name_or_path "${ADAPTER_DIR}" \
  --template qwen3 \
  --finetuning_type lora \
  --export_dir "${MERGED_MODEL_DIR}" \
  --export_size 5 \
  --export_legacy_format false

echo "[2/3] Stopping anything on port 8000..."
docker ps --filter "publish=8000" --format '{{.Names}}' | xargs -r docker stop
docker rm -f sft-vllm 2>/dev/null || true

echo "[3/3] Starting merged model..."
docker run -d \
  --name sft-vllm \
  --gpus '"device=0,1"' \
  --ipc=host \
  -v /data:/datas \
  -p 8000:8000 \
  vllm/vllm-openai:latest \
  "${DOCKER_MERGED_MODEL_DIR}" \
  --served-model-name "${MERGED_MODEL_NAME}" \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --tensor-parallel-size 1

echo "Merged model started on http://localhost:8000/v1"
