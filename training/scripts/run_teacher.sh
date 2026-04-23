#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

echo "=== Step 2: Teacher Trajectory Collection ==="
echo "Assuming teacher-vllm is already running on http://localhost:8001/v1"
STOP_TEACHER_ON_EXIT="${STOP_TEACHER_ON_EXIT:-0}"
FRAMES_TRAIN_N="${FRAMES_TRAIN_N:-50}"
SIMPLEQA_TRAIN_N="${SIMPLEQA_TRAIN_N:-50}"
GAIA_TRAIN_N="${GAIA_TRAIN_N:-20}"

# You need to start the teacher manually, for example:
# docker run --name teacher-vllm \
#   --gpus '"device=0,1"' \
#   --ipc=host \
#   -e NCCL_P2P_DISABLE=1 \
#   -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
#   -v /data:/datas \
#   -p 8001:8000 \
#   vllm/vllm-openai:latest \
#   /datas/qwen3.5-122b-a10b-gptq-int4 \
#   --served-model-name teacher \
#   --language-model-only \
#   --tensor-parallel-size 2 \
#   --max-model-len 4096 \
#   --gpu-memory-utilization 0.80 \
#   --quantization moe_wna16 \
#   --reasoning-parser qwen3 \
#   --enable-auto-tool-choice \
#   --tool-call-parser qwen3_coder \
#   --enforce-eager \
#   --disable-custom-all-reduce

export BASE_URL="http://localhost:8001/v1"
export MODEL_NAME_AT_ENDPOINT="teacher"

uv run python training/collect_teacher.py -b frames -n "${FRAMES_TRAIN_N}"
uv run python training/collect_teacher.py -b simpleqa -n "${SIMPLEQA_TRAIN_N}"
uv run python training/collect_teacher.py -b gaia -n "${GAIA_TRAIN_N}" --text-only

if [ "${STOP_TEACHER_ON_EXIT}" = "1" ]; then
    docker stop teacher-vllm
    docker rm teacher-vllm
fi

echo "Teacher collection complete."
echo "Next step: bash training/scripts/run_training.sh"
