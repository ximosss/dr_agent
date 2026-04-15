#!/usr/bin/env bash
set -euo pipefail

echo "=== Step 2: Teacher Data Collection ==="

TEACHER_MODEL="/data/qwen3.5-122b-a10b-gptq-int4"
TEACHER_PORT=8001

# Check if teacher model is downloaded
if [ ! -d "$TEACHER_MODEL" ]; then
    echo "ERROR: Teacher model not found at $TEACHER_MODEL"
    echo "Download it first:"
    echo "  huggingface-cli download Qwen/Qwen3.5-122B-A10B-GPTQ-Int4 --local-dir $TEACHER_MODEL"
    exit 1
fi

# 1. Stop the existing Qwen3-8B container
echo "[1/5] Stopping Qwen3-8B container..."
docker stop interesting_kepler || true
sleep 3

# 2. Start teacher model with TP=2 (both GPUs)
echo "[2/5] Starting teacher model on 2xGPU (TP=2)..."
docker run -d \
    --name teacher-vllm \
    --runtime nvidia \
    --gpus all \
    -v /data:/data \
    -p ${TEACHER_PORT}:8000 \
    vllm/vllm-openai:latest \
    --model /data/qwen3.5-122b-a10b-gptq-int4 \
    --served-model-name teacher \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --tensor-parallel-size 2 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.90 \
    --quantization gptq

echo "Waiting for teacher model to load (~2-5 min)..."
for i in $(seq 1 60); do
    if curl -s http://localhost:${TEACHER_PORT}/v1/models | grep -q "teacher"; then
        echo "Teacher model is ready!"
        break
    fi
    sleep 5
done

# 3. Run data collection (overriding env vars)
echo "[3/5] Collecting teacher trajectories..."
cd /home/ubuntu/workspace/dr_agent

export BASE_URL="http://localhost:${TEACHER_PORT}/v1"
export MODEL_NAME_AT_ENDPOINT="teacher"

# Collect from each benchmark (reserved train split only)
uv run python training/collect_teacher.py -b frames -n 40
uv run python training/collect_teacher.py -b simpleqa -n 20
uv run python training/collect_teacher.py -b gaia -n 10 --text-only

# 4. Run augmentation (also uses teacher model, no Tavily)
echo "[4/5] Running synthetic data augmentation..."
uv run python training/augment_data.py --num-intent 150 --num-answer 100 --num-reuse 40

# 5. Stop teacher and restore Qwen3-8B
echo "[5/5] Cleaning up..."
docker stop teacher-vllm || true
docker rm teacher-vllm || true
docker start interesting_kepler

echo "Waiting for Qwen3-8B to restore..."
for i in $(seq 1 30); do
    if curl -s http://localhost:8000/v1/models | grep -q "qwen3-8b"; then
        echo "Qwen3-8B restored!"
        break
    fi
    sleep 5
done

echo "Export the new Weave traces to a local weave_export_*.jsonl before rerunning training/extract_weave.py."
echo "=== Teacher data collection complete ==="
