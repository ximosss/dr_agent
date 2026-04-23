#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

echo "=== Step 6: LoRA SFT Training ==="

cd "${REPO_ROOT}"

RUN_ID="${SFT_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
ADAPTER_DIR="/data/sft_output/qwen3-8b-dr-agent-${RUN_ID}"
RUN_NAME="qwen3-8b-dr-agent-sft-${RUN_ID}"
LATEST_ADAPTER_FILE="${REPO_ROOT}/training/data/latest_adapter_dir.txt"

WEAVE_EXPORT_PATH="${WEAVE_EXPORT_PATH:-}"
WEAVE_FETCH_LIMIT="${WEAVE_FETCH_LIMIT:-50000}"
WEAVE_EXTRACTED_DIR="${WEAVE_EXTRACTED_DIR:-${REPO_ROOT}/training/data/weave_extracted}"
PUBLIC_DEEPRESEARCH_DIR="${PUBLIC_DEEPRESEARCH_DIR:-/data/ximo/deepresearch-traj}"
PUBLIC_EDR_DIR="${PUBLIC_EDR_DIR:-/data/ximo/edr-200}"
PUBLIC_DEEPRESEARCH_MIN_PASS_RATE="${PUBLIC_DEEPRESEARCH_MIN_PASS_RATE:-0.5}"
PUBLIC_DEEPRESEARCH_MAX="${PUBLIC_DEEPRESEARCH_MAX:-400}"
PUBLIC_EDR_MAX="${PUBLIC_EDR_MAX:-120}"
PUBLIC_MAX_TOOL_RESPONSES="${PUBLIC_MAX_TOOL_RESPONSES:-20}"
PUBLIC_EDR_MAX_SEARCHES_PER_ITERATION="${PUBLIC_EDR_MAX_SEARCHES_PER_ITERATION:-4}"

AUGMENTED_DIR="${AUGMENTED_DIR:-${REPO_ROOT}/training/data/augmented}"
SFT_READY_DIR="${SFT_READY_DIR:-${REPO_ROOT}/training/data/sft_ready}"
AUGMENT_NUM_INTENT="${AUGMENT_NUM_INTENT:-150}"
AUGMENT_NUM_ANSWER="${AUGMENT_NUM_ANSWER:-100}"
AUGMENT_NUM_REUSE="${AUGMENT_NUM_REUSE:-40}"
AUGMENT_BASE_URL="${AUGMENT_BASE_URL:-${BASE_URL:-http://localhost:8000/v1}}"
AUGMENT_MODEL_NAME="${AUGMENT_MODEL_NAME:-${MODEL_NAME_AT_ENDPOINT:-teacher}}"
AUGMENT_BASE_KEY="${AUGMENT_BASE_KEY:-${BASE_KEY:-EMPTY}}"
SFT_SKIP_TRAIN="${SFT_SKIP_TRAIN:-0}"
SFT_NUM_TRAIN_EPOCHS="${SFT_NUM_TRAIN_EPOCHS:-}"

echo "[0/8] Syncing training environment..."
uv sync --group training --inexact

mkdir -p "${WEAVE_EXTRACTED_DIR}" "${AUGMENTED_DIR}" "${SFT_READY_DIR}"

rm -f \
    "${WEAVE_EXTRACTED_DIR}/search_trajectories.jsonl" \
    "${WEAVE_EXTRACTED_DIR}/intent_examples.jsonl" \
    "${WEAVE_EXTRACTED_DIR}/planning_examples.jsonl" \
    "${WEAVE_EXTRACTED_DIR}/answer_examples.jsonl" \
    "${WEAVE_EXTRACTED_DIR}/extraction_summary.json" \
    "${WEAVE_EXTRACTED_DIR}/public_import_summary.json" \
    "${AUGMENTED_DIR}/intent_examples.jsonl" \
    "${AUGMENTED_DIR}/planning_examples.jsonl" \
    "${AUGMENTED_DIR}/answer_examples.jsonl" \
    "${AUGMENTED_DIR}/search_reuse_examples.jsonl" \
    "${SFT_READY_DIR}/search.json" \
    "${SFT_READY_DIR}/intent.json" \
    "${SFT_READY_DIR}/planning.json" \
    "${SFT_READY_DIR}/answer.json" \
    "${SFT_READY_DIR}/search_reuse.json" \
    "${SFT_READY_DIR}/train.json" \
    "${SFT_READY_DIR}/eval.json" \
    "${SFT_READY_DIR}/conversion_stats.json" \
    "${SFT_READY_DIR}/merge_stats.json"

# 1. Download traces from Weave unless the user provided an explicit local export
if [ -z "${WEAVE_EXPORT_PATH}" ]; then
    WEAVE_EXPORT_PATH="${REPO_ROOT}/weave_export_${RUN_ID}.jsonl"
    echo "[1/8] Downloading completed traces from Weave..."
    uv run python -m training.download_weave_export \
        --output-path "${WEAVE_EXPORT_PATH}" \
        --fetch-limit "${WEAVE_FETCH_LIMIT}"
elif [ ! -f "${WEAVE_EXPORT_PATH}" ]; then
    echo "ERROR: WEAVE_EXPORT_PATH does not exist: ${WEAVE_EXPORT_PATH}"
    exit 1
fi

echo "[2/8] Extracting trajectories from local Weave export..."
echo "  Weave export: ${WEAVE_EXPORT_PATH}"
uv run python -m training.extract_weave \
    --weave-path "${WEAVE_EXPORT_PATH}" \
    --output-dir "${WEAVE_EXTRACTED_DIR}"

echo "[2b/8] Importing public search trajectories..."
echo "  deepresearch-traj: ${PUBLIC_DEEPRESEARCH_DIR}"
echo "  edr-200: ${PUBLIC_EDR_DIR}"
uv run python -m training.import_public_trajectories \
    --output-path "${WEAVE_EXTRACTED_DIR}/search_trajectories.jsonl" \
    --deepresearch-dir "${PUBLIC_DEEPRESEARCH_DIR}" \
    --edr-dir "${PUBLIC_EDR_DIR}" \
    --min-deepresearch-pass-rate "${PUBLIC_DEEPRESEARCH_MIN_PASS_RATE}" \
    --max-deepresearch "${PUBLIC_DEEPRESEARCH_MAX}" \
    --max-edr "${PUBLIC_EDR_MAX}" \
    --max-tool-responses "${PUBLIC_MAX_TOOL_RESPONSES}" \
    --max-edr-searches-per-iteration "${PUBLIC_EDR_MAX_SEARCHES_PER_ITERATION}"

echo "[3/8] Running augmentation with the teacher model..."
echo "  Teacher endpoint: ${AUGMENT_BASE_URL} (${AUGMENT_MODEL_NAME})"
echo "  Requested augmentation counts: intent=${AUGMENT_NUM_INTENT}, answer=${AUGMENT_NUM_ANSWER}, reuse=${AUGMENT_NUM_REUSE}"
if [ "${AUGMENT_NUM_INTENT}" -gt 0 ] || [ "${AUGMENT_NUM_ANSWER}" -gt 0 ] || [ "${AUGMENT_NUM_REUSE}" -gt 0 ]; then
    if ! curl --max-time 15 -fsS "${AUGMENT_BASE_URL}/models" >/dev/null; then
        echo "ERROR: augmentation is enabled but the teacher endpoint is not reachable at ${AUGMENT_BASE_URL}"
        exit 1
    fi
    BASE_URL="${AUGMENT_BASE_URL}" MODEL_NAME_AT_ENDPOINT="${AUGMENT_MODEL_NAME}" BASE_KEY="${AUGMENT_BASE_KEY}" \
        uv run python -m training.augment_data \
        --weave-dir "${WEAVE_EXTRACTED_DIR}" \
        --output-dir "${AUGMENTED_DIR}" \
        --num-intent "${AUGMENT_NUM_INTENT}" \
        --num-answer "${AUGMENT_NUM_ANSWER}" \
        --num-reuse "${AUGMENT_NUM_REUSE}"
else
    echo "  Skipping augmentation because all requested counts are zero."
    : > "${AUGMENTED_DIR}/intent_examples.jsonl"
    : > "${AUGMENTED_DIR}/planning_examples.jsonl"
    : > "${AUGMENTED_DIR}/answer_examples.jsonl"
    : > "${AUGMENTED_DIR}/search_reuse_examples.jsonl"
fi

echo "[4/8] Converting extracted + augmented data to ShareGPT format..."
uv run python -m training.convert_sharegpt \
    --weave-dir "${WEAVE_EXTRACTED_DIR}" \
    --augmented-dir "${AUGMENTED_DIR}" \
    --output-dir "${SFT_READY_DIR}"

echo "[5/8] Building SFT train/eval splits..."
uv run python -m training.merge_and_split --sft-dir "${SFT_READY_DIR}"

echo "[6/8] Verifying dataset..."
uv run python -c "
import json
with open('${SFT_READY_DIR}/train.json') as f:
    train_data = json.load(f)
with open('${SFT_READY_DIR}/eval.json') as f:
    eval_data = json.load(f)
print(f'Training examples: {len(train_data)}')
print(f'Eval examples: {len(eval_data)}')
with open('${SFT_READY_DIR}/conversion_stats.json') as f:
    conversion_stats = json.load(f)
print(f'Converted sources: {conversion_stats.get(\"sources\", {})}')
# Check first example
ex = train_data[0]
print(f'First example keys: {list(ex.keys())}')
print(f'Conversation turns: {len(ex[\"conversations\"])}')
for turn in ex['conversations'][:3]:
    print(f'  {turn[\"from\"]}: {turn[\"value\"][:80]}...')
"

if [ "${SFT_SKIP_TRAIN}" = "1" ]; then
    echo "[7/8] Skipping LLaMA-Factory training because SFT_SKIP_TRAIN=1"
    exit 0
fi

echo "[7/8] Starting SFT training on GPU 1..."
echo "  Run ID: ${RUN_ID}"
echo "  Adapter output: ${ADAPTER_DIR}"
WANDB_PROJECT="${WANDB_PROJECT:-deep_research_agent}" CUDA_VISIBLE_DEVICES=1 uv run --group training \
    llamafactory-cli train \
    "${REPO_ROOT}/training/configs/qwen3_8b_sft.yaml" \
    output_dir="${ADAPTER_DIR}" \
    run_name="${RUN_NAME}" \
    ${SFT_NUM_TRAIN_EPOCHS:+num_train_epochs="${SFT_NUM_TRAIN_EPOCHS}"}

printf '%s\n' "${ADAPTER_DIR}" > "${LATEST_ADAPTER_FILE}"

echo "[8/8] Training complete."
echo "=== SFT Training complete ==="
echo "Output: ${ADAPTER_DIR}"
