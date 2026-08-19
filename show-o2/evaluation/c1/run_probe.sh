#!/usr/bin/env bash
set -euo pipefail

SHOWO2_ROOT="${SHOWO2_ROOT:-/root/autodl-tmp/Show-o/show-o2}"
PYTHON="${SHOWO2_EVAL_PYTHON:-/root/autodl-tmp/venvs/showo2-eval/bin/python}"
C0_RUN="${C0_RUN:-/root/autodl-tmp/runs/showo2_c0/c0-full-20260802T125630Z}"
GENEVAL_RESULTS="${GENEVAL_RESULTS:-${C0_RUN}/generation/geneval/results/results.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/autodl-tmp/runs/showo2_c1}"
RUN_ID="${RUN_ID:-c1-conflict-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_DIR="${RUN_DIR:-${OUTPUT_ROOT}/${RUN_ID}}"
PER_TAG="${PER_TAG:-8}"
LIMIT="${LIMIT:-0}"
SEED="${SEED:-2026}"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore}"

mkdir -p "${RUN_DIR}/logs"

"${PYTHON}" "${SHOWO2_ROOT}/evaluation/c1/build_geneval_probe.py" \
  --results "${GENEVAL_RESULTS}" \
  --base-prompts "${SHOWO2_ROOT}/prompts/geneval_meta_data.jsonl" \
  --output "${RUN_DIR}/manifest.jsonl" \
  --per-tag "${PER_TAG}" \
  --seed "${SEED}" \
  2>&1 | tee "${RUN_DIR}/logs/manifest.log"

probe_args=(
  --manifest "${RUN_DIR}/manifest.jsonl"
  --output-dir "${RUN_DIR}"
  --showo-root "${SHOWO2_ROOT}"
  --config "${SHOWO2_CONFIG:-${SHOWO2_ROOT}/configs/showo2_1.5b_demo_432x432.yaml}"
  --model "${SHOWO2_MODEL:-showlab/show-o2-1.5B}"
  --vae-path "${SHOWO2_VAE_PATH:-${SHOWO2_ROOT}/Wan2.1_VAE.pth}"
  --device "${DEVICE:-cuda}"
  --weight-type "${WEIGHT_TYPE:-bfloat16}"
  --vae-weight-type "${VAE_WEIGHT_TYPE:-float32}"
  --seed "${SEED}"
)
if [[ "${LIMIT}" -gt 0 ]]; then
  probe_args+=(--limit "${LIMIT}")
fi

"${PYTHON}" "${SHOWO2_ROOT}/evaluation/c1/gradient_probe.py" "${probe_args[@]}" \
  2>&1 | tee -a "${RUN_DIR}/logs/probe.log"

"${PYTHON}" "${SHOWO2_ROOT}/evaluation/c1/summarize_probe.py" \
  --samples "${RUN_DIR}/samples.jsonl" \
  --output-dir "${RUN_DIR}" \
  --seed "${SEED}" \
  2>&1 | tee -a "${RUN_DIR}/logs/summary.log"

echo "C1 gradient-conflict probe completed: ${RUN_DIR}"
