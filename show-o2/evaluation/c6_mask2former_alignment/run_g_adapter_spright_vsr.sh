#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/Show-o/show-o2
PYTHON=/root/autodl-tmp/venvs/geneval-score/bin/python
: "${RUN_DIR:?RUN_DIR is required}"

export PYTHONPATH="${ROOT}"
export OMP_NUM_THREADS=1
export HF_HOME=/root/autodl-tmp/cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

args=(
  evaluation/c6_mask2former_alignment/train_g_adapter_spright_vsr.py
  --run-dir "${RUN_DIR}"
  --data-root /root/autodl-tmp/data/drsa
  --spright-limit "${SPRIGHT_LIMIT:-40504}"
  --vsr-limit "${VSR_LIMIT:-3489}"
  --learning-rate "${LEARNING_RATE:-1e-5}"
  --vsr-loss-weight "${VSR_LOSS_WEIGHT:-0.01}"
  --warmup-steps "${WARMUP_STEPS:-100}"
  --early-save-until "${EARLY_SAVE_UNTIL:-1000}"
  --early-save-every "${EARLY_SAVE_EVERY:-100}"
  --late-save-every "${LATE_SAVE_EVERY:-2000}"
  --seed "${SEED:-20260813}"
  --device "${DEVICE:-cuda:0}"
)

if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
  args+=(--resume-checkpoint "${RESUME_CHECKPOINT}")
fi
if [[ -n "${STOP_AFTER_STEP:-}" ]]; then
  args+=(--stop-after-step "${STOP_AFTER_STEP}")
fi

cd "${ROOT}"
exec "${PYTHON}" "${args[@]}"
