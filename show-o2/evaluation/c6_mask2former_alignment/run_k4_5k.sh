#!/usr/bin/env bash
set -euo pipefail

SHOWO2_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKSPACE_ROOT="$(cd "${SHOWO2_ROOT}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${WORKSPACE_ROOT}/venvs/geneval-score/bin/python}"
RUN_ID="${RUN_ID:-mask-flow-k4-5k-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_DIR="${RUN_DIR:-${WORKSPACE_ROOT}/runs/showo2_mask2former_alignment/${RUN_ID}}"

export PYTHONPATH="${WORKSPACE_ROOT}/mmdetection-2.28.2:${SHOWO2_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DIFFUSERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "${RUN_DIR}"
exec "${PYTHON_BIN}" -u \
  "${SHOWO2_ROOT}/evaluation/c6_mask2former_alignment/train_online_alignment.py" \
  --run-dir "${RUN_DIR}" \
  --manifest "${WORKSPACE_ROOT}/data/drsa/mask2former_alignment_5k/train.jsonl" \
  --adaptation-route "${ADAPTATION_ROUTE:-g_adapter}" \
  --parameter-match-tolerance 0.01 \
  --max-steps 5000 \
  --save-every 500 \
  --preview-every 500 \
  --seed 20260808 \
  --lr-spatial 1e-5 \
  --weight-decay 0.01 \
  --warmup-steps 100 \
  --grad-clip 1.0 \
  --loss-ablation original_mask_logits \
  --weight-mask-bce 0.01 \
  --weight-mask-dice 0.07 \
  --weight-presence 0.16 \
  --weight-relation-distill 0.5 \
  --weight-relation-pass 1.0 \
  --weight-object-detect 1.0 \
  --weight-flow 0.25 \
  --weight-teacher-retention 0.20 \
  --weight-u-task 0.0 \
  --t-min 0.10 \
  --t-max 0.80 \
  --rollout-steps 4 \
  --rollout-schedule-points 50 \
  --rollout-time-shifting-factor 3.0 \
  --rollout-guidance-scale 0.0
