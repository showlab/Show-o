#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/Show-o/show-o2
PYTHON=/root/autodl-tmp/venvs/geneval-score/bin/python
: "${RUN_DIR:?RUN_DIR is required}"
LOSS_ABLATION=${LOSS_ABLATION:-original_mask_logits}
WEIGHT_FLOW=${WEIGHT_FLOW:-0.25}
WEIGHT_TEACHER_RETENTION=${WEIGHT_TEACHER_RETENTION:-0.20}

export PYTHONPATH="${ROOT}"
export OMP_NUM_THREADS=1
export HF_HOME=/root/autodl-tmp/cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd "${ROOT}"
exec "${PYTHON}" evaluation/c6_mask2former_alignment/train_online_alignment.py \
  --run-dir "${RUN_DIR}" \
  --manifest /root/autodl-tmp/data/drsa/mask2former_alignment_5k/train.jsonl \
  --adaptation-route "${ADAPTATION_ROUTE:-g_adapter}" \
  --parameter-match-tolerance 0.01 \
  --max-steps 5000 \
  --save-every 500 \
  --preview-every 500 \
  --detector-size 800 \
  --lr-spatial 1e-5 \
  --warmup-steps 100 \
  --grad-clip 1.0 \
  --weight-mask-bce 0.01 \
  --weight-mask-dice 0.07 \
  --weight-presence 0.16 \
  --weight-relation-distill 0.5 \
  --weight-relation-pass 1.0 \
  --weight-object-detect 0.0 \
  --weight-flow "${WEIGHT_FLOW}" \
  --weight-teacher-retention "${WEIGHT_TEACHER_RETENTION}" \
  --object-threshold 0.3 \
  --object-margin 0.05 \
  --object-teacher-ratio 0.8 \
  --loss-ablation "${LOSS_ABLATION}"
