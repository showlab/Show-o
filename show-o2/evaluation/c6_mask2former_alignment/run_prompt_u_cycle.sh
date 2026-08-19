#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/Show-o/show-o2
PYTHON=/root/autodl-tmp/venvs/geneval-score/bin/python
: "${RUN_DIR:?RUN_DIR is required}"
: "${WEIGHT_U_TASK:?WEIGHT_U_TASK is required; calibrate it from component gradients}"

MANIFEST=${MANIFEST:-/root/autodl-tmp/data/drsa/mask2former_alignment_5k/train.jsonl}
MAX_STEPS=${MAX_STEPS:-5000}
SAVE_EVERY=${SAVE_EVERY:-500}
SAVE_STEPS=${SAVE_STEPS:-}
PREVIEW_EVERY=${PREVIEW_EVERY:-500}
LOSS_ABLATION=${LOSS_ABLATION:-original_mask_logits}
WEIGHT_FLOW=${WEIGHT_FLOW:-0.25}
WEIGHT_TEACHER_RETENTION=${WEIGHT_TEACHER_RETENTION:-0.20}
U_CYCLE_RAMP_STEPS=${U_CYCLE_RAMP_STEPS:-100}
U_TASK_MODE=${U_TASK_MODE:-relation_choice}
U_GATE_POLICY=${U_GATE_POLICY:-strict}
U_GATE_MARGIN=${U_GATE_MARGIN:-0.0}
ADAPTATION_ROUTE=${ADAPTATION_ROUTE:-g_adapter}
SEED=${SEED:-20260808}
ROLLOUT_STEPS=${ROLLOUT_STEPS:-4}
PARAMETER_MATCH_TOLERANCE=${PARAMETER_MATCH_TOLERANCE:-0.00001}
PROBE_COMPONENT_GRADS=${PROBE_COMPONENT_GRADS:-0}

extra_args=()
if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
  extra_args+=(--resume-checkpoint "${RESUME_CHECKPOINT}")
fi
if [[ "${PROBE_COMPONENT_GRADS}" == "1" ]]; then
  extra_args+=(--probe-component-grads)
fi

export PYTHONPATH="${ROOT}"
export OMP_NUM_THREADS=1
export HF_HOME=/root/autodl-tmp/cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd "${ROOT}"
exec "${PYTHON}" evaluation/c6_mask2former_alignment/train_online_alignment.py \
  --run-dir "${RUN_DIR}" \
  --manifest "${MANIFEST}" \
  --adaptation-route "${ADAPTATION_ROUTE}" \
  --parameter-match-tolerance "${PARAMETER_MATCH_TOLERANCE}" \
  --max-steps "${MAX_STEPS}" \
  --save-every "${SAVE_EVERY}" \
  --save-steps "${SAVE_STEPS}" \
  --preview-every "${PREVIEW_EVERY}" \
  --seed "${SEED}" \
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
  --weight-u-task "${WEIGHT_U_TASK}" \
  --u-task-mode "${U_TASK_MODE}" \
  --u-cycle-ramp-steps "${U_CYCLE_RAMP_STEPS}" \
  --u-gate-policy "${U_GATE_POLICY}" \
  --u-gate-margin "${U_GATE_MARGIN}" \
  --object-threshold 0.3 \
  --object-margin 0.05 \
  --object-teacher-ratio 0.8 \
  --loss-ablation "${LOSS_ABLATION}" \
  --rollout-steps "${ROLLOUT_STEPS}" \
  --rollout-schedule-points 50 \
  --rollout-time-shifting-factor 3.0 \
  --rollout-guidance-scale 0.0 \
  --no-u-cycle-use-drsa-understanding \
  "${extra_args[@]}"
