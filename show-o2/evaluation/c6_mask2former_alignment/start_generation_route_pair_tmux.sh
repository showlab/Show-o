#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/Show-o/show-o2
WORKSPACE=/root/autodl-tmp
SESSION_NAME=${SESSION_NAME:-showo2_generation_route_pair}
RUN_ID=${RUN_ID:-generation-route-pair-$(date -u +%Y%m%dT%H%M%SZ)}
RUN_ROOT=${RUN_ROOT:-${WORKSPACE}/runs/showo2_generation_route_pair/${RUN_ID}}
WAIT_FOR_BENCHMARK_ROOT=${WAIT_FOR_BENCHMARK_ROOT:-}
MAX_STEPS=${MAX_STEPS:-5000}
EVAL_G_ADAPTER=${EVAL_G_ADAPTER:-0}
EVAL_SEED=${EVAL_SEED:-20260810}
TRAIN_SEED=${TRAIN_SEED:-20260808}
WEIGHT_U_TASK=${WEIGHT_U_TASK:-1.0}
WEIGHT_FLOW=${WEIGHT_FLOW:-0.25}
WEIGHT_TEACHER_RETENTION=${WEIGHT_TEACHER_RETENTION:-0.20}
ROLLOUT_STEPS=${ROLLOUT_STEPS:-4}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-zero_init_direct_cycle}
TRAIN_MANIFEST=${TRAIN_MANIFEST:-${WORKSPACE}/data/drsa/mask2former_alignment_5k/train.jsonl}
SAVE_STEPS=${SAVE_STEPS:-}

if tmux has-session -t "=${SESSION_NAME}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION_NAME}" >&2
  exit 1
fi

mkdir -p "${RUN_ROOT}"
tmux new-session -d -s "${SESSION_NAME}" \
  "cd ${ROOT@Q} && set -o pipefail; RUN_ID=${RUN_ID@Q} RUN_ROOT=${RUN_ROOT@Q} WAIT_FOR_BENCHMARK_ROOT=${WAIT_FOR_BENCHMARK_ROOT@Q} MAX_STEPS=${MAX_STEPS@Q} EVAL_G_ADAPTER=${EVAL_G_ADAPTER@Q} EVAL_SEED=${EVAL_SEED@Q} TRAIN_SEED=${TRAIN_SEED@Q} WEIGHT_U_TASK=${WEIGHT_U_TASK@Q} WEIGHT_FLOW=${WEIGHT_FLOW@Q} WEIGHT_TEACHER_RETENTION=${WEIGHT_TEACHER_RETENTION@Q} ROLLOUT_STEPS=${ROLLOUT_STEPS@Q} EXPERIMENT_NAME=${EXPERIMENT_NAME@Q} TRAIN_MANIFEST=${TRAIN_MANIFEST@Q} SAVE_STEPS=${SAVE_STEPS@Q} ${ROOT@Q}/evaluation/c6_mask2former_alignment/run_generation_route_pair_queue.sh 2>&1 | tee -a ${RUN_ROOT@Q}/queue.log; exec bash"

printf 'session=%s\nrun_root=%s\nmonitor=%s %s\n' \
  "${SESSION_NAME}" "${RUN_ROOT}" \
  "${ROOT}/evaluation/c6_mask2former_alignment/watch_generation_route_pair.sh" \
  "${RUN_ROOT}"
