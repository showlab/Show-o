#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/Show-o/show-o2
WORKSPACE=/root/autodl-tmp
SESSION_NAME=${SESSION_NAME:-showo2_mask_retention_flow_factorial}
RUN_ID=${RUN_ID:-mask-retention-flow-factorial-$(date -u +%Y%m%dT%H%M%SZ)}
RUN_ROOT=${RUN_ROOT:-${WORKSPACE}/runs/showo2_generation_factorial/${RUN_ID}}
PURE_ROOT=${PURE_ROOT:?PURE_ROOT is required}
UPSTREAM_BENCHMARK_ROOT=${UPSTREAM_BENCHMARK_ROOT:?UPSTREAM_BENCHMARK_ROOT is required}
TRAIN_MANIFEST=${TRAIN_MANIFEST:-${WORKSPACE}/data/drsa/bprime_coco_geneval_v2/train.jsonl}
MAX_STEPS=${MAX_STEPS:-17100}
SAVE_STEPS=${SAVE_STEPS:-100,200,500,1000,2000,4000,5000,6000,8000,10000,13000,17100}

if tmux has-session -t "=${SESSION_NAME}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION_NAME}" >&2
  exit 1
fi

mkdir -p "${RUN_ROOT}"
tmux new-session -d -s "${SESSION_NAME}" \
  "cd ${ROOT@Q} && set -o pipefail; RUN_ROOT=${RUN_ROOT@Q} PURE_ROOT=${PURE_ROOT@Q} UPSTREAM_BENCHMARK_ROOT=${UPSTREAM_BENCHMARK_ROOT@Q} TRAIN_MANIFEST=${TRAIN_MANIFEST@Q} MAX_STEPS=${MAX_STEPS@Q} SAVE_STEPS=${SAVE_STEPS@Q} ${ROOT@Q}/evaluation/c6_mask2former_alignment/run_mask_retention_flow_factorial_followup.sh 2>&1 | tee -a ${RUN_ROOT@Q}/queue.log; exec bash"

printf 'session=%s\nrun_root=%s\nmonitor=%s %s %s\n' \
  "${SESSION_NAME}" "${RUN_ROOT}" \
  "${ROOT}/evaluation/c6_mask2former_alignment/watch_mask_retention_flow_factorial.sh" \
  "${RUN_ROOT}" "${PURE_ROOT}"
