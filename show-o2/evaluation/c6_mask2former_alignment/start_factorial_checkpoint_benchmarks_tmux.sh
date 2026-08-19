#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/Show-o/show-o2
WORKSPACE=/root/autodl-tmp
SESSION_NAME=${SESSION_NAME:-showo2_factorial_checkpoint_benchmarks}
RUN_ID=${RUN_ID:-factorial-checkpoint-benchmarks-$(date -u +%Y%m%dT%H%M%SZ)}
RUN_ROOT=${RUN_ROOT:-${WORKSPACE}/runs/showo2_generation_factorial/${RUN_ID}}
PURE_ROOT=${PURE_ROOT:?PURE_ROOT is required}
FACTORIAL_ROOT=${FACTORIAL_ROOT:?FACTORIAL_ROOT is required}
SAVE_STEPS=${SAVE_STEPS:-100,200,500,1000,2000,4000,5000,6000,8000,10000,13000,17100}
BENCHMARK_STEPS=${BENCHMARK_STEPS:-100,500,1000,5000,17100}

if tmux has-session -t "=${SESSION_NAME}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION_NAME}" >&2
  exit 1
fi

mkdir -p "${RUN_ROOT}"
tmux new-session -d -s "${SESSION_NAME}" \
  "cd ${ROOT@Q} && set -o pipefail; RUN_ROOT=${RUN_ROOT@Q} PURE_ROOT=${PURE_ROOT@Q} FACTORIAL_ROOT=${FACTORIAL_ROOT@Q} SAVE_STEPS=${SAVE_STEPS@Q} BENCHMARK_STEPS=${BENCHMARK_STEPS@Q} ${ROOT@Q}/evaluation/c6_mask2former_alignment/run_factorial_checkpoint_benchmarks_followup.sh 2>&1 | tee -a ${RUN_ROOT@Q}/queue.log; exec bash"

printf 'session=%s\nrun_root=%s\nmonitor=%s %s %s %s\n' \
  "${SESSION_NAME}" "${RUN_ROOT}" \
  "${ROOT}/evaluation/c6_mask2former_alignment/watch_factorial_checkpoint_benchmarks.sh" \
  "${FACTORIAL_ROOT}" "${PURE_ROOT}" "${RUN_ROOT}"
