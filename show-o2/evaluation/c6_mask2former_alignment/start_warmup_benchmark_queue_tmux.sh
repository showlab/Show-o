#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/Show-o/show-o2
: "${RUN_ROOT:?RUN_ROOT is required}"
: "${TRAIN_ROOT:?TRAIN_ROOT is required}"
SESSION_NAME=${SESSION_NAME:-showo2_warmup_benchmarks}
PYTHON=/root/autodl-tmp/venvs/showo2-eval/bin/python

if tmux has-session -t "=${SESSION_NAME}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION_NAME}" >&2
  exit 1
fi
mkdir -p "${RUN_ROOT}"
tmux new-session -d -s "${SESSION_NAME}" \
  "cd '${ROOT}' && '${PYTHON}' -u evaluation/c6_mask2former_alignment/run_warmup_benchmark_queue.py --run-root '${RUN_ROOT}' --train-root '${TRAIN_ROOT}' 2>&1 | tee -a '${RUN_ROOT}/queue.log'"
echo "session=${SESSION_NAME}"
echo "run=${RUN_ROOT}"
echo "monitor=${PYTHON} ${ROOT}/evaluation/c6_mask2former_alignment/monitor_warmup_benchmark_queue.py ${RUN_ROOT} --session ${SESSION_NAME} --watch"
