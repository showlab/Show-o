#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/root/autodl-tmp/Show-o/show-o2
WORKSPACE=/root/autodl-tmp
PYTHON=${PYTHON:-${WORKSPACE}/venvs/geneval-score/bin/python}
TRAINING_ROOT=${TRAINING_ROOT:-${WORKSPACE}/runs/showo2_tifa_cycle/external-tifa12-k4-four-route-20260816T173500Z}
SESSION=${SESSION:-showo2_tifa12_posttrain_benchmarks}
RUN_ID=${RUN_ID:-external-tifa12-six-checkpoints-$(date -u +%Y%m%dT%H%M%SZ)}
RUN_ROOT=${RUN_ROOT:-${WORKSPACE}/runs/showo2_tifa_cycle_benchmarks/${RUN_ID}}
BENCHMARK_STEPS=${BENCHMARK_STEPS:-100,500,1000,6000,17000,34012}

export OMP_NUM_THREADS=${QUEUE_OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${QUEUE_MKL_NUM_THREADS:-4}

for required in \
  "${PYTHON}" \
  "${TRAINING_ROOT}/queue_status.json" \
  "${ROOT}/evaluation/c6_mask2former_alignment/run_tifa12_checkpoint_benchmark_queue.py"; do
  [[ -e "${required}" ]] || { echo "missing required resource: ${required}" >&2; exit 1; }
done

if tmux has-session -t "=${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 1
fi

mkdir -p "${RUN_ROOT}"
"${PYTHON}" -u "${ROOT}/evaluation/c6_mask2former_alignment/run_tifa12_checkpoint_benchmark_queue.py" \
  --run-root "${RUN_ROOT}" \
  --training-root "${TRAINING_ROOT}" \
  --benchmark-steps "${BENCHMARK_STEPS}" \
  --plan-only > "${RUN_ROOT}/launch_plan.json"

command=(
  "${PYTHON}" -u
  "${ROOT}/evaluation/c6_mask2former_alignment/run_tifa12_checkpoint_benchmark_queue.py"
  --run-root "${RUN_ROOT}"
  --training-root "${TRAINING_ROOT}"
  --benchmark-steps "${BENCHMARK_STEPS}"
)
printf -v quoted_command '%q ' "${command[@]}"
printf -v quoted_root '%q' "${ROOT}"
printf -v quoted_log '%q' "${RUN_ROOT}/queue.log"

tmux new-session -d -s "${SESSION}" \
  "cd ${quoted_root} && set -o pipefail; ${quoted_command}2>&1 | tee -a ${quoted_log}; exec bash"

printf 'session=%s\nrun_root=%s\ntraining_root=%s\nsteps=%s\n' \
  "${SESSION}" "${RUN_ROOT}" "${TRAINING_ROOT}" "${BENCHMARK_STEPS}"
printf 'watch=bash %q %q 5\n' \
  "${ROOT}/evaluation/c6_mask2former_alignment/watch_tifa12_checkpoint_benchmarks.sh" \
  "${RUN_ROOT}"
