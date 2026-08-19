#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/root/autodl-tmp/Show-o/show-o2
WORKSPACE=/root/autodl-tmp
PYTHON=${PYTHON:-${WORKSPACE}/venvs/geneval-score/bin/python}
export OMP_NUM_THREADS=${QUEUE_OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${QUEUE_MKL_NUM_THREADS:-4}
SESSION=${SESSION:-showo2_multiaxis_ab_benchmarks}
RUN_ID=${RUN_ID:-multiaxis-ab-five-checkpoints-$(date -u +%Y%m%dT%H%M%SZ)}
RUN_ROOT=${RUN_ROOT:-${WORKSPACE}/runs/showo2_multiaxis_ab/${RUN_ID}}

S0_MASK_ROOT=${S0_MASK_ROOT:-${WORKSPACE}/runs/showo2_generation_route_pair/generation-route-pair-masklogits-full17100-ckpt12-20260814T113818Z}
S0_MASK_FLOW_ROOT=${S0_MASK_FLOW_ROOT:-${WORKSPACE}/runs/showo2_generation_factorial/mask-flow-teacher-replay-factorial-full17100-ckpt12-20260814T113818Z/mask_flow}
S1_MASK_ROOT=${S1_MASK_ROOT:-${WORKSPACE}/runs/showo2_generation_route_pair/mask-logits-multiaxis-full17100-20260815T072000Z}
S1_MASK_FLOW_ROOT=${S1_MASK_FLOW_ROOT:-${WORKSPACE}/runs/showo2_generation_route_pair/mask-flow-multiaxis-full17100-20260815T113653Z}
BENCHMARK_STEPS=${BENCHMARK_STEPS:-100,500,1000,5000,17100}
WAIT_ROOT=${WAIT_ROOT:-${S1_MASK_FLOW_ROOT}}
WAIT_MARKER=${WAIT_MARKER:-QUEUE_COMPLETED}
WAIT_LABEL=${WAIT_LABEL:-new multi-axis Mask-logits+Flow training}

mkdir -p "${RUN_ROOT}"
if tmux has-session -t "=${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 1
fi

command=(
  "${PYTHON}" -u
  "${ROOT}/evaluation/c6_mask2former_alignment/run_multiaxis_ab_checkpoint_queue.py"
  --run-root "${RUN_ROOT}"
  --s0-mask-root "${S0_MASK_ROOT}"
  --s0-mask-flow-root "${S0_MASK_FLOW_ROOT}"
  --s1-mask-root "${S1_MASK_ROOT}"
  --s1-mask-flow-root "${S1_MASK_FLOW_ROOT}"
  --wait-root "${WAIT_ROOT}"
  --wait-marker "${WAIT_MARKER}"
  --wait-label "${WAIT_LABEL}"
  --benchmark-steps "${BENCHMARK_STEPS}"
)
printf -v quoted_command '%q ' "${command[@]}"
printf -v quoted_root '%q' "${ROOT}"
printf -v quoted_log '%q' "${RUN_ROOT}/queue.log"

tmux new-session -d -s "${SESSION}" \
  "cd ${quoted_root} && set -o pipefail; ${quoted_command}2>&1 | tee -a ${quoted_log}; exec bash"

printf 'session=%s\nrun_root=%s\n' "${SESSION}" "${RUN_ROOT}"
printf 'watch=%q %q 5\n' \
  "${ROOT}/evaluation/c6_mask2former_alignment/watch_multiaxis_ab_checkpoint_queue.sh" \
  "${RUN_ROOT}"
