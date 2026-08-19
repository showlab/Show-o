#!/usr/bin/env bash
set -euo pipefail

SHOWO2_ROOT="${SHOWO2_ROOT:-/root/autodl-tmp/Show-o/show-o2}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/autodl-tmp/runs/showo2_c1}"
RUN_ID="${RUN_ID:-c1-conflict-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_DIR="${RUN_DIR:-${OUTPUT_ROOT}/${RUN_ID}}"
TMUX_SESSION="${TMUX_SESSION:-showo2_c1_conflict}"

if tmux has-session -t "${TMUX_SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${TMUX_SESSION}" >&2
  exit 1
fi

mkdir -p "${RUN_DIR}/logs"
printf '%s\n' "${RUN_DIR}" > "${OUTPUT_ROOT}/latest_run.txt"

command="cd $(printf '%q' "${SHOWO2_ROOT}") && export RUN_ID=$(printf '%q' "${RUN_ID}") RUN_DIR=$(printf '%q' "${RUN_DIR}")"
for name in SHOWO2_ROOT SHOWO2_EVAL_PYTHON C0_RUN GENEVAL_RESULTS SHOWO2_CONFIG SHOWO2_MODEL SHOWO2_VAE_PATH PER_TAG LIMIT SEED DEVICE WEIGHT_TYPE VAE_WEIGHT_TYPE HF_HUB_OFFLINE TRANSFORMERS_OFFLINE; do
  if [[ -n "${!name:-}" ]]; then
    command+=" $(printf '%s=%q' "${name}" "${!name}")"
  fi
done
command+=" && ./evaluation/c1/run_probe.sh 2>&1 | tee -a $(printf '%q' "${RUN_DIR}/logs/queue.log")"

tmux new-session -d -s "${TMUX_SESSION}" "${command}"
echo "Started ${TMUX_SESSION}"
echo "Run directory: ${RUN_DIR}"
echo "Attach: tmux attach -t ${TMUX_SESSION}"
echo "Monitor: ${SHOWO2_ROOT}/evaluation/c1/monitor_probe.sh ${RUN_DIR} 5"
