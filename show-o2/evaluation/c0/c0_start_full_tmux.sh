#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"
SESSION=${SESSION:-showo2_c0_full}
if tmux has-session -t "${SESSION}" 2>/dev/null; then
    die "tmux session already exists: ${SESSION}"
fi
RUN_ID=${RUN_ID:-c0-full-$(timestamp_utc)}
RUN_DIR=${RUN_DIR:-"${OUTPUT_ROOT}/${RUN_ID}"}
mkdir -p "${RUN_DIR}/logs"
printf '%s\n' "${RUN_DIR}" >"${OUTPUT_ROOT}/latest_full_run"

command=$(printf \
    'cd %q && export RUN_ID=%q RUN_DIR=%q HF_HOME=%q HF_DATASETS_CACHE=%q RESUME_COMPLETED=1 && export HF_ENDPOINT=%q HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_OFFLINE=%q TRANSFORMERS_OFFLINE=%q && ./evaluation/c0/c0_full_queue.sh 2>&1 | tee -a %q' \
    "${SHOWO2_ROOT}" "${RUN_ID}" "${RUN_DIR}" "${HF_HOME}" "${HF_DATASETS_CACHE}" \
    "${FULL_HF_ENDPOINT:-https://hf-mirror.com}" "${FULL_HF_HUB_OFFLINE:-0}" \
    "${FULL_TRANSFORMERS_OFFLINE:-1}" "${RUN_DIR}/logs/queue.log"
)
tmux new-session -d -s "${SESSION}" "${command}"
sleep 2
if ! tmux has-session -t "${SESSION}" 2>/dev/null; then
    die "tmux session exited during startup; inspect ${RUN_DIR}/logs/queue.log"
fi
echo "SESSION=${SESSION}"
echo "RUN_DIR=${RUN_DIR}"
echo "Attach:  tmux attach -t ${SESSION}"
echo "Pane:    tmux capture-pane -pt ${SESSION} -S -100"
echo "Monitor: ${C0_DIR}/c0_monitor.sh ${RUN_DIR} 5"
