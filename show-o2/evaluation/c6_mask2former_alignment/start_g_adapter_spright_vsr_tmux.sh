#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/Show-o/show-o2
: "${RUN_DIR:?RUN_DIR is required}"
SESSION_NAME=${SESSION_NAME:-showo2_g_adapter_spright_vsr}

if tmux has-session -t "=${SESSION_NAME}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION_NAME}" >&2
  exit 1
fi

mkdir -p "${RUN_DIR}"
tmux new-session -d -s "${SESSION_NAME}" \
  "cd '${ROOT}' && RUN_DIR='${RUN_DIR}' ./evaluation/c6_mask2former_alignment/run_g_adapter_spright_vsr.sh 2>&1 | tee -a '${RUN_DIR}/run.log'"

echo "session=${SESSION_NAME}"
echo "run=${RUN_DIR}"
echo "monitor=/root/autodl-tmp/venvs/geneval-score/bin/python ${ROOT}/evaluation/c6_mask2former_alignment/monitor_g_adapter_spright_vsr.py ${RUN_DIR} --session ${SESSION_NAME} --watch"
