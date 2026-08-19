#!/usr/bin/env bash
set -euo pipefail

C6_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd -- "${C6_DIR}/../.." && pwd)
WORKSPACE=$(cd -- "${ROOT}/../.." && pwd)
SESSION_NAME=${SESSION_NAME:-showo2_flow_retention_eval}
RUN_ID=${RUN_ID:-"mask-flow-retention-eval-$(date -u +%Y%m%dT%H%M%SZ)"}
RUN_ROOT=${RUN_ROOT:-${WORKSPACE}/runs/showo2_mask2former_benchmark/${RUN_ID}}
LOG_PATH=${RUN_ROOT}/queue.log

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
    echo "ERROR: tmux session already exists: ${SESSION_NAME}" >&2
    exit 2
fi

mkdir -p "${RUN_ROOT}" "${WORKSPACE}/runs/showo2_mask2former_benchmark"
printf '%s\n' "${RUN_ROOT}" > "${WORKSPACE}/runs/showo2_mask2former_benchmark/latest_flow_retention_eval.txt"

tmux new-session -d -s "${SESSION_NAME}" \
    "cd ${ROOT@Q} && export RUN_ROOT=${RUN_ROOT@Q}; set -o pipefail; ${C6_DIR@Q}/run_flow_retention_eval_queue.sh 2>&1 | tee -a ${LOG_PATH@Q}; run_code=\${PIPESTATUS[0]}; echo \"Flow-retention evaluation queue exited with code \${run_code}\"; exec bash"

echo "SESSION_NAME=${SESSION_NAME}"
echo "RUN_ROOT=${RUN_ROOT}"
echo "LOG_PATH=${LOG_PATH}"
echo "MONITOR=${WORKSPACE}/venvs/showo2-eval/bin/python ${C6_DIR}/monitor_flow_retention_eval.py ${RUN_ROOT} --session ${SESSION_NAME} --watch"
