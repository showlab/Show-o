#!/usr/bin/env bash
set -euo pipefail

C4_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SHOWO2_ROOT=$(cd -- "${C4_DIR}/../.." && pwd)
WORKSPACE_ROOT=$(cd -- "${SHOWO2_ROOT}/../.." && pwd)
SESSION_NAME=${SESSION_NAME:-showo2_drsa_data}
RUN_ID=${RUN_ID:-"drsa-data-$(date -u +%Y%m%dT%H%M%SZ)"}
RUN_DIR=${RUN_DIR:-"${WORKSPACE_ROOT}/runs/showo2_drsa_data/${RUN_ID}"}
DATA_ROOT=${DATA_ROOT:-"${WORKSPACE_ROOT}/data/drsa"}
SCOPE=${SCOPE:-train}
WORKERS=${WORKERS:-2}
HF_ENDPOINT=${DRSA_HF_ENDPOINT:-https://hf-mirror.com}
WAIT_FOR=${WAIT_FOR:-}
WAIT_FOR_STATUS=${WAIT_FOR_STATUS:-}

if tmux list-sessions -F '#S' 2>/dev/null | grep -Fxq -- "${SESSION_NAME}"; then
    echo "ERROR: tmux session already exists: ${SESSION_NAME}" >&2
    exit 2
fi

mkdir -p "${RUN_DIR}" "${DATA_ROOT}" "${WORKSPACE_ROOT}/runs/showo2_drsa_data"
printf '%s\n' "${RUN_DIR}" > "${WORKSPACE_ROOT}/runs/showo2_drsa_data/latest.txt"

tmux new-session -d -s "${SESSION_NAME}" \
    "cd ${SHOWO2_ROOT@Q} && export HF_ENDPOINT=${HF_ENDPOINT@Q}; set -o pipefail; if [[ -n ${WAIT_FOR@Q} ]]; then while [[ ! -f ${WAIT_FOR@Q} ]]; do echo \"Waiting for prerequisite file: ${WAIT_FOR@Q}\"; sleep 30; done; fi; if [[ -n ${WAIT_FOR_STATUS@Q} ]]; then ${WORKSPACE_ROOT@Q}/venvs/showo2-eval/bin/python ${C4_DIR@Q}/wait_for_download_terminal.py ${WAIT_FOR_STATUS@Q}; fi; ${WORKSPACE_ROOT@Q}/venvs/showo2-eval/bin/python ${C4_DIR@Q}/download_drsa_data.py --data-root ${DATA_ROOT@Q} --scope ${SCOPE@Q} --workers ${WORKERS@Q} --manifest-path ${RUN_DIR@Q}/manifest.json --status-path ${RUN_DIR@Q}/status.json 2>&1 | tee -a ${RUN_DIR@Q}/run.log; download_code=\${PIPESTATUS[0]}; if [[ \${download_code} -eq 0 ]]; then ${WORKSPACE_ROOT@Q}/venvs/showo2-eval/bin/python ${C4_DIR@Q}/smoke_drsa_data.py --data-root ${DATA_ROOT@Q} --spright-samples 32 --vsr-samples 128 --require-vsr-images --output ${RUN_DIR@Q}/smoke.json 2>&1 | tee -a ${RUN_DIR@Q}/run.log; smoke_code=\${PIPESTATUS[0]}; else smoke_code=99; fi; echo \"DRSA data preparation exited: download=\${download_code} smoke=\${smoke_code}\"; exec bash"

echo "SESSION_NAME=${SESSION_NAME}"
echo "RUN_DIR=${RUN_DIR}"
echo "DATA_ROOT=${DATA_ROOT}"
echo "SCOPE=${SCOPE}"
echo "WAIT_FOR=${WAIT_FOR:-none}"
echo "WAIT_FOR_STATUS=${WAIT_FOR_STATUS:-none}"
echo "MONITOR=${C4_DIR}/monitor_drsa_data.sh"
