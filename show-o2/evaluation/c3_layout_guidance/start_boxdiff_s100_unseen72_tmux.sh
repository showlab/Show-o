#!/usr/bin/env bash
set -euo pipefail

C3_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SHOWO2_ROOT=$(cd -- "${C3_DIR}/../.." && pwd)
WORKSPACE_ROOT=$(cd -- "${SHOWO2_ROOT}/../.." && pwd)
SESSION_NAME=${SESSION_NAME:-showo2_boxdiff_s100_unseen72}
RUN_ID=${RUN_ID:-"boxdiff-s100-unseen72-$(date -u +%Y%m%dT%H%M%SZ)"}
RUN_DIR=${RUN_DIR:-"${WORKSPACE_ROOT}/runs/showo2_boxdiff_unseen72/${RUN_ID}"}
LOG_PATH="${RUN_DIR}/run.log"

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
    echo "ERROR: tmux session already exists: ${SESSION_NAME}" >&2
    exit 2
fi

mkdir -p "${RUN_DIR}" "${WORKSPACE_ROOT}/runs/showo2_boxdiff_unseen72"
printf '%s\n' "${RUN_DIR}" > "${WORKSPACE_ROOT}/runs/showo2_boxdiff_unseen72/latest.txt"

tmux new-session -d -s "${SESSION_NAME}" \
    "cd ${SHOWO2_ROOT@Q} && export RUN_DIR=${RUN_DIR@Q} && export HF_HOME=${WORKSPACE_ROOT@Q}/cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false && set -o pipefail; ./evaluation/c3_layout_guidance/run_boxdiff_s100_unseen72.sh 2>&1 | tee -a ${LOG_PATH@Q}; run_code=\${PIPESTATUS[0]}; echo \"BoxDiff s100 unseen72 exited with code \${run_code}\"; exec bash"

echo "SESSION_NAME=${SESSION_NAME}"
echo "RUN_DIR=${RUN_DIR}"
echo "LOG_PATH=${LOG_PATH}"
