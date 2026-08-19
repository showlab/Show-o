#!/usr/bin/env bash
set -euo pipefail

C8_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SHOWO2_ROOT=$(cd -- "${C8_DIR}/../.." && pwd)
WORKSPACE_ROOT=$(cd -- "${SHOWO2_ROOT}/../.." && pwd)
PYTHON=${PYTHON:-${WORKSPACE_ROOT}/venvs/showo2-eval/bin/python}
SESSION_NAME=${SESSION_NAME:-showo2_c8_caption_cycle_baseline}
RUN_ID=${RUN_ID:-"c8-caption-cycle-baseline-$(date -u +%Y%m%dT%H%M%SZ)"}
RUN_DIR=${RUN_DIR:-${WORKSPACE_ROOT}/runs/showo2_c8/${RUN_ID}}
GENERATION_RUN=${GENERATION_RUN:-${WORKSPACE_ROOT}/runs/showo2_c8/c8-self-cycle-baseline-20260809T135133Z}
MAX_CAPTION_TOKENS=${MAX_CAPTION_TOKENS:-96}
LOG_PATH=${RUN_DIR}/run.log

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
    echo "ERROR: tmux session already exists: ${SESSION_NAME}" >&2
    exit 2
fi

mkdir -p "${RUN_DIR}" "${WORKSPACE_ROOT}/runs/showo2_c8"
printf '%s\n' "${RUN_DIR}" > "${WORKSPACE_ROOT}/runs/showo2_c8/latest_caption_cycle.txt"

tmux new-session -d -s "${SESSION_NAME}" \
    "cd ${SHOWO2_ROOT@Q} && export HF_HOME=${WORKSPACE_ROOT@Q}/cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 MKL_NUM_THREADS=4; set -o pipefail; ${PYTHON@Q} -u ${C8_DIR@Q}/run_caption_cycle.py --run-dir ${RUN_DIR@Q} --generation-run ${GENERATION_RUN@Q} --max-caption-tokens ${MAX_CAPTION_TOKENS@Q} 2>&1 | tee -a ${LOG_PATH@Q}; run_code=\${PIPESTATUS[0]}; echo \"C8 caption cycle exited with code \${run_code}\"; exec bash"

echo "SESSION_NAME=${SESSION_NAME}"
echo "RUN_DIR=${RUN_DIR}"
echo "LOG_PATH=${LOG_PATH}"
echo "MONITOR=${PYTHON} ${C8_DIR}/monitor_caption_cycle.py ${RUN_DIR} --watch"
