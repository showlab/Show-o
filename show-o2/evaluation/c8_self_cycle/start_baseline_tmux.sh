#!/usr/bin/env bash
set -euo pipefail

C8_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SHOWO2_ROOT=$(cd -- "${C8_DIR}/../.." && pwd)
WORKSPACE_ROOT=$(cd -- "${SHOWO2_ROOT}/../.." && pwd)
SESSION_NAME=${SESSION_NAME:-showo2_c8_self_cycle_baseline}
RUN_ID=${RUN_ID:-"c8-self-cycle-baseline-$(date -u +%Y%m%dT%H%M%SZ)"}
RUN_DIR=${RUN_DIR:-"${WORKSPACE_ROOT}/runs/showo2_c8/${RUN_ID}"}
LOG_PATH=${RUN_DIR}/run.log
LIMIT_PROMPTS=${LIMIT_PROMPTS:-}
NUM_INFERENCE_STEPS=${NUM_INFERENCE_STEPS:-50}
GENERATION_BATCH_SIZE=${GENERATION_BATCH_SIZE:-4}
SAMPLES_PER_PROMPT=${SAMPLES_PER_PROMPT:-1}
GUIDANCE_SCALE=${GUIDANCE_SCALE:-5.0}

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
    echo "ERROR: tmux session already exists: ${SESSION_NAME}" >&2
    exit 2
fi

mkdir -p "${RUN_DIR}" "${WORKSPACE_ROOT}/runs/showo2_c8"
printf '%s\n' "${RUN_DIR}" > "${WORKSPACE_ROOT}/runs/showo2_c8/latest_baseline.txt"

tmux new-session -d -s "${SESSION_NAME}" \
    "cd ${SHOWO2_ROOT@Q} && export RUN_DIR=${RUN_DIR@Q} LIMIT_PROMPTS=${LIMIT_PROMPTS@Q} NUM_INFERENCE_STEPS=${NUM_INFERENCE_STEPS@Q} GENERATION_BATCH_SIZE=${GENERATION_BATCH_SIZE@Q} SAMPLES_PER_PROMPT=${SAMPLES_PER_PROMPT@Q} GUIDANCE_SCALE=${GUIDANCE_SCALE@Q}; set -o pipefail; ${C8_DIR@Q}/run_baseline.sh 2>&1 | tee -a ${LOG_PATH@Q}; run_code=\${PIPESTATUS[0]}; echo \"C8 self-cycle baseline exited with code \${run_code}\"; exec bash"

echo "SESSION_NAME=${SESSION_NAME}"
echo "RUN_DIR=${RUN_DIR}"
echo "LOG_PATH=${LOG_PATH}"
echo "MONITOR=${WORKSPACE_ROOT}/venvs/showo2-eval/bin/python ${C8_DIR}/monitor_self_cycle.py ${RUN_DIR} --watch"
