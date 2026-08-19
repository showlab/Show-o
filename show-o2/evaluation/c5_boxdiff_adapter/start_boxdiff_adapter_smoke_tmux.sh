#!/usr/bin/env bash
set -euo pipefail

C5_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SHOWO2_ROOT=$(cd -- "${C5_DIR}/../.." && pwd)
WORKSPACE_ROOT=$(cd -- "${SHOWO2_ROOT}/../.." && pwd)
SESSION_NAME=${SESSION_NAME:-showo2_boxdiff_adapter_smoke}
RUN_ID=${RUN_ID:-"boxdiff-adapter-smoke-$(date -u +%Y%m%dT%H%M%SZ)"}
RUN_DIR=${RUN_DIR:-"${WORKSPACE_ROOT}/runs/showo2_boxdiff_adapter/${RUN_ID}"}

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
    echo "ERROR: tmux session already exists: ${SESSION_NAME}" >&2
    exit 2
fi

mkdir -p "${RUN_DIR}"
printf '%s\n' "${RUN_DIR}" > "${WORKSPACE_ROOT}/runs/showo2_boxdiff_adapter/latest.txt"

tmux new-session -d -s "${SESSION_NAME}" \
    "cd ${SHOWO2_ROOT@Q} && export HF_HOME=${WORKSPACE_ROOT@Q}/cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} && set -o pipefail; ${WORKSPACE_ROOT@Q}/venvs/showo2-eval/bin/python ${C5_DIR@Q}/boxdiff_adapter_smoke.py --run-dir ${RUN_DIR@Q} --manifest ${WORKSPACE_ROOT@Q}/data/showo2_layout/position_unseen72_v1.jsonl --max-steps ${MAX_STEPS:-4} 2>&1 | tee -a ${RUN_DIR@Q}/run.log; code=\${PIPESTATUS[0]}; echo \"BoxDiff adapter smoke exited with code \${code}\"; exec bash"

echo "SESSION_NAME=${SESSION_NAME}"
echo "RUN_ID=${RUN_ID}"
echo "RUN_DIR=${RUN_DIR}"
echo "STATUS=${RUN_DIR}/status.json"
echo "LOG=${RUN_DIR}/run.log"
