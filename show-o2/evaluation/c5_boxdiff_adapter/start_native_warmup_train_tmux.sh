#!/usr/bin/env bash
set -euo pipefail

C5_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SHOWO2_ROOT=$(cd -- "${C5_DIR}/../.." && pwd)
WORKSPACE_ROOT=$(cd -- "${SHOWO2_ROOT}/../.." && pwd)
SESSION_NAME=${SESSION_NAME:-showo2_native_warmup_train}
RUN_ID=${RUN_ID:-"native-warmup-500-$(date -u +%Y%m%dT%H%M%SZ)"}
RUN_DIR=${RUN_DIR:-"${WORKSPACE_ROOT}/runs/showo2_native_warmup/${RUN_ID}"}
DATA_MANIFEST=${DATA_MANIFEST:-"${WORKSPACE_ROOT}/runs/showo2_native_warmup/native_warmup_data_manifest.json"}
MAX_STEPS=${MAX_STEPS:-500}
SAVE_EVERY=${SAVE_EVERY:-100}
WARMUP_STEPS=${WARMUP_STEPS:-50}

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
    echo "ERROR: tmux session already exists: ${SESSION_NAME}" >&2
    exit 2
fi

mkdir -p "${RUN_DIR}" "${WORKSPACE_ROOT}/runs/showo2_native_warmup"
printf '%s\n' "${RUN_DIR}" > "${WORKSPACE_ROOT}/runs/showo2_native_warmup/latest.txt"

tmux new-session -d -s "${SESSION_NAME}" \
    "cd ${SHOWO2_ROOT@Q} && export HF_HOME=${WORKSPACE_ROOT@Q}/cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} && set -o pipefail; ${WORKSPACE_ROOT@Q}/venvs/showo2-eval/bin/python ${C5_DIR@Q}/native_objective_warmup_smoke.py --run-dir ${RUN_DIR@Q} --data-root ${WORKSPACE_ROOT@Q}/data/drsa --data-manifest ${DATA_MANIFEST@Q} --max-steps ${MAX_STEPS@Q} --save-every ${SAVE_EVERY@Q} --warmup-steps ${WARMUP_STEPS@Q} --workers 0 --t2i-loss-weight ${T2I_LOSS_WEIGHT:-1.0} --mmu-loss-weight ${MMU_LOSS_WEIGHT:-0.01} 2>&1 | tee -a ${RUN_DIR@Q}/run.log; code=\${PIPESTATUS[0]}; echo \"Native warm-up train exited with code \${code}\"; exec bash"

echo "SESSION_NAME=${SESSION_NAME}"
echo "RUN_ID=${RUN_ID}"
echo "RUN_DIR=${RUN_DIR}"
echo "MAX_STEPS=${MAX_STEPS}"
echo "SAVE_EVERY=${SAVE_EVERY}"
echo "WARMUP_STEPS=${WARMUP_STEPS}"
echo "STATUS=${RUN_DIR}/status.json"
echo "LOG=${RUN_DIR}/run.log"
