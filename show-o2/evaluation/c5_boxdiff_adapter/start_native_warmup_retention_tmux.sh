#!/usr/bin/env bash
set -euo pipefail

C5_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SHOWO2_ROOT=$(cd -- "${C5_DIR}/../.." && pwd)
WORKSPACE_ROOT=$(cd -- "${SHOWO2_ROOT}/../.." && pwd)
SESSION_NAME=${SESSION_NAME:-showo2_native_warmup_retention}
TRAIN_RUN_DIR=${TRAIN_RUN_DIR:?Set TRAIN_RUN_DIR to the completed 500-step run}
EVAL_DIR=${EVAL_DIR:-"${TRAIN_RUN_DIR}/evaluation"}
VSR_LIMIT=${VSR_LIMIT:-256}
POSITION_LIMIT_PROMPTS=${POSITION_LIMIT_PROMPTS:-8}
POSITION_STEPS=${POSITION_STEPS:-12}

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
    echo "ERROR: tmux session already exists: ${SESSION_NAME}" >&2
    exit 2
fi

mkdir -p "${EVAL_DIR}"
tmux new-session -d -s "${SESSION_NAME}" \
    "cd ${SHOWO2_ROOT@Q} && export TRAIN_RUN_DIR=${TRAIN_RUN_DIR@Q} EVAL_DIR=${EVAL_DIR@Q} VSR_LIMIT=${VSR_LIMIT@Q} POSITION_LIMIT_PROMPTS=${POSITION_LIMIT_PROMPTS@Q} POSITION_STEPS=${POSITION_STEPS@Q} HF_HOME=${WORKSPACE_ROOT@Q}/cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} && set -o pipefail; ./evaluation/c5_boxdiff_adapter/evaluate_native_warmup_retention.sh 2>&1 | tee -a ${EVAL_DIR@Q}/retention_queue.log; code=\${PIPESTATUS[0]}; echo \"Native retention evaluation exited with code \${code}\"; exec bash"

echo "SESSION_NAME=${SESSION_NAME}"
echo "TRAIN_RUN_DIR=${TRAIN_RUN_DIR}"
echo "EVAL_DIR=${EVAL_DIR}"
echo "STATUS_VSR=${EVAL_DIR}/vsr/status.json"
echo "STATUS_POSITION=${EVAL_DIR}/position_paired8_v2/status.json"
echo "LOG=${EVAL_DIR}/retention_queue.log"
