#!/usr/bin/env bash
set -euo pipefail

C3_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SHOWO2_ROOT=$(cd -- "${C3_DIR}/../.." && pwd)
WORKSPACE_ROOT=$(cd -- "${SHOWO2_ROOT}/../.." && pwd)
source "${SHOWO2_ROOT}/evaluation/c0/common.sh"

: "${RUN_DIR:?Set RUN_DIR to a new holdout20 output directory}"
MANIFEST=${MANIFEST:-"${WORKSPACE_ROOT}/data/showo2_layout/position_holdout20_v1.jsonl"}

export HF_HOME=${HF_HOME:-"${WORKSPACE_ROOT}/cache/huggingface"}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

"${PYTHON}" "${C3_DIR}/build_position_holdout20.py" --output "${MANIFEST}"
"${PYTHON}" "${C3_DIR}/run_quick8_v2.py" \
    --run-dir "${RUN_DIR}" \
    --protocol-version position-holdout20-v1-relative-margin \
    --manifest "${MANIFEST}" \
    --config "${SHOWO2_CONFIG}" \
    --model "${SHOWO2_MODEL}" \
    --vae-path "${SHOWO2_VAE_PATH}" \
    --variants baseline,guided \
    --samples-per-prompt 4 \
    --sample-batch-size 4 \
    --base-seed 2026 \
    --guidance-scale 7.5 \
    --num-inference-steps 100 \
    --inside-weight 0.0 \
    --cross-weight 0.0 \
    --relation-weight 1.0 \
    --relation-margin 0.20 \
    --guided-guidance-steps 8 \
    --guided-inner-steps 3 \
    --guided-energy-step-size 15.0 \
    --guided-layers 7,14,21 \
    --geneval-python "${GENEVAL_PYTHON}" \
    --geneval-root "${GENEVAL_ROOT}" \
    --geneval-model-config "${GENEVAL_MODEL_CONFIG}" \
    --geneval-model-path "${GENEVAL_MODEL_PATH}" \
    --geneval-clip-path "${GENEVAL_CLIP_PATH}"
