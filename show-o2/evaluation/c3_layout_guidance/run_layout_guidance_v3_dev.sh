#!/usr/bin/env bash
set -euo pipefail

C3_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SHOWO2_ROOT=$(cd -- "${C3_DIR}/../.." && pwd)
WORKSPACE_ROOT=$(cd -- "${SHOWO2_ROOT}/../.." && pwd)
source "${SHOWO2_ROOT}/evaluation/c0/common.sh"

: "${RUN_DIR:?Set RUN_DIR to a new layout-guidance-v3 output directory}"
SELECTION_MANIFEST=${SELECTION_MANIFEST:-"${WORKSPACE_ROOT}/data/showo2_layout/position_quick8_v2.jsonl"}
MANIFEST=${MANIFEST:-"${WORKSPACE_ROOT}/data/showo2_layout/position_quick8_c0_v2.jsonl"}
VARIANT_CONFIG=${VARIANT_CONFIG:-"${C3_DIR}/layout_guidance_v3_variants.json"}

export HF_HOME=${HF_HOME:-"${WORKSPACE_ROOT}/cache/huggingface"}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

"${PYTHON}" "${C3_DIR}/build_position_c0_parity.py" \
    --selection-manifest "${SELECTION_MANIFEST}" \
    --output "${MANIFEST}"

"${PYTHON}" "${C3_DIR}/run_quick8_v2.py" \
    --run-dir "${RUN_DIR}" \
    --protocol-version layout-guidance-v3-c0-parity-dev \
    --manifest "${MANIFEST}" \
    --variant-config-json "${VARIANT_CONFIG}" \
    --config "${SHOWO2_CONFIG}" \
    --model "${SHOWO2_MODEL}" \
    --vae-path "${SHOWO2_VAE_PATH}" \
    --weight-type bfloat16 \
    --vae-weight-type bfloat16 \
    --variants baseline,relation_qwen21_v3 \
    --samples-per-prompt 4 \
    --sample-batch-size 4 \
    --base-seed 2026 \
    --guidance-scale 7.5 \
    --num-inference-steps 100 \
    --sampling-attention-mask flex \
    --inside-weight 0.0 \
    --cross-weight 0.0 \
    --relation-weight 1.0 \
    --relation-margin 0.03 \
    --relation-dominance-weight 0.5 \
    --relation-dominance-ratio 0.5 \
    --entropy-weight 0.1 \
    --entropy-target 0.95 \
    --presence-weight 0.2 \
    --presence-min 0.0001 \
    --boxdiff-inner-weight 0.0 \
    --boxdiff-outer-weight 0.0 \
    --boxdiff-corner-weight 0.0 \
    --boxdiff-token-mode per_token \
    --attention-normalization text_conditional \
    --text-gate-power 0.0 \
    --geneval-python "${GENEVAL_PYTHON}" \
    --geneval-root "${GENEVAL_ROOT}" \
    --geneval-model-config "${GENEVAL_MODEL_CONFIG}" \
    --geneval-model-path "${GENEVAL_MODEL_PATH}" \
    --geneval-clip-path "${GENEVAL_CLIP_PATH}"

"${PYTHON}" "${C3_DIR}/summarize_boxdiff_seed_ab.py" \
    --run-dir "${RUN_DIR}"
