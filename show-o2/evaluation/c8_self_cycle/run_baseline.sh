#!/usr/bin/env bash
set -euo pipefail

C8_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SHOWO2_ROOT=$(cd -- "${C8_DIR}/../.." && pwd)
WORKSPACE_ROOT=$(cd -- "${SHOWO2_ROOT}/../.." && pwd)
PYTHON=${PYTHON:-${WORKSPACE_ROOT}/venvs/showo2-eval/bin/python}
RUN_DIR=${RUN_DIR:?RUN_DIR is required}
DATASET=${DATASET:-${WORKSPACE_ROOT}/data/showo2_self_cycle/unicycle/self_built.jsonl}
CONFIG=${CONFIG:-${SHOWO2_ROOT}/configs/showo2_1.5b_demo_432x432.yaml}
MODEL=${MODEL:-showlab/show-o2-1.5B}
VAE_PATH=${VAE_PATH:-${SHOWO2_ROOT}/Wan2.1_VAE.pth}
LIMIT_PROMPTS=${LIMIT_PROMPTS:-}

mkdir -p "${RUN_DIR}"
export HF_HOME=${HF_HOME:-${WORKSPACE_ROOT}/cache/huggingface}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

for required in "${PYTHON}" "${DATASET}" "${CONFIG}" "${VAE_PATH}"; do
    [[ -e "${required}" ]] || { echo "missing required resource: ${required}" >&2; exit 1; }
done

args=(
    --run-dir "${RUN_DIR}"
    --dataset "${DATASET}"
    --config "${CONFIG}"
    --model "${MODEL}"
    --variant-name baseline
    --vae-path "${VAE_PATH}"
    --guidance-scale "${GUIDANCE_SCALE:-5.0}"
    --num-inference-steps "${NUM_INFERENCE_STEPS:-50}"
    --generation-batch-size "${GENERATION_BATCH_SIZE:-4}"
    --samples-per-prompt "${SAMPLES_PER_PROMPT:-1}"
    --base-seed "${BASE_SEED:-20260809}"
    --max-open-answer-tokens "${MAX_OPEN_ANSWER_TOKENS:-12}"
)
if [[ -n "${LIMIT_PROMPTS}" ]]; then
    args+=(--limit-prompts "${LIMIT_PROMPTS}")
fi

"${PYTHON}" -u "${C8_DIR}/run_self_cycle.py" "${args[@]}"
touch "${RUN_DIR}/BENCHMARK_COMPLETED"
