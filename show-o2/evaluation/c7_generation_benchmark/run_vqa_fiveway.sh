#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/Show-o/show-o2
WORKSPACE=/root/autodl-tmp
VQA_PYTHON=${VQA_PYTHON:-${WORKSPACE}/venvs/showo2-score-vqa/bin/python}
MODEL_PATH=${MODEL_PATH:-${WORKSPACE}/models/c7/Qwen2.5-VL-7B-Instruct}
RUN_ROOT=${RUN_ROOT:?RUN_ROOT is required}
OUTPUT_DIR=${RUN_ROOT}/vqascore

BASE_ROOT=${WORKSPACE}/runs/showo2_mask2former_rollback/masklogits-step500-full-20260809T090000Z/full_position/baseline/geneval
A400_ROOT=${WORKSPACE}/runs/showo2_c7/c7-clip-a400-b1-mask500-20260809T062221Z/a_400_generation/native_400/geneval
B1_ROOT=${WORKSPACE}/runs/showo2_mask2former_rollback/masklogits-step500-full-20260809T090000Z/full_position/b1_200/geneval
MASK500_ROOT=${WORKSPACE}/runs/showo2_mask2former_rollback/masklogits-step500-full-20260809T090000Z/full_position/masklogits_0500/geneval
TEACHER500_ROOT=${WORKSPACE}/runs/showo2_mask2former_teacher_validity/teacher-validity-only-b1-full-20260809T000700Z/full_position/c6_0500/geneval

mkdir -p "${RUN_ROOT}/logs" "${OUTPUT_DIR}"
export PATH="${WORKSPACE}/venvs/showo2-score-vqa/bin:${PATH}"
export HF_HOME=${HF_HOME:-${WORKSPACE}/cache/huggingface}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

for required in "${VQA_PYTHON}" "${MODEL_PATH}" "${BASE_ROOT}" "${A400_ROOT}" \
    "${B1_ROOT}" "${MASK500_ROOT}" "${TEACHER500_ROOT}"; do
    [[ -e "${required}" ]] || { echo "missing required resource: ${required}" >&2; exit 1; }
done

on_exit() {
    code=$?
    if [[ ${code} -ne 0 ]]; then
        printf 'phase=failed status=failed exit_code=%s\n' "${code}" >"${RUN_ROOT}/queue.status"
    fi
}
trap on_exit EXIT

printf 'phase=vqascore status=running\n' >"${RUN_ROOT}/queue.status"
"${VQA_PYTHON}" -u "${ROOT}/evaluation/c7_generation_benchmark/score_vqa.py" \
    --variant "original=${BASE_ROOT}" \
    --variant "a_400=${A400_ROOT}" \
    --variant "b1_200=${B1_ROOT}" \
    --variant "mask_logits_500=${MASK500_ROOT}" \
    --variant "teacher_valid_500=${TEACHER500_ROOT}" \
    --output-dir "${OUTPUT_DIR}" \
    --model-path "${MODEL_PATH}" \
    --batch-size "${VQA_BATCH_SIZE:-8}" \
    --attn-implementation sdpa \
    2>&1 | tee "${RUN_ROOT}/logs/vqascore.log"

printf 'phase=completed status=completed\n' >"${RUN_ROOT}/queue.status"
touch "${RUN_ROOT}/BENCHMARK_COMPLETED"
trap - EXIT
echo "C7 five-way VQAScore completed: ${RUN_ROOT}"
