#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/Show-o/show-o2
WORKSPACE=/root/autodl-tmp
CLIP_PYTHON=${CLIP_PYTHON:-${WORKSPACE}/venvs/geneval-score/bin/python}
REWARD_PYTHON=${REWARD_PYTHON:-${WORKSPACE}/venvs/showo2-score-reward/bin/python}
RUN_ROOT=${RUN_ROOT:?RUN_ROOT is required}
CLIP_CHECKPOINT=${CLIP_CHECKPOINT:-${WORKSPACE}/models/geneval/clip/ViT-L-14.state_dict.pt}
REWARD_CHECKPOINT=${REWARD_CHECKPOINT:-${WORKSPACE}/models/c7/ImageReward-v1.0/ImageReward.pt}
REWARD_MED_CONFIG=${REWARD_MED_CONFIG:-${WORKSPACE}/models/c7/ImageReward-v1.0/med_config.json}
REWARD_BERT=${REWARD_BERT:-${WORKSPACE}/models/c7/bert-base-uncased-tokenizer}
CLIP_RUN=${RUN_ROOT}/clip
REWARD_RUN=${RUN_ROOT}/image_reward

CORRECTED_ROOT=${WORKSPACE}/runs/showo2_mask2former_corrected/corrected-b1-full-20260808T134500Z/full_position
BASE_ROOT=${CORRECTED_ROOT}/baseline/geneval
A400_ROOT=${WORKSPACE}/runs/showo2_c7/c7-clip-a400-b1-mask500-20260809T062221Z/a_400_generation/native_400/geneval
B1_ROOT=${CORRECTED_ROOT}/b1_200/geneval
ORIGINAL500_ROOT=${WORKSPACE}/runs/showo2_mask2former_rollback/masklogits-step500-full-20260809T090000Z/full_position/masklogits_0500/geneval
CORRECTED500_ROOT=${CORRECTED_ROOT}/c6_0500/geneval
CORRECTED1000_ROOT=${CORRECTED_ROOT}/c6_1000/geneval
CORRECTED2500_ROOT=${CORRECTED_ROOT}/c6_2500/geneval
CORRECTED5000_ROOT=${CORRECTED_ROOT}/c6_5000/geneval

mkdir -p "${RUN_ROOT}/logs" "${CLIP_RUN}" "${REWARD_RUN}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

for required in "${CLIP_CHECKPOINT}" "${REWARD_CHECKPOINT}" "${REWARD_MED_CONFIG}" "${REWARD_BERT}" \
    "${BASE_ROOT}" "${A400_ROOT}" "${B1_ROOT}" "${ORIGINAL500_ROOT}" \
    "${CORRECTED500_ROOT}" "${CORRECTED1000_ROOT}" "${CORRECTED2500_ROOT}" "${CORRECTED5000_ROOT}"; do
    [[ -e "${required}" ]] || { echo "missing required resource: ${required}" >&2; exit 1; }
done

on_exit() {
    code=$?
    if [[ ${code} -ne 0 ]]; then
        printf 'phase=failed status=failed exit_code=%s\n' "${code}" >"${RUN_ROOT}/queue.status"
    fi
}
trap on_exit EXIT

score_clip() {
    local name=$1
    local image_root=$2
    "${CLIP_PYTHON}" "${ROOT}/evaluation/c7_generation_benchmark/score_clip.py" \
        --image-root "${image_root}" \
        --output-dir "${CLIP_RUN}/${name}" \
        --variant "${name}" \
        --clip-checkpoint "${CLIP_CHECKPOINT}" \
        --batch-size 32
}

printf 'phase=clip status=running\n' >"${RUN_ROOT}/queue.status"
score_clip baseline "${BASE_ROOT}"
score_clip a_400 "${A400_ROOT}"
score_clip b1_200 "${B1_ROOT}"
score_clip original_masklogits_0500 "${ORIGINAL500_ROOT}"
score_clip corrected_0500 "${CORRECTED500_ROOT}"
score_clip corrected_1000 "${CORRECTED1000_ROOT}"
score_clip corrected_2500 "${CORRECTED2500_ROOT}"
score_clip corrected_5000 "${CORRECTED5000_ROOT}"

"${CLIP_PYTHON}" "${ROOT}/evaluation/c7_generation_benchmark/compare_clip_variants.py" \
    --run-root "${CLIP_RUN}" \
    --variants baseline,a_400,b1_200,original_masklogits_0500,corrected_0500,corrected_1000,corrected_2500,corrected_5000 \
    --output "${CLIP_RUN}/comparison_summary.json" \
    2>&1 | tee "${RUN_ROOT}/logs/clip_comparison.log"

printf 'phase=image_reward status=running\n' >"${RUN_ROOT}/queue.status"
PYTHONPATH= "${REWARD_PYTHON}" "${ROOT}/evaluation/c7_generation_benchmark/score_image_reward.py" \
    --variant "baseline=${BASE_ROOT}" \
    --variant "a_400=${A400_ROOT}" \
    --variant "b1_200=${B1_ROOT}" \
    --variant "original_masklogits_0500=${ORIGINAL500_ROOT}" \
    --variant "corrected_0500=${CORRECTED500_ROOT}" \
    --variant "corrected_1000=${CORRECTED1000_ROOT}" \
    --variant "corrected_2500=${CORRECTED2500_ROOT}" \
    --variant "corrected_5000=${CORRECTED5000_ROOT}" \
    --output-dir "${REWARD_RUN}" \
    --checkpoint "${REWARD_CHECKPOINT}" \
    --med-config "${REWARD_MED_CONFIG}" \
    --bert-tokenizer "${REWARD_BERT}" \
    --batch-size 16 \
    2>&1 | tee "${RUN_ROOT}/logs/image_reward.log"

printf 'phase=completed status=completed\n' >"${RUN_ROOT}/queue.status"
touch "${RUN_ROOT}/BENCHMARK_COMPLETED"
trap - EXIT
echo "C7 corrected CLIP/ImageReward benchmark completed: ${RUN_ROOT}"
