#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/Show-o/show-o2
WORKSPACE=/root/autodl-tmp
SHOWO_PYTHON=${SHOWO_PYTHON:-${WORKSPACE}/venvs/showo2-eval/bin/python}
CLIP_PYTHON=${CLIP_PYTHON:-${WORKSPACE}/venvs/geneval-score/bin/python}
REWARD_PYTHON=${REWARD_PYTHON:-${WORKSPACE}/venvs/showo2-score-reward/bin/python}
RUN_ROOT=${RUN_ROOT:?RUN_ROOT is required}
MANIFEST=${MANIFEST:-${WORKSPACE}/data/showo2_layout/position_unseen72_v1.jsonl}
VARIANT_CONFIG=${VARIANT_CONFIG:-${WORKSPACE}/runs/showo2_mask2former_benchmark/c6-gate-20260808T112708Z/quick_variants.json}
CLIP_CHECKPOINT=${CLIP_CHECKPOINT:-${WORKSPACE}/models/geneval/clip/ViT-L-14.state_dict.pt}
REWARD_CHECKPOINT=${REWARD_CHECKPOINT:-${WORKSPACE}/models/c7/ImageReward-v1.0/ImageReward.pt}
REWARD_MED_CONFIG=${REWARD_MED_CONFIG:-${WORKSPACE}/models/c7/ImageReward-v1.0/med_config.json}
REWARD_BERT=${REWARD_BERT:-${WORKSPACE}/models/c7/bert-base-uncased-tokenizer}
GEN_RUN=${RUN_ROOT}/generation
CLIP_RUN=${RUN_ROOT}/clip
REWARD_RUN=${RUN_ROOT}/image_reward

BASE_ROOT=${WORKSPACE}/runs/showo2_mask2former_rollback/masklogits-step500-full-20260809T090000Z/full_position/baseline/geneval
A400_ROOT=${WORKSPACE}/runs/showo2_c7/c7-clip-a400-b1-mask500-20260809T062221Z/a_400_generation/native_400/geneval
B1_ROOT=${WORKSPACE}/runs/showo2_mask2former_rollback/masklogits-step500-full-20260809T090000Z/full_position/b1_200/geneval
MASK500_ROOT=${WORKSPACE}/runs/showo2_mask2former_rollback/masklogits-step500-full-20260809T090000Z/full_position/masklogits_0500/geneval
MASK1000_ROOT=${GEN_RUN}/c6_1000/geneval
MASK2500_ROOT=${GEN_RUN}/c6_2500/geneval

mkdir -p "${RUN_ROOT}/logs" "${GEN_RUN}" "${CLIP_RUN}" "${REWARD_RUN}"
export HF_HOME=${HF_HOME:-${WORKSPACE}/cache/huggingface}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

for required in "${MANIFEST}" "${VARIANT_CONFIG}" "${CLIP_CHECKPOINT}" \
    "${REWARD_CHECKPOINT}" "${REWARD_MED_CONFIG}" "${ROOT}/Wan2.1_VAE.pth"; do
    [[ -e "${required}" ]] || { echo "missing required resource: ${required}" >&2; exit 1; }
done
for image_root in "${BASE_ROOT}" "${A400_ROOT}" "${B1_ROOT}" "${MASK500_ROOT}"; do
    [[ -d "${image_root}" ]] || { echo "missing image root: ${image_root}" >&2; exit 1; }
done

on_exit() {
    code=$?
    if [[ ${code} -ne 0 ]]; then
        printf 'phase=failed status=failed exit_code=%s\n' "${code}" >"${RUN_ROOT}/queue.status"
    fi
}
trap on_exit EXIT

printf 'phase=generation status=running\n' >"${RUN_ROOT}/queue.status"
"${SHOWO_PYTHON}" -u "${ROOT}/evaluation/c3_layout_guidance/run_quick8_v2.py" \
    --run-dir "${GEN_RUN}" \
    --protocol-version "c7-original-masklogits-1000-2500-position72-v1" \
    --manifest "${MANIFEST}" \
    --config "${ROOT}/configs/showo2_1.5b_demo_432x432.yaml" \
    --model showlab/show-o2-1.5B \
    --vae-path "${ROOT}/Wan2.1_VAE.pth" \
    --variant-config-json "${VARIANT_CONFIG}" \
    --variants c6_1000,c6_2500 \
    --samples-per-prompt 4 \
    --sample-batch-size 4 \
    --base-seed 12026 \
    --guidance-scale 5.0 \
    --num-inference-steps 12 \
    --guided-guidance-steps 0 \
    --inside-weight 1.0 \
    --cross-weight 0.0 \
    --relation-weight 0.0 \
    --limit-prompts 72 \
    --skip-score \
    2>&1 | tee "${RUN_ROOT}/logs/generation.log"

printf 'phase=clip status=running\n' >"${RUN_ROOT}/queue.status"
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
score_clip baseline "${BASE_ROOT}"
score_clip a_400 "${A400_ROOT}"
score_clip b1_200 "${B1_ROOT}"
score_clip masklogits_0500 "${MASK500_ROOT}"
score_clip masklogits_1000 "${MASK1000_ROOT}"
score_clip masklogits_2500 "${MASK2500_ROOT}"

"${CLIP_PYTHON}" "${ROOT}/evaluation/c7_generation_benchmark/compare_clip_variants.py" \
    --run-root "${CLIP_RUN}" \
    --variants baseline,a_400,b1_200,masklogits_0500,masklogits_1000,masklogits_2500 \
    --output "${CLIP_RUN}/comparison_summary.json" \
    2>&1 | tee "${RUN_ROOT}/logs/clip_comparison.log"

printf 'phase=image_reward status=running\n' >"${RUN_ROOT}/queue.status"
PYTHONPATH= "${REWARD_PYTHON}" "${ROOT}/evaluation/c7_generation_benchmark/score_image_reward.py" \
    --variant "baseline=${BASE_ROOT}" \
    --variant "a_400=${A400_ROOT}" \
    --variant "b1_200=${B1_ROOT}" \
    --variant "masklogits_0500=${MASK500_ROOT}" \
    --variant "masklogits_1000=${MASK1000_ROOT}" \
    --variant "masklogits_2500=${MASK2500_ROOT}" \
    --output-dir "${REWARD_RUN}" \
    --checkpoint "${REWARD_CHECKPOINT}" \
    --med-config "${REWARD_MED_CONFIG}" \
    --bert-tokenizer "${REWARD_BERT}" \
    --batch-size 16 \
    2>&1 | tee "${RUN_ROOT}/logs/image_reward.log"

printf 'phase=completed status=completed\n' >"${RUN_ROOT}/queue.status"
touch "${RUN_ROOT}/BENCHMARK_COMPLETED"
trap - EXIT
echo "C7 mask-logits 1000/2500 benchmark completed: ${RUN_ROOT}"
