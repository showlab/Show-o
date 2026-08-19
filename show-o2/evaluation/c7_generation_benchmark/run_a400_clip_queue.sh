#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/Show-o/show-o2
WORKSPACE=/root/autodl-tmp
PYTHON=${PYTHON:-${WORKSPACE}/venvs/showo2-eval/bin/python}
CLIP_PYTHON=${CLIP_PYTHON:-${WORKSPACE}/venvs/geneval-score/bin/python}
RUN_ROOT=${RUN_ROOT:?RUN_ROOT is required}
A400_CONFIG=${A400_CONFIG:-${WORKSPACE}/runs/showo2_stage_a_gate/stage-a-gate-20260807T142951Z/evaluation/variants.json}
MANIFEST=${MANIFEST:-${WORKSPACE}/data/showo2_layout/position_unseen72_v1.jsonl}
CLIP_CHECKPOINT=${CLIP_CHECKPOINT:-${WORKSPACE}/models/geneval/clip/ViT-L-14.state_dict.pt}
GEN_RUN=${RUN_ROOT}/a_400_generation

mkdir -p "${RUN_ROOT}/logs" "${GEN_RUN}"
export HF_HOME=${HF_HOME:-${WORKSPACE}/cache/huggingface}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

for required in "${A400_CONFIG}" "${MANIFEST}" "${CLIP_CHECKPOINT}" "${ROOT}/Wan2.1_VAE.pth"; do
    [[ -f "${required}" ]] || { echo "missing required file: ${required}" >&2; exit 1; }
done
for variant in b1_200 masklogits_0500; do
    [[ -f "${RUN_ROOT}/${variant}/summary.json" ]] || {
        echo "missing precomputed CLIP summary: ${RUN_ROOT}/${variant}/summary.json" >&2
        exit 1
    }
done

on_exit() {
    code=$?
    if [[ ${code} -ne 0 ]]; then
        printf 'phase=failed status=failed exit_code=%s\n' "${code}" >"${RUN_ROOT}/queue.status"
    fi
}
trap on_exit EXIT

printf 'phase=a400_generation status=running\n' >"${RUN_ROOT}/queue.status"
"${PYTHON}" -u "${ROOT}/evaluation/c3_layout_guidance/run_quick8_v2.py" \
    --run-dir "${GEN_RUN}" \
    --protocol-version "c7-a400-full-position72-v1" \
    --manifest "${MANIFEST}" \
    --config "${ROOT}/configs/showo2_1.5b_demo_432x432.yaml" \
    --model showlab/show-o2-1.5B \
    --vae-path "${ROOT}/Wan2.1_VAE.pth" \
    --variant-config-json "${A400_CONFIG}" \
    --variants native_400 \
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
    2>&1 | tee "${RUN_ROOT}/logs/a400_generation.log"

printf 'phase=a400_clip status=running\n' >"${RUN_ROOT}/queue.status"
"${CLIP_PYTHON}" "${ROOT}/evaluation/c7_generation_benchmark/score_clip.py" \
    --image-root "${GEN_RUN}/native_400/geneval" \
    --output-dir "${RUN_ROOT}/a_400" \
    --variant a_400 \
    --clip-checkpoint "${CLIP_CHECKPOINT}" \
    --batch-size 32 \
    2>&1 | tee "${RUN_ROOT}/logs/a400_clip.log"

printf 'phase=paired_comparison status=running\n' >"${RUN_ROOT}/queue.status"
"${CLIP_PYTHON}" "${ROOT}/evaluation/c7_generation_benchmark/compare_clip_variants.py" \
    --run-root "${RUN_ROOT}" \
    --variants a_400,b1_200,masklogits_0500 \
    --output "${RUN_ROOT}/comparison_summary.json" \
    2>&1 | tee "${RUN_ROOT}/logs/comparison.log"

printf 'phase=completed status=completed\n' >"${RUN_ROOT}/queue.status"
touch "${RUN_ROOT}/BENCHMARK_COMPLETED"
trap - EXIT
echo "C7 A-400/B1-200/mask-logits-500 CLIP benchmark completed: ${RUN_ROOT}"
