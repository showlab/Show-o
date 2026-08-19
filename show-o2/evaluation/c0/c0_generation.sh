#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

ACTION=${1:-all}
RUN_DIR=${RUN_DIR:?RUN_DIR must point to the active C0 run}
GENEVAL_SAMPLES="${RUN_DIR}/generation/geneval/samples"
GENEVAL_RESULTS="${RUN_DIR}/generation/geneval/results"
DPG_SAMPLES="${RUN_DIR}/generation/dpg/samples"
DPG_RESULTS="${RUN_DIR}/generation/dpg/results"
mkdir -p "${GENEVAL_SAMPLES}" "${GENEVAL_RESULTS}" "${DPG_SAMPLES}" "${DPG_RESULTS}" "${RUN_DIR}/logs"

mapfile -t GPU_IDS < <(visible_gpu_ids)
(( ${#GPU_IDS[@]} > 0 )) || die "No visible GPU was detected"
NUM_PROCESSES=${NUM_PROCESSES:-${#GPU_IDS[@]}}
(( NUM_PROCESSES >= 1 && NUM_PROCESSES <= ${#GPU_IDS[@]} )) || die "NUM_PROCESSES=${NUM_PROCESSES} exceeds visible GPUs (${#GPU_IDS[@]})"

run_sharded_generation() {
    local script=$1
    local outdir=$2
    local prompts=$3
    local batch_size=$4
    local guidance=$5
    local steps=$6
    local name=$7
    local pids=()
    local rank

    for ((rank = 0; rank < NUM_PROCESSES; rank++)); do
        CUDA_VISIBLE_DEVICES=${GPU_IDS[$rank]} \
            PYTHONPATH="${SHOWO2_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
            "${PYTHON}" "${script}" \
            config="${SHOWO2_CONFIG}" \
            model.showo.pretrained_model_path="${SHOWO2_MODEL}" \
            model.vae_model.pretrained_model_path="${SHOWO2_VAE_PATH}" \
            outdir="${outdir}" \
            validation_prompts_file="${prompts}" \
            batch_size="${batch_size}" \
            guidance_scale="${guidance}" \
            num_inference_steps="${steps}" \
            device_id="${rank}" \
            num_devices="${NUM_PROCESSES}" \
            >"${RUN_DIR}/logs/${name}-rank${rank}.log" 2>&1 &
        pids+=("$!")
    done

    local failed=0
    for rank in "${!pids[@]}"; do
        if ! wait "${pids[$rank]}"; then
            echo "${name} rank ${rank} failed; see ${RUN_DIR}/logs/${name}-rank${rank}.log" >&2
            failed=1
        fi
    done
    (( failed == 0 )) || return 1
}

generate_geneval() {
    require_dir "${GENEVAL_ROOT}"
    run_sharded_generation \
        "${SHOWO2_ROOT}/evaluation/inference_geneval.py" \
        "${GENEVAL_SAMPLES}" \
        "${SHOWO2_ROOT}/prompts/geneval_meta_data_rewritten.jsonl" \
        "${GENEVAL_BATCH_SIZE:-4}" \
        "${GENEVAL_GUIDANCE_SCALE:-7.5}" \
        "${GENEVAL_STEPS:-100}" \
        geneval
}

generate_dpg() {
    local relative_outdir
    relative_outdir=$(realpath --relative-to="${SHOWO2_ROOT}" "${DPG_SAMPLES}")
    run_sharded_generation \
        "${SHOWO2_ROOT}/evaluation/inference_dpg.py" \
        "${relative_outdir}" \
        "${SHOWO2_ROOT}/prompts/dpg_bench_meta_data.json" \
        "${DPG_BATCH_SIZE:-4}" \
        "${DPG_GUIDANCE_SCALE:-10}" \
        "${DPG_STEPS:-50}" \
        dpg
}

score_geneval() {
    require_executable "${GENEVAL_PYTHON}"
    require_dir "${GENEVAL_MODEL_PATH}"
    require_file "${GENEVAL_MODEL_CONFIG}"
    require_file "${GENEVAL_CLIP_PATH}"
    require_file "${GENEVAL_MODEL_PATH}/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth"
    "${GENEVAL_PYTHON}" -c 'import clip_benchmark, mmdet, open_clip' >/dev/null 2>&1 || \
        die "GENEVAL_PYTHON lacks mmdet/open_clip/clip_benchmark; prepare the official GenEval scoring environment"
    "${GENEVAL_PYTHON}" "${GENEVAL_ROOT}/evaluation/evaluate_images.py" \
        "${GENEVAL_SAMPLES}" \
        --outfile "${GENEVAL_RESULTS}/results.jsonl" \
        --model-path "${GENEVAL_MODEL_PATH}" \
        --model-config "${GENEVAL_MODEL_CONFIG}" \
        --clip-path "${GENEVAL_CLIP_PATH}" \
        >"${RUN_DIR}/logs/geneval-score.log" 2>&1
    "${GENEVAL_PYTHON}" "${GENEVAL_ROOT}/evaluation/summary_scores.py" \
        "${GENEVAL_RESULTS}/results.jsonl" \
        | tee "${GENEVAL_RESULTS}/summary.txt"
}

score_dpg() {
    require_executable "${DPG_PYTHON}"
    require_dir "${DPG_MPLUG_PATH}"
    "${DPG_PYTHON}" -c 'import modelscope' >/dev/null 2>&1 || \
        die "DPG_PYTHON lacks ModelScope/mPLUG dependencies; run c0_prepare_dpg.sh"
    local processes=${DPG_SCORE_PROCESSES:-${NUM_PROCESSES}}
    (( processes >= 1 && processes <= ${#GPU_IDS[@]} )) || \
        die "DPG_SCORE_PROCESSES=${processes} exceeds visible GPUs (${#GPU_IDS[@]})"
    local score_args=(
        "${SHOWO2_ROOT}/evaluation/compute_dpg_bench.py"
        --image-root-path "${DPG_SAMPLES}"
        --resolution "${DPG_RESOLUTION:-432}"
        --pic-num "${DPG_BATCH_SIZE:-4}"
        --csv "${SHOWO2_ROOT}/evaluation/dpg_bench.csv"
        --res-path "${DPG_RESULTS}/scores.txt"
        --vqa-model mplug
        --vqa-model-path "${DPG_MPLUG_PATH}"
    )

    if (( processes == 1 )); then
        CUDA_VISIBLE_DEVICES=${GPU_IDS[0]} "${DPG_PYTHON}" "${score_args[@]}" \
            >"${RUN_DIR}/logs/dpg-score.log" 2>&1
    else
        require_executable "${DPG_ACCELERATE}"
        "${DPG_ACCELERATE}" launch \
            --num_machines 1 \
            --num_processes "${processes}" \
            --multi_gpu \
            --mixed_precision fp16 \
            --main_process_port "${DPG_PORT:-29500}" \
            "${score_args[@]}" \
            >"${RUN_DIR}/logs/dpg-score.log" 2>&1
    fi
}

case "${ACTION}" in
    geneval) generate_geneval ;;
    dpg) generate_dpg ;;
    samples) generate_geneval; generate_dpg ;;
    score-geneval) score_geneval ;;
    score-dpg) score_dpg ;;
    score) score_geneval; score_dpg ;;
    all) generate_geneval; generate_dpg; score_geneval; score_dpg ;;
    *) die "Usage: $0 [geneval|dpg|samples|score-geneval|score-dpg|score|all]" ;;
esac

