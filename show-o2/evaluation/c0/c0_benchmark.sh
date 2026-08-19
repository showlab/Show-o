#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

COMMAND=${1:-all}
DEFAULT_TASKS="mme,gqa,seedbench,mmbench_en_dev,mmmu_val,mmstar,ai2d"
TASKS=${TASKS:-${DEFAULT_TASKS}}

new_run_dir() {
    if [[ -z "${RUN_DIR:-}" ]]; then
        RUN_DIR="${OUTPUT_ROOT}/c0-$(timestamp_utc)"
    fi
    export RUN_DIR
    mkdir -p "${RUN_DIR}/logs" "${RUN_DIR}/understanding" "${RUN_DIR}/generation"
}

run_understanding() {
    new_run_dir
    "${C0_DIR}/c0_smoke.sh" real | tee "${RUN_DIR}/logs/adapter-real-smoke.log"

    local args=(
        eval
        --model showo2_qwen2_5
        --model_args "$(model_args)"
        --tasks "${TASKS}"
        --batch_size 1
        --log_samples
        --output_path "${RUN_DIR}/understanding"
    )
    if [[ -n "${LIMIT:-}" ]]; then
        args+=(--limit "${LIMIT}")
    fi
    if [[ -n "${LMMS_CACHE:-}" ]]; then
        args+=(--use_cache "${LMMS_CACHE}")
    fi

    local processes=${NUM_PROCESSES:-1}
    if (( processes > 1 )); then
        "${ACCELERATE}" launch --num_processes "${processes}" --multi_gpu \
            "${LMMS_EVAL}" "${args[@]}" \
            2>&1 | tee "${RUN_DIR}/logs/understanding.log"
    else
        "${LMMS_EVAL}" "${args[@]}" \
            2>&1 | tee "${RUN_DIR}/logs/understanding.log"
    fi
}

run_generation() {
    local action=${1:-all}
    new_run_dir
    "${C0_DIR}/c0_smoke.sh" real | tee "${RUN_DIR}/logs/adapter-real-smoke.log"
    RUN_DIR="${RUN_DIR}" "${C0_DIR}/c0_generation.sh" "${action}"
}

case "${COMMAND}" in
    preflight) "${C0_DIR}/c0_preflight.sh" full ;;
    smoke) "${C0_DIR}/c0_smoke.sh" "${2:-real}" ;;
    understanding) run_understanding ;;
    geneval) run_generation geneval ;;
    dpg) run_generation dpg ;;
    generation) run_generation all ;;
    generation-samples) run_generation samples ;;
    generation-score)
        new_run_dir
        RUN_DIR="${RUN_DIR}" "${C0_DIR}/c0_generation.sh" score
        ;;
    full-queue) "${C0_DIR}/c0_full_queue.sh" ;;
    start-full) "${C0_DIR}/c0_start_full_tmux.sh" ;;
    archive) "${C0_DIR}/c0_archive.sh" "${RUN_DIR:-}" ;;
    all)
        new_run_dir
        run_understanding
        RUN_DIR="${RUN_DIR}" "${C0_DIR}/c0_generation.sh" all
        "${C0_DIR}/c0_archive.sh" "${RUN_DIR}"
        ;;
    *)
        die "Usage: $0 [preflight|smoke [dry|real|lmms|video-dry|video-real]|understanding|full-queue|start-full|geneval|dpg|generation|generation-samples|generation-score|archive|all]"
        ;;
esac

