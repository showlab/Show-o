#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

RUN_ID=${RUN_ID:-c0-full-$(timestamp_utc)}
RUN_DIR=${RUN_DIR:-"${OUTPUT_ROOT}/${RUN_ID}"}
STATE_FILE=${STATE_FILE:-"${RUN_DIR}/status.json"}
QUEUE_LOG=${QUEUE_LOG:-"${RUN_DIR}/logs/queue.log"}
TASKS_CSV=${TASKS:-"mme,gqa,mmbench_en_dev,mmmu_val,mmstar,ai2d,seedbench_image,seedbench_video"}
RESUME_COMPLETED=${RESUME_COMPLETED:-1}
mkdir -p "${RUN_DIR}/logs" "${RUN_DIR}/understanding"
IFS=',' read -r -a TASK_LIST <<<"${TASKS_CSV}"
(( ${#TASK_LIST[@]} > 0 )) || die "No benchmark tasks were configured"

declare -A TASK_SIZES
declare -A TASK_MODELS
TASKS_JSON="[]"
OVERALL_TOTAL=0
COMPLETED_BEFORE=0

progress() {
    "${PYTHON}" "${C0_DIR}/c0_progress.py" --state "${STATE_FILE}" "$@"
}

mark_failed() {
    local code=$?
    local message="Queue failed with exit code ${code}"
    progress --status failed --error "${message}" \
        --completed-before "${COMPLETED_BEFORE}" --overall-total "${OVERALL_TOTAL}" --finish || true
    echo "${message}" >&2
    exit "${code}"
}
trap mark_failed ERR

resolve_task_size() {
    local task=$1
    local size_log="${RUN_DIR}/logs/size-${task}.log"
    local marker
    echo "Resolving task data and size: ${task}"
    "${PYTHON}" "${C0_DIR}/c0_task_size.py" \
        --task "${task}" --include-path "${SHOWO2_TASKS_PATH}" 2>&1 | tee "${size_log}"
    marker=$(tr '\r' '\n' <"${size_log}" | sed -n 's/^SHOWO2_TASK_SIZE_JSON=//p' | tail -n 1)
    [[ -n "${marker}" ]] || die "Could not parse task size for ${task}; see ${size_log}"
    TASK_SIZES["${task}"]=$("${PYTHON}" -c 'import json,sys; print(json.loads(sys.argv[1])["size"])' "${marker}")
    TASKS_JSON=$(
        "${PYTHON}" -c \
            'import json,sys; rows=json.loads(sys.argv[1]); meta=json.loads(sys.argv[2]); rows.append(meta); print(json.dumps(rows,separators=(",",":")))' \
            "${TASKS_JSON}" "${marker}"
    )
    OVERALL_TOTAL=$((OVERALL_TOTAL + TASK_SIZES["${task}"]))
}

progress --run-id "${RUN_ID}" --status resolving_data --task-index 0 \
    --task-total "${#TASK_LIST[@]}" --completed-samples 0 --completed-before 0 \
    --overall-total 0 --pid "$$" --log "${QUEUE_LOG}"

size_index=0
for task in "${TASK_LIST[@]}"; do
    size_index=$((size_index + 1))
    progress --status resolving_data --task "${task}" --task-index "${size_index}" \
        --task-total "${#TASK_LIST[@]}" --pid "$$" --log "${RUN_DIR}/logs/size-${task}.log"
    if [[ "${task}" == "seedbench_video" ]]; then
        TASK_MODELS["${task}"]="video"
    else
        TASK_MODELS["${task}"]="image"
    fi
    resolve_task_size "${task}"
done

progress --run-id "${RUN_ID}" --status preflight --task-index 0 \
    --task-total "${#TASK_LIST[@]}" --completed-samples 0 --completed-before 0 \
    --overall-total "${OVERALL_TOTAL}" --pid "$$" --log "${QUEUE_LOG}" \
    --tasks-json "${TASKS_JSON}"

"${C0_DIR}/c0_preflight.sh" full
"${C0_DIR}/c0_smoke.sh" real | tee "${RUN_DIR}/logs/image-smoke.log"
"${C0_DIR}/c0_smoke.sh" video-real | tee "${RUN_DIR}/logs/video-smoke.log"
progress --status running --completed-samples 0 --completed-before 0 \
    --overall-total "${OVERALL_TOTAL}" --pid "$$"

task_index=0
for task in "${TASK_LIST[@]}"; do
    task_index=$((task_index + 1))
    expected=${TASK_SIZES["${task}"]}
    task_output="${RUN_DIR}/understanding/${task}"
    task_log="${RUN_DIR}/logs/${task}.log"
    mkdir -p "${task_output}"
    if [[ "${RESUME_COMPLETED}" == "1" ]] && \
        verification=$("${PYTHON}" "${C0_DIR}/c0_verify_task_output.py" \
            --output "${task_output}" --expected "${expected}" 2>/dev/null); then
        echo "Skipping completed task ${task_index}/${#TASK_LIST[@]}: ${task} (${expected} samples)"
        echo "Resume verification: ${verification}" | tee -a "${task_log}"
        COMPLETED_BEFORE=$((COMPLETED_BEFORE + expected))
        progress --status running --task "${task}" --task-index "${task_index}" \
            --task-total "${#TASK_LIST[@]}" --expected-samples "${expected}" \
            --completed-samples "${expected}" --completed-before "$((COMPLETED_BEFORE - expected))" \
            --overall-total "${OVERALL_TOTAL}" --pid "$$" --log "${task_log}" --error ""
        continue
    fi


    progress --status running --task "${task}" --task-index "${task_index}" \
        --task-total "${#TASK_LIST[@]}" --expected-samples "${expected}" \
        --completed-samples 0 --completed-before "${COMPLETED_BEFORE}" \
        --overall-total "${OVERALL_TOTAL}" --pid "$$" --log "${task_log}" --error ""

    export SHOWO2_PROGRESS_FILE="${STATE_FILE}"
    export SHOWO2_PROGRESS_COMPLETED_BEFORE="${COMPLETED_BEFORE}"
    export SHOWO2_PROGRESS_OVERALL_TOTAL="${OVERALL_TOTAL}"
    export SHOWO2_PROGRESS_TASK="${task}"
    if [[ "${TASK_MODELS["${task}"]}" == "video" ]]; then
        model_name=showo2_qwen2_5_video
        args=$(video_model_args)
    else
        model_name=showo2_qwen2_5
        args=$(model_args)
    fi

    echo "Starting ${task_index}/${#TASK_LIST[@]}: ${task} (${expected} samples, ${model_name})"
    "${LMMS_EVAL}" eval \
        --model "${model_name}" --model_args "${args}" --tasks "${task}" \
        --include_path "${SHOWO2_TASKS_PATH}" --batch_size 1 \
        --gen_kwargs "temperature=0,do_sample=false,top_k=1" \
        --log_samples --output_path "${task_output}" 2>&1 | tee "${task_log}"

    "${PYTHON}" "${C0_DIR}/c0_verify_task_output.py" \
        --output "${task_output}" --expected "${expected}" | tee -a "${task_log}"

    COMPLETED_BEFORE=$((COMPLETED_BEFORE + expected))
    progress --status running --task "${task}" --task-index "${task_index}" \
        --task-total "${#TASK_LIST[@]}" --expected-samples "${expected}" \
        --completed-samples "${expected}" --completed-before "$((COMPLETED_BEFORE - expected))" \
        --overall-total "${OVERALL_TOTAL}" --pid "$$" --log "${task_log}"
done

progress --status completed --task-index "${#TASK_LIST[@]}" --task-total "${#TASK_LIST[@]}" \
    --completed-samples 0 --completed-before "${OVERALL_TOTAL}" \
    --overall-total "${OVERALL_TOTAL}" --pid "$$" --log "${QUEUE_LOG}" --finish
echo "C0 full understanding benchmark completed: ${RUN_DIR}"
