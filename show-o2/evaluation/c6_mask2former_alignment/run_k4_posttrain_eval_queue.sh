#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/root/autodl-tmp/Show-o/show-o2
WORKSPACE=/root/autodl-tmp
C6_DIR=${ROOT}/evaluation/c6_mask2former_alignment
PYTHON=${PYTHON:-${WORKSPACE}/venvs/geneval-score/bin/python}
VQA_PYTHON=${VQA_PYTHON:-${WORKSPACE}/venvs/showo2-score-vqa/bin/python}
VQA_MODEL=${VQA_MODEL:-${WORKSPACE}/models/c7/Qwen2.5-VL-7B-Instruct}
RUN_ROOT=${RUN_ROOT:?RUN_ROOT is required}
TRAIN_ROOT=${TRAIN_ROOT:-${WORKSPACE}/runs/showo2_mask2former_alignment/mask-flow-k4-5k-20260811T163802Z}
TRAIN_SESSION=${TRAIN_SESSION:-showo2_mask_flow_k4_5k}
STEPS=${STEPS:-500,1000,2500}
SEEDS=${SEEDS:-20260810,20260811,20260812,20260813}
WAIT_SECONDS=${WAIT_SECONDS:-30}
MAX_ATTEMPTS=${MAX_ATTEMPTS:-2}

BASELINE_ROOT=${BASELINE_ROOT:-${WORKSPACE}/runs/showo2_mask2former_rollback/masklogits-step500-full-20260809T090000Z/full_position/baseline/geneval}
MASK_LOGITS_500_ROOT=${MASK_LOGITS_500_ROOT:-${WORKSPACE}/runs/showo2_mask2former_rollback/masklogits-step500-full-20260809T090000Z/full_position/masklogits_0500/geneval}
MASK_FLOW_500_ROOT=${MASK_FLOW_500_ROOT:-${WORKSPACE}/runs/showo2_mask2former_benchmark/mask-flow-retention-checkpoint-sweep-20260811T013302Z/step-500/geneval_position72/mask_flow_retention_500_prompt_conditioned/geneval}

QUEUE_STATUS=${RUN_ROOT}/queue.status
LOG_DIR=${RUN_ROOT}/logs
VALIDATION_DIR=${RUN_ROOT}/validation
VQA_OUTPUT=${RUN_ROOT}/vqascore_fourway
VQA_VARIANTS=baseline,mask_logits_500,mask_flow_500,mask_flow_k4_500
ACTIVE_TASK=initializing

mkdir -p "${RUN_ROOT}" "${LOG_DIR}" "${VALIDATION_DIR}"

export HF_HOME=${HF_HOME:-${WORKSPACE}/cache/huggingface}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

write_status() {
    local phase=$1
    local state=$2
    local detail=${3:-none}
    local temporary=${QUEUE_STATUS}.tmp
    printf 'phase=%s\nstatus=%s\ndetail=%s\nactive_task=%s\nupdated_at=%s\n' \
        "${phase}" "${state}" "${detail}" "${ACTIVE_TASK}" \
        "$(date --iso-8601=seconds)" > "${temporary}"
    mv "${temporary}" "${QUEUE_STATUS}"
}

on_exit() {
    local code=$?
    if [[ ${code} -ne 0 ]]; then
        touch "${RUN_ROOT}/POSTTRAIN_FAILED"
        write_status failed failed "exit_code=${code}"
    fi
}
trap on_exit EXIT

for required in \
    "${PYTHON}" \
    "${VQA_PYTHON}" \
    "${VQA_MODEL}/model.safetensors.index.json" \
    "${TRAIN_ROOT}/status.json" \
    "${BASELINE_ROOT}" \
    "${MASK_LOGITS_500_ROOT}" \
    "${MASK_FLOW_500_ROOT}" \
    "${C6_DIR}/run_flow_retention_eval_queue.sh" \
    "${C6_DIR}/validate_k4_posttrain.py"; do
    [[ -e "${required}" ]] || { echo "missing required resource: ${required}" >&2; exit 1; }
done

ACTIVE_TASK=wait_for_k4_training
while true; do
    read -r train_state train_step train_total < <(
        "${PYTHON}" - "${TRAIN_ROOT}/status.json" <<'PY'
import json
import sys

try:
    data = json.load(open(sys.argv[1], encoding="utf-8"))
except (OSError, json.JSONDecodeError):
    data = {}
print(data.get("status", "pending"), data.get("step", 0), data.get("total_steps", 5000))
PY
    )
    write_status waiting_for_training running "step=${train_step}/${train_total}"
    if [[ "${train_state}" == completed ]]; then
        break
    fi
    if [[ "${train_state}" == failed ]]; then
        echo "K=4 training reported failure" >&2
        exit 1
    fi
    if ! tmux has-session -t "=${TRAIN_SESSION}" 2>/dev/null; then
        echo "K=4 training tmux disappeared before status=completed" >&2
        exit 1
    fi
    sleep "${WAIT_SECONDS}"
done

ACTIVE_TASK=validate_training
write_status validating_training running
"${PYTHON}" "${C6_DIR}/validate_k4_posttrain.py" training "${TRAIN_ROOT}" \
    --steps 500,1000,2500,5000 > "${VALIDATION_DIR}/training.json"

run_step() {
    local step=$1
    local checkpoint=${TRAIN_ROOT}/checkpoints/step-$(printf '%06d' "${step}").pt
    local step_root=${RUN_ROOT}/step-${step}
    local variant=mask_flow_k4_${step}_prompt_conditioned
    local protocol=c6-mask-flow-k4-${step}-position72-v1
    local validation=${VALIDATION_DIR}/step-${step}.json
    local attempt

    ACTIVE_TASK=step_${step}_geneval_and_self_cycle
    if "${PYTHON}" "${C6_DIR}/validate_k4_posttrain.py" step "${step_root}" \
        --variant "${variant}" --step "${step}" --seeds "${SEEDS}" \
        > "${validation}.tmp" 2>/dev/null; then
        mv "${validation}.tmp" "${validation}"
        write_status benchmark_step_completed running "step=${step} already_validated"
        return 0
    fi

    mkdir -p "${step_root}"
    for attempt in $(seq 1 "${MAX_ATTEMPTS}"); do
        write_status benchmarking running "step=${step} attempt=${attempt}/${MAX_ATTEMPTS}"
        if env \
            RUN_ROOT="${step_root}" \
            CHECKPOINT="${checkpoint}" \
            VARIANT="${variant}" \
            PROTOCOL_VERSION="${protocol}" \
            SEEDS="${SEEDS}" \
            "${C6_DIR}/run_flow_retention_eval_queue.sh" \
            >> "${step_root}/queue-worker.log" 2>&1; then
            if "${PYTHON}" "${C6_DIR}/validate_k4_posttrain.py" step "${step_root}" \
                --variant "${variant}" --step "${step}" --seeds "${SEEDS}" \
                > "${validation}.tmp"; then
                mv "${validation}.tmp" "${validation}"
                write_status benchmark_step_completed running "step=${step} validated"
                return 0
            fi
        fi
        printf 'step=%s attempt=%s failed_at=%s\n' \
            "${step}" "${attempt}" "$(date --iso-8601=seconds)" \
            >> "${LOG_DIR}/retry_history.log"
        sleep 30
    done
    echo "step ${step} failed validation after ${MAX_ATTEMPTS} attempts" >&2
    return 1
}

IFS=',' read -r -a step_values <<< "${STEPS}"
for step in "${step_values[@]}"; do
    run_step "${step}"
done

K4_500_ROOT=${RUN_ROOT}/step-500/geneval_position72/mask_flow_k4_500_prompt_conditioned/geneval

run_vqa() {
    local attempt
    ACTIVE_TASK=vqascore_fourway
    if "${PYTHON}" "${C6_DIR}/validate_k4_posttrain.py" vqa "${VQA_OUTPUT}" \
        --variants "${VQA_VARIANTS}" > "${VALIDATION_DIR}/vqascore.json.tmp" 2>/dev/null; then
        mv "${VALIDATION_DIR}/vqascore.json.tmp" "${VALIDATION_DIR}/vqascore.json"
        return 0
    fi

    mkdir -p "${VQA_OUTPUT}"
    for attempt in $(seq 1 "${MAX_ATTEMPTS}"); do
        write_status vqascore running "attempt=${attempt}/${MAX_ATTEMPTS}"
        if "${VQA_PYTHON}" -u "${ROOT}/evaluation/c7_generation_benchmark/score_vqa.py" \
            --variant "baseline=${BASELINE_ROOT}" \
            --variant "mask_logits_500=${MASK_LOGITS_500_ROOT}" \
            --variant "mask_flow_500=${MASK_FLOW_500_ROOT}" \
            --variant "mask_flow_k4_500=${K4_500_ROOT}" \
            --output-dir "${VQA_OUTPUT}" \
            --model-path "${VQA_MODEL}" \
            --batch-size 8 \
            --attn-implementation sdpa \
            >> "${LOG_DIR}/vqascore.log" 2>&1; then
            if "${PYTHON}" "${C6_DIR}/validate_k4_posttrain.py" vqa "${VQA_OUTPUT}" \
                --variants "${VQA_VARIANTS}" > "${VALIDATION_DIR}/vqascore.json.tmp"; then
                mv "${VALIDATION_DIR}/vqascore.json.tmp" "${VALIDATION_DIR}/vqascore.json"
                return 0
            fi
        fi
        printf 'task=vqascore attempt=%s failed_at=%s\n' \
            "${attempt}" "$(date --iso-8601=seconds)" \
            >> "${LOG_DIR}/retry_history.log"
        sleep 30
    done
    echo "VQAScore failed validation after ${MAX_ATTEMPTS} attempts" >&2
    return 1
}

run_vqa

ACTIVE_TASK=final_validation
write_status final_validation running
for step in "${step_values[@]}"; do
    variant=mask_flow_k4_${step}_prompt_conditioned
    "${PYTHON}" "${C6_DIR}/validate_k4_posttrain.py" step "${RUN_ROOT}/step-${step}" \
        --variant "${variant}" --step "${step}" --seeds "${SEEDS}" \
        > "${VALIDATION_DIR}/step-${step}.json"
done
"${PYTHON}" "${C6_DIR}/validate_k4_posttrain.py" vqa "${VQA_OUTPUT}" \
    --variants "${VQA_VARIANTS}" > "${VALIDATION_DIR}/vqascore.json"

ACTIVE_TASK=none
write_status completed completed all_artifacts_validated
touch "${RUN_ROOT}/POSTTRAIN_COMPLETED"
trap - EXIT
echo "K=4 post-training benchmark queue completed and validated: ${RUN_ROOT}"
