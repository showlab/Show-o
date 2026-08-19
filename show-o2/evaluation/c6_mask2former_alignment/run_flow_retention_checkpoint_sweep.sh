#!/usr/bin/env bash
set -euo pipefail

C6_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE=/root/autodl-tmp
RUN_ROOT=${RUN_ROOT:?RUN_ROOT is required}
TRAIN_ROOT=${TRAIN_ROOT:-${WORKSPACE}/runs/showo2_mask2former_alignment/mask-flow-retention-5k-20260810T132235Z}
STEPS=${STEPS:-500,1000,2500}
SEEDS=${SEEDS:-20260810,20260811,20260812,20260813}
SWEEP_STATUS=${RUN_ROOT}/sweep.status

mkdir -p "${RUN_ROOT}"

write_status() {
    local step=$1
    local state=$2
    printf 'step=%s status=%s updated_at=%s\n' \
        "${step}" "${state}" "$(date --iso-8601=seconds)" > "${SWEEP_STATUS}"
}

on_exit() {
    local code=$?
    if [[ ${code} -ne 0 ]]; then
        touch "${RUN_ROOT}/SWEEP_FAILED"
        printf 'status=failed exit_code=%s updated_at=%s\n' \
            "${code}" "$(date --iso-8601=seconds)" >> "${SWEEP_STATUS}"
    fi
}
trap on_exit EXIT

IFS=',' read -r -a step_values <<< "${STEPS}"
for step in "${step_values[@]}"; do
    checkpoint="${TRAIN_ROOT}/checkpoints/step-$(printf '%06d' "${step}").pt"
    subroot="${RUN_ROOT}/step-${step}"
    variant="mask_flow_retention_${step}_prompt_conditioned"
    protocol="c6-mask-flow-retention-${step}-position72-v1"
    [[ -f "${checkpoint}" ]] || { echo "missing checkpoint: ${checkpoint}" >&2; exit 1; }
    if [[ -f "${subroot}/QUEUE_COMPLETED" ]]; then
        continue
    fi
    mkdir -p "${subroot}"
    write_status "${step}" running
    RUN_ROOT="${subroot}" \
    CHECKPOINT="${checkpoint}" \
    VARIANT="${variant}" \
    PROTOCOL_VERSION="${protocol}" \
    SEEDS="${SEEDS}" \
        "${C6_DIR}/run_flow_retention_eval_queue.sh" \
        2>&1 | tee -a "${subroot}/worker.log"
done

write_status none completed
touch "${RUN_ROOT}/SWEEP_COMPLETED"
trap - EXIT
echo "Flow-retention checkpoint sweep completed: ${RUN_ROOT}"
