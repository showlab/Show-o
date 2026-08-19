#!/usr/bin/env bash
set -Eeuo pipefail

WORKSPACE=${WORKSPACE:-/root/autodl-tmp}
SHOWO_ROOT=${SHOWO_ROOT:-${WORKSPACE}/Show-o/show-o2}
PYTHON=${PYTHON:-${WORKSPACE}/venvs/showo2-eval/bin/python}
K4_RUN_ROOT=${K4_RUN_ROOT:?K4_RUN_ROOT is required}
QUALITY_RUN=${QUALITY_RUN:?QUALITY_RUN is required}
QUALITY_PGID=${QUALITY_PGID:?QUALITY_PGID is required}
GATE_PGID=${GATE_PGID:?GATE_PGID is required}
TRAIN_SESSION=${TRAIN_SESSION:-showo2_g_adapter_warm}
TRAIN_RUN_ID=${TRAIN_RUN_ID:-g-adapter-warm-$(date -u +%Y%m%dT%H%M%SZ)}
TRAIN_RUN_ROOT=${TRAIN_RUN_ROOT:-${WORKSPACE}/runs/showo2_generation_adaptation/${TRAIN_RUN_ID}}
POLL_SECONDS=${POLL_SECONDS:-30}

HANDOFF_ROOT=${HANDOFF_ROOT:-${WORKSPACE}/runs/showo2_coco_quality/k4-to-g-adapter-20260813T000000Z}
STATUS_FILE=${HANDOFF_ROOT}/handoff.status
LOG_DIR=${HANDOFF_ROOT}/logs
VALIDATION_FILE=${HANDOFF_ROOT}/k4_step5000_validation.json

mkdir -p "${HANDOFF_ROOT}" "${LOG_DIR}"

write_status() {
    local phase=$1
    local state=$2
    local detail=${3:-none}
    local tmp=${STATUS_FILE}.tmp
    printf 'phase=%s\nstatus=%s\ndetail=%s\ntrain_session=%s\ntrain_run_root=%s\nupdated_at=%s\n' \
        "${phase}" "${state}" "${detail}" "${TRAIN_SESSION}" \
        "${TRAIN_RUN_ROOT}" "$(date --iso-8601=seconds)" > "${tmp}"
    mv "${tmp}" "${STATUS_FILE}"
}

on_exit() {
    local code=$?
    if [[ ${code} -ne 0 ]]; then
        touch "${HANDOFF_ROOT}/HANDOFF_FAILED"
        write_status failed failed "exit_code=${code}"
    fi
}
trap on_exit EXIT

for required in \
    "${PYTHON}" \
    "${SHOWO_ROOT}/evaluation/c6_mask2former_alignment/validate_k4_posttrain.py" \
    "${SHOWO_ROOT}/evaluation/c6_mask2former_alignment/run_5k.sh"; do
    [[ -e "${required}" ]] || { echo "missing required resource: ${required}" >&2; exit 1; }
done

write_status waiting_for_k4 running waiting_for_gen_eval_and_four_seed_self_cycle
while [[ ! -f "${K4_RUN_ROOT}/validation/step-5000.json" ]]; do
    if [[ -f "${K4_RUN_ROOT}/QUEUE_FAILED" ]]; then
        echo "K4 benchmark queue failed" >&2
        exit 1
    fi
    sleep "${POLL_SECONDS}"
done

write_status validating_k4 running validating_gen_eval_and_self_cycle_artifacts
"${PYTHON}" "${SHOWO_ROOT}/evaluation/c6_mask2former_alignment/validate_k4_posttrain.py" step "${K4_RUN_ROOT}" \
    --variant mask_flow_k4_5000_prompt_conditioned --step 5000 \
    --seeds 20260810,20260811,20260812,20260813 > "${VALIDATION_FILE}.tmp"
mv "${VALIDATION_FILE}.tmp" "${VALIDATION_FILE}"

# Keep the requested quality experiment fail-closed. The K4 COCO process is
# waiting on this marker, so it cannot load the model after this handoff.
write_status cancelling_coco_quality running preventing_k4_coco_and_quality_scoring
touch "${QUALITY_RUN}/COCO_QUALITY_CANCEL_REQUESTED"
touch "${K4_RUN_ROOT}/PRE_COCO_GATE_FAILED"
if kill -TERM -- "-${QUALITY_PGID}" 2>/dev/null; then
    for _ in $(seq 1 30); do
        kill -0 "${QUALITY_PGID}" 2>/dev/null || break
        sleep 1
    done
    kill -KILL -- "-${QUALITY_PGID}" 2>/dev/null || true
fi
kill -TERM -- "-${GATE_PGID}" 2>/dev/null || true
touch "${QUALITY_RUN}/K4_COCO_NOT_STARTED_BY_REQUEST"

if tmux has-session -t "=${TRAIN_SESSION}" 2>/dev/null; then
    echo "training session already exists: ${TRAIN_SESSION}" >&2
    exit 1
fi

write_status starting_g_adapter running "run=${TRAIN_RUN_ID};max_steps=5000;save_every=500;warmup_steps=100"
mkdir -p "${TRAIN_RUN_ROOT}"
tmux new-session -d -s "${TRAIN_SESSION}" \
    "cd ${SHOWO_ROOT@Q} && set -o pipefail; RUN_DIR=${TRAIN_RUN_ROOT@Q} ADAPTATION_ROUTE=g_adapter ${SHOWO_ROOT@Q}/evaluation/c6_mask2former_alignment/run_5k.sh 2>&1 | tee -a ${TRAIN_RUN_ROOT@Q}/queue.log; exec bash"

write_status training_started running "session=${TRAIN_SESSION};run=${TRAIN_RUN_ROOT}"
touch "${HANDOFF_ROOT}/G_ADAPTER_TRAINING_STARTED"
trap - EXIT
echo "K4 benchmark validated; COCO quality cancelled; G-adapter warm training started: ${TRAIN_RUN_ROOT}"
