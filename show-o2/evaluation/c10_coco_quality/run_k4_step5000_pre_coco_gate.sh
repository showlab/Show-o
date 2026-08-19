#!/usr/bin/env bash
set -Eeuo pipefail

WORKSPACE=${WORKSPACE:-/root/autodl-tmp}
SHOWO_ROOT=${SHOWO_ROOT:-${WORKSPACE}/Show-o/show-o2}
C6_DIR=${SHOWO_ROOT}/evaluation/c6_mask2former_alignment
PYTHON=${PYTHON:-${WORKSPACE}/venvs/showo2-eval/bin/python}
QUALITY_RUN=${QUALITY_RUN:?QUALITY_RUN is required}
RUN_ROOT=${RUN_ROOT:?RUN_ROOT is required}
CHECKPOINT=${CHECKPOINT:-${WORKSPACE}/runs/showo2_mask2former_alignment/mask-flow-k4-5k-20260811T163802Z/checkpoints/step-005000.pt}
VARIANT=${VARIANT:-mask_flow_k4_5000_prompt_conditioned}
SEEDS=${SEEDS:-20260810,20260811,20260812,20260813}
MAX_ATTEMPTS=${MAX_ATTEMPTS:-2}
POLL_SECONDS=${POLL_SECONDS:-15}

QUALITY_GENERATION=${QUALITY_RUN}/generation/mask_flow_500
QUALITY_MARKER=${QUALITY_GENERATION}/GENERATION_COMPLETED
QUALITY_STATUS=${QUALITY_GENERATION}/status.json
STATUS_FILE=${RUN_ROOT}/gate.status
LOG_DIR=${RUN_ROOT}/logs
VALIDATION_DIR=${RUN_ROOT}/validation

mkdir -p "${RUN_ROOT}" "${LOG_DIR}" "${VALIDATION_DIR}"

write_status() {
    local phase=$1
    local state=$2
    local detail=${3:-none}
    local temporary=${STATUS_FILE}.tmp
    printf 'phase=%s\nstatus=%s\ndetail=%s\nupdated_at=%s\n' \
        "${phase}" "${state}" "${detail}" \
        "$(date --iso-8601=seconds)" > "${temporary}"
    mv "${temporary}" "${STATUS_FILE}"
}

on_exit() {
    local code=$?
    if [[ ${code} -ne 0 ]]; then
        touch "${RUN_ROOT}/PRE_COCO_GATE_FAILED"
        write_status failed failed "exit_code=${code};k4_coco_generation_remains_gated"
        echo "K4 step-5000 pre-COCO gate failed; K4 COCO generation remains gated" >&2
    fi
}
trap on_exit EXIT

for required in \
    "${PYTHON}" \
    "${CHECKPOINT}" \
    "${QUALITY_STATUS}" \
    "${C6_DIR}/run_flow_retention_eval_queue.sh" \
    "${C6_DIR}/validate_k4_posttrain.py"; do
    [[ -e "${required}" ]] || { echo "missing required resource: ${required}" >&2; exit 1; }
done

write_status waiting_for_mask_flow_500 running k4_coco_generation_file_gated

while [[ ! -f "${QUALITY_MARKER}" ]]; do
    sleep "${POLL_SECONDS}"
done

# The generator writes its completion marker after releasing model references.
# Leave a short process-return margin before starting the next GPU workload.
sleep 5

write_status validating_mask_flow_500 running checking_5000_rgb_pngs
"${PYTHON}" - "${QUALITY_GENERATION}" > "${VALIDATION_DIR}/mask_flow_500.json.tmp" <<'PY'
import json
import sys
from pathlib import Path

from PIL import Image

root = Path(sys.argv[1])
status = json.loads((root / "status.json").read_text(encoding="utf-8"))
if status.get("status") != "completed" or status.get("completed_images") != 5000:
    raise RuntimeError(f"incomplete Mask+Flow step-500 generation: {status}")
images = sorted((root / "images").glob("*.png"))
if len(images) != 5000:
    raise RuntimeError(f"expected 5000 generated images, found {len(images)}")
bad = []
for path in images:
    with Image.open(path) as image:
        image.load()
        if image.size != (432, 432) or image.mode != "RGB":
            bad.append({"path": path.name, "size": image.size, "mode": image.mode})
if bad:
    raise RuntimeError(f"invalid generated images: {bad[:5]}")
print(json.dumps({"status": "completed", "images": len(images), "size": [432, 432], "mode": "RGB"}, indent=2))
PY
mv "${VALIDATION_DIR}/mask_flow_500.json.tmp" "${VALIDATION_DIR}/mask_flow_500.json"

attempt=1
validated=false
while (( attempt <= MAX_ATTEMPTS )); do
    write_status benchmarking_k4_step5000 running "attempt=${attempt}/${MAX_ATTEMPTS}"
    if env \
        RUN_ROOT="${RUN_ROOT}" \
        CHECKPOINT="${CHECKPOINT}" \
        VARIANT="${VARIANT}" \
        PROTOCOL_VERSION=c6-mask-flow-k4-5000-position72-v1 \
        SEEDS="${SEEDS}" \
        "${C6_DIR}/run_flow_retention_eval_queue.sh" \
        >> "${LOG_DIR}/benchmark.log" 2>&1; then
        if "${PYTHON}" "${C6_DIR}/validate_k4_posttrain.py" step "${RUN_ROOT}" \
            --variant "${VARIANT}" --step 5000 --seeds "${SEEDS}" \
            > "${VALIDATION_DIR}/step-5000.json.tmp"; then
            mv "${VALIDATION_DIR}/step-5000.json.tmp" "${VALIDATION_DIR}/step-5000.json"
            if [[ -f "${QUALITY_RUN}/COCO_QUALITY_CANCEL_REQUESTED" ]]; then
                touch "${RUN_ROOT}/PRE_COCO_BENCHMARK_COMPLETED"
                touch "${RUN_ROOT}/PRE_COCO_GATE_FAILED"
                write_status cancelled completed k4_benchmark_validated_coco_quality_cancelled_by_request
                touch "${RUN_ROOT}/PRE_COCO_GATE_CANCELLED"
                trap - EXIT
                echo "K4 step-5000 benchmark completed; COCO quality cancelled by request"
                exit 0
            fi
            validated=true
            break
        fi
    fi
    if [[ -f "${RUN_ROOT}/QUEUE_FAILED" ]]; then
        mv "${RUN_ROOT}/QUEUE_FAILED" "${RUN_ROOT}/QUEUE_FAILED.attempt-${attempt}"
    fi
    printf 'attempt=%s failed_at=%s\n' "${attempt}" "$(date --iso-8601=seconds)" \
        >> "${LOG_DIR}/retry_history.log"
    attempt=$((attempt + 1))
    sleep 30
done

[[ "${validated}" == true ]] || {
    echo "K4 step-5000 benchmark failed validation after ${MAX_ATTEMPTS} attempts" >&2
    exit 1
}

touch "${RUN_ROOT}/PRE_COCO_BENCHMARK_COMPLETED"
write_status releasing_k4_coco_generation running all_k4_step5000_benchmarks_validated
touch "${RUN_ROOT}/QUALITY_QUEUE_RESUMED"
write_status completed completed k4_coco_file_gate_released_after_validated_benchmarks
touch "${RUN_ROOT}/PRE_COCO_GATE_COMPLETED"
trap - EXIT
echo "K4 step-5000 pre-COCO benchmark completed; quality queue resumed"
