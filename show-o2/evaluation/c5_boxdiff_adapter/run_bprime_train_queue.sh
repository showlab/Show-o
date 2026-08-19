#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
WORKSPACE_ROOT=$(cd -- "${ROOT}/../.." && pwd)
PYTHON=${PYTHON:-"${WORKSPACE_ROOT}/venvs/showo2-eval/bin/python"}
RUN_ROOT=${RUN_ROOT:?Set RUN_ROOT to a new B-prime run directory}
A_CHECKPOINT=${A_CHECKPOINT:-"${WORKSPACE_ROOT}/runs/showo2_native_warmup/native-warmup-500-20260807T070000Z/checkpoints/step-000400.pt"}
DATA_MANIFEST=${DATA_MANIFEST:-"${WORKSPACE_ROOT}/runs/showo2_drsa_data/drsa-data-train-20260805T015200Z/manifest.json"}
BPRIME_MANIFEST=${BPRIME_MANIFEST:-"${WORKSPACE_ROOT}/data/drsa/bprime_coco_boxes_v1/train.jsonl"}
B1_STEPS=${B1_STEPS:-200}
B2_STEPS=${B2_STEPS:-300}

mkdir -p "${RUN_ROOT}/b1" "${RUN_ROOT}/b2" "${RUN_ROOT}/logs"
export HF_HOME=${HF_HOME:-"${WORKSPACE_ROOT}/cache/huggingface"}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

echo "phase=b1 status=starting steps=${B1_STEPS}" | tee "${RUN_ROOT}/queue.status"
"${PYTHON}" -u "${ROOT}/evaluation/c5_boxdiff_adapter/native_objective_warmup_smoke.py" \
    --run-dir "${RUN_ROOT}/b1" \
    --data-manifest "${DATA_MANIFEST}" \
    --init-checkpoint "${A_CHECKPOINT}" \
    --max-steps "${B1_STEPS}" \
    --native-steps "${B1_STEPS}" \
    --spatial-ramp-steps 1 \
    --save-every 100 \
    --warmup-steps 20 \
    --workers 0 \
    2>&1 | tee "${RUN_ROOT}/logs/b1.log"

B1_CHECKPOINT="${RUN_ROOT}/b1/checkpoints/step-$(printf '%06d' "${B1_STEPS}").pt"
if [[ ! -f "${B1_CHECKPOINT}" ]]; then
    echo "missing B1 checkpoint: ${B1_CHECKPOINT}" >&2
    exit 1
fi

echo "phase=b2 status=starting steps=${B2_STEPS}" | tee "${RUN_ROOT}/queue.status"
"${PYTHON}" -u "${ROOT}/evaluation/c5_boxdiff_adapter/train_bprime_position.py" \
    --run-dir "${RUN_ROOT}/b2" \
    --manifest "${BPRIME_MANIFEST}" \
    --init-checkpoint "${B1_CHECKPOINT}" \
    --max-steps "${B2_STEPS}" \
    --save-every 100 \
    --monitor-every 25 \
    --warmup-steps 20 \
    --lr-spatial 1e-5 \
    2>&1 | tee "${RUN_ROOT}/logs/b2.log"

echo "phase=completed status=completed" | tee "${RUN_ROOT}/queue.status"
touch "${RUN_ROOT}/TRAINING_COMPLETED"
echo "B-prime training completed: ${RUN_ROOT}"
