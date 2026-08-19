#!/usr/bin/env bash
set -euo pipefail

# Smoke-gated full B-prime run. The old 8k experiment is kept intact.
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
WORKSPACE_ROOT=$(cd -- "${ROOT}/../.." && pwd)
PYTHON=${PYTHON:-"${WORKSPACE_ROOT}/venvs/showo2-eval/bin/python"}
RUN_ROOT=${RUN_ROOT:?Set RUN_ROOT to a new run directory}
A_CHECKPOINT=${A_CHECKPOINT:-"${WORKSPACE_ROOT}/runs/showo2_native_warmup/native-warmup-500-20260807T070000Z/checkpoints/step-000400.pt"}
DATA_MANIFEST=${DATA_MANIFEST:-"${WORKSPACE_ROOT}/runs/showo2_drsa_data/drsa-data-train-20260805T015200Z/manifest.json"}
BPRIME_MANIFEST=${BPRIME_MANIFEST:-"${WORKSPACE_ROOT}/data/drsa/bprime_coco_boxes_v2_fulltrain/train.jsonl"}
SMOKE_INIT_CHECKPOINT=${SMOKE_INIT_CHECKPOINT:-"${A_CHECKPOINT}"}
B1_STEPS=${B1_STEPS:-200}
B2_STEPS=${B2_STEPS:-17820}

mkdir -p "${RUN_ROOT}/smoke" "${RUN_ROOT}/logs"
export HF_HOME=${HF_HOME:-"${WORKSPACE_ROOT}/cache/huggingface"}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

echo "phase=unit_smoke status=starting" | tee "${RUN_ROOT}/queue.status"
"${PYTHON}" -m unittest \
    evaluation.c5_boxdiff_adapter.test_bprime_position_loss \
    evaluation.c5_boxdiff_adapter.test_staged_adapter_loss \
    2>&1 | tee "${RUN_ROOT}/logs/unit_smoke.log"

echo "phase=position_cuda_smoke status=starting" | tee -a "${RUN_ROOT}/queue.status"
"${PYTHON}" -u "${ROOT}/evaluation/c5_boxdiff_adapter/train_bprime_position.py" \
    --run-dir "${RUN_ROOT}/smoke/position" \
    --manifest "${BPRIME_MANIFEST}" \
    --init-checkpoint "${SMOKE_INIT_CHECKPOINT}" \
    --max-steps 1 \
    --save-every 1 \
    --monitor-every 1 \
    --warmup-steps 1 \
    --lr-spatial 1e-5 \
    2>&1 | tee "${RUN_ROOT}/logs/position_cuda_smoke.log"

if [[ ! -f "${RUN_ROOT}/smoke/position/status.json" ]]; then
    echo "position smoke did not produce status.json" >&2
    exit 1
fi
if ! grep -q '"status": "completed"' "${RUN_ROOT}/smoke/position/status.json"; then
    echo "position smoke did not complete successfully" >&2
    exit 1
fi

echo "phase=training status=starting b1_steps=${B1_STEPS} b2_steps=${B2_STEPS}" | tee -a "${RUN_ROOT}/queue.status"
RUN_ROOT="${RUN_ROOT}" \
A_CHECKPOINT="${A_CHECKPOINT}" \
DATA_MANIFEST="${DATA_MANIFEST}" \
BPRIME_MANIFEST="${BPRIME_MANIFEST}" \
B1_STEPS="${B1_STEPS}" \
B2_STEPS="${B2_STEPS}" \
    "${ROOT}/evaluation/c5_boxdiff_adapter/run_bprime_train_queue.sh" \
    2>&1 | tee "${RUN_ROOT}/logs/training_queue.log"

echo "phase=completed status=completed" | tee -a "${RUN_ROOT}/queue.status"
touch "${RUN_ROOT}/TRAINING_COMPLETED"
echo "B-prime smoke and training completed: ${RUN_ROOT}"
