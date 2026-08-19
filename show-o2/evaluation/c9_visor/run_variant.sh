#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
WORKSPACE=$(cd "${ROOT}/../.." && pwd)
PYTHON=${PYTHON:-${WORKSPACE}/venvs/showo2-eval/bin/python}
RUN_DIR=${RUN_DIR:?RUN_DIR is required}
MANIFEST=${MANIFEST:-${WORKSPACE}/data/visor/sr2d_balanced200_seed20260811.jsonl}
VARIANT_NAME=${VARIANT_NAME:?VARIANT_NAME is required}
DRSA_CHECKPOINT=${DRSA_CHECKPOINT:-}
SAMPLES_PER_PROMPT=${SAMPLES_PER_PROMPT:-4}
BASE_SEED=${BASE_SEED:-20260811}
NUM_INFERENCE_STEPS=${NUM_INFERENCE_STEPS:-50}
GENERATION_BATCH_SIZE=${GENERATION_BATCH_SIZE:-4}

checkpoint_args=()
if [[ -n "${DRSA_CHECKPOINT}" ]]; then
  checkpoint_args+=(--drsa-checkpoint "${DRSA_CHECKPOINT}")
fi

cd "${ROOT}"
"${PYTHON}" evaluation/c9_visor/generate_images.py \
  --run-dir "${RUN_DIR}/generation" \
  --manifest "${MANIFEST}" \
  --variant-name "${VARIANT_NAME}" \
  --samples-per-prompt "${SAMPLES_PER_PROMPT}" \
  --base-seed "${BASE_SEED}" \
  --num-inference-steps "${NUM_INFERENCE_STEPS}" \
  --generation-batch-size "${GENERATION_BATCH_SIZE}" \
  "${checkpoint_args[@]}"

"${PYTHON}" evaluation/c9_visor/score_owlvit.py \
  --run-dir "${RUN_DIR}/owlvit" \
  --generation-run-dir "${RUN_DIR}/generation" \
  --manifest "${MANIFEST}" \
  --samples-per-prompt "${SAMPLES_PER_PROMPT}" \
  --threshold 0.1

