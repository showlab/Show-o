#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
WORKSPACE_ROOT=$(cd -- "${ROOT}/../.." && pwd)
PYTHON=${PYTHON:-"${WORKSPACE_ROOT}/venvs/showo2-eval/bin/python"}
TRAIN_RUN_DIR=${TRAIN_RUN_DIR:-"${WORKSPACE_ROOT}/runs/showo2_stage_b/stage-b-500-20260807T145350Z"}
RUN_DIR=${RUN_DIR:?Set RUN_DIR to a new Position-72 evaluation directory}
MANIFEST=${MANIFEST:-"${WORKSPACE_ROOT}/data/showo2_layout/position_unseen72_v1.jsonl"}
SAMPLES_PER_PROMPT=${SAMPLES_PER_PROMPT:-4}
SAMPLE_BATCH_SIZE=${SAMPLE_BATCH_SIZE:-4}
INFERENCE_STEPS=${INFERENCE_STEPS:-12}
BASE_SEED=${BASE_SEED:-12026}
CHECKPOINT_STEPS=${CHECKPOINT_STEPS:-"100 200 300 400 500"}

mkdir -p "${RUN_DIR}"
if [[ ! -f "${MANIFEST}" ]]; then
    echo "missing manifest: ${MANIFEST}" >&2
    exit 1
fi

manifest_rows=$(wc -l < "${MANIFEST}")
if [[ "${manifest_rows}" -ne 72 ]]; then
    echo "expected 72 manifest rows, found ${manifest_rows}: ${MANIFEST}" >&2
    exit 1
fi

for step in ${CHECKPOINT_STEPS}; do
    checkpoint="${TRAIN_RUN_DIR}/checkpoints/step-$(printf '%06d' "${step}").pt"
    if [[ ! -f "${checkpoint}" ]]; then
        echo "missing checkpoint: ${checkpoint}" >&2
        exit 1
    fi
done

VARIANT_CONFIG="${RUN_DIR}/variants.json"
export VARIANT_CONFIG TRAIN_RUN_DIR CHECKPOINT_STEPS
"${PYTHON}" - <<'PY'
import json
import os
from pathlib import Path

train_run = Path(os.environ["TRAIN_RUN_DIR"]).resolve()
steps = [int(value) for value in os.environ["CHECKPOINT_STEPS"].split()]
variants = {}
for step in steps:
    variants[f"b_{step:03d}"] = {
        "guidance_steps": 0,
        "inner_steps": 1,
        "energy_step_size": 1.0,
        "layers": [7],
        "attention_source": "qwen",
        "max_grad_rms": 0.0,
        "drsa_checkpoint": str(
            train_run / "checkpoints" / f"step-{step:06d}.pt"
        ),
    }
Path(os.environ["VARIANT_CONFIG"]).write_text(
    json.dumps(variants, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY

VARIANTS="baseline"
for step in ${CHECKPOINT_STEPS}; do
    VARIANTS="${VARIANTS},b_$(printf '%03d' "${step}")"
done

export HF_HOME=${HF_HOME:-"${WORKSPACE_ROOT}/cache/huggingface"}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

echo "[stage-b-position72] 72 prompts x ${SAMPLES_PER_PROMPT} samples x 6 variants"
"${PYTHON}" -u "${ROOT}/evaluation/c3_layout_guidance/run_quick8_v2.py" \
    --run-dir "${RUN_DIR}" \
    --protocol-version "stage-b-position72-v1" \
    --manifest "${MANIFEST}" \
    --config "${ROOT}/configs/showo2_1.5b_demo_432x432.yaml" \
    --model showlab/show-o2-1.5B \
    --vae-path "${ROOT}/Wan2.1_VAE.pth" \
    --variant-config-json "${VARIANT_CONFIG}" \
    --variants "${VARIANTS}" \
    --samples-per-prompt "${SAMPLES_PER_PROMPT}" \
    --sample-batch-size "${SAMPLE_BATCH_SIZE}" \
    --base-seed "${BASE_SEED}" \
    --guidance-scale 5.0 \
    --num-inference-steps "${INFERENCE_STEPS}" \
    --guided-guidance-steps 0 \
    --inside-weight 1.0 \
    --cross-weight 0.0 \
    --relation-weight 0.0 \
    --limit-prompts 72 \
    --geneval-python "${WORKSPACE_ROOT}/venvs/geneval-score/bin/python" \
    --geneval-root "${WORKSPACE_ROOT}/geneval" \
    --geneval-model-config "${WORKSPACE_ROOT}/mmdetection-2.28.2/mmdet/.mim/configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py" \
    --geneval-model-path "${WORKSPACE_ROOT}/models/geneval" \
    --geneval-clip-path "${WORKSPACE_ROOT}/models/geneval/clip/ViT-L-14.state_dict.pt" \
    2>&1 | tee -a "${RUN_DIR}/position.log"

"${PYTHON}" "${ROOT}/evaluation/c3_layout_guidance/summarize_boxdiff_seed_ab.py" \
    --run-dir "${RUN_DIR}" \
    --output "${RUN_DIR}/summary.json" \
    2>&1 | tee -a "${RUN_DIR}/position.log"

echo "Stage B Position-72 gate completed: ${RUN_DIR}"
