#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
WORKSPACE_ROOT=$(cd -- "${ROOT}/../.." && pwd)
PYTHON=${PYTHON:-"${WORKSPACE_ROOT}/venvs/showo2-eval/bin/python"}
TRAIN_RUN_DIR=${TRAIN_RUN_DIR:?Set TRAIN_RUN_DIR to the completed native warm-up run}
EVAL_DIR=${EVAL_DIR:-"${TRAIN_RUN_DIR}/evaluation"}
POSITION_EVAL_DIR=${POSITION_EVAL_DIR:-"${EVAL_DIR}/position_paired8_v2"}
VSR_LIMIT=${VSR_LIMIT:-256}
POSITION_LIMIT_PROMPTS=${POSITION_LIMIT_PROMPTS:-8}
POSITION_STEPS=${POSITION_STEPS:-12}
BASE_SEED=${BASE_SEED:-12026}
CHECKPOINT_STEPS=${CHECKPOINT_STEPS:-"100 200 300 400 500"}

mkdir -p "${POSITION_EVAL_DIR}" "${EVAL_DIR}/vsr"
for step in ${CHECKPOINT_STEPS}; do
    checkpoint="${TRAIN_RUN_DIR}/checkpoints/step-$(printf '%06d' "${step}").pt"
    if [[ ! -f "${checkpoint}" ]]; then
        echo "missing checkpoint: ${checkpoint}" >&2
        exit 1
    fi
done

VARIANT_CONFIG="${EVAL_DIR}/variants.json"
export VARIANT_CONFIG TRAIN_RUN_DIR
"${PYTHON}" - <<'PY'
import json
import os
from pathlib import Path

train_run = Path(os.environ["TRAIN_RUN_DIR"]).resolve()
variants = {}
for step in (100, 200, 300, 400, 500):
    variants[f"native_{step:03d}"] = {
        "guidance_steps": 0,
        "inner_steps": 1,
        "energy_step_size": 1.0,
        "layers": [7],
        "attention_source": "qwen",
        "max_grad_rms": 0.0,
        "drsa_checkpoint": str(train_run / "checkpoints" / f"step-{step:06d}.pt"),
    }
Path(os.environ["VARIANT_CONFIG"]).write_text(
    json.dumps(variants, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY

VARIANTS="baseline,native_100,native_200,native_300,native_400,native_500"
export HF_HOME=${HF_HOME:-"${WORKSPACE_ROOT}/cache/huggingface"}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

echo "[retention] VSR fixed subset: ${VSR_LIMIT} rows"
"${PYTHON}" -u "${ROOT}/evaluation/c4_drsa_adapter/eval_drsa_vsr.py" \
    --run-dir "${EVAL_DIR}/vsr" \
    --variant-config-json "${VARIANT_CONFIG}" \
    --variants "${VARIANTS}" \
    --model showlab/show-o2-1.5B \
    --vae-path "${ROOT}/Wan2.1_VAE.pth" \
    --data-root "${WORKSPACE_ROOT}/data/drsa" \
    --limit "${VSR_LIMIT}" \
    2>&1 | tee "${EVAL_DIR}/vsr.log"

echo "[retention] Position paired subset: ${POSITION_LIMIT_PROMPTS} prompts, seed=${BASE_SEED}"
"${PYTHON}" -u "${ROOT}/evaluation/c3_layout_guidance/run_quick8_v2.py" \
    --run-dir "${POSITION_EVAL_DIR}" \
    --protocol-version "native-warmup-position-retention-v1" \
    --manifest "${WORKSPACE_ROOT}/data/showo2_layout/position_unseen72_v1.jsonl" \
    --config "${ROOT}/configs/showo2_1.5b_demo_432x432.yaml" \
    --model showlab/show-o2-1.5B \
    --vae-path "${ROOT}/Wan2.1_VAE.pth" \
    --variant-config-json "${VARIANT_CONFIG}" \
    --variants "${VARIANTS}" \
    --samples-per-prompt 4 \
    --sample-batch-size 4 \
    --base-seed "${BASE_SEED}" \
    --guidance-scale 5.0 \
    --num-inference-steps "${POSITION_STEPS}" \
    --guided-guidance-steps 0 \
    --inside-weight 1.0 \
    --cross-weight 0.0 \
    --relation-weight 0.0 \
    --limit-prompts "${POSITION_LIMIT_PROMPTS}" \
    --geneval-python "${WORKSPACE_ROOT}/venvs/geneval-score/bin/python" \
    --geneval-root "${WORKSPACE_ROOT}/geneval" \
    --geneval-model-config "${WORKSPACE_ROOT}/mmdetection-2.28.2/mmdet/.mim/configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py" \
    --geneval-model-path "${WORKSPACE_ROOT}/models/geneval" \
    --geneval-clip-path "${WORKSPACE_ROOT}/models/geneval/clip/ViT-L-14.state_dict.pt" \
    2>&1 | tee "${EVAL_DIR}/position.log"

"${PYTHON}" "${ROOT}/evaluation/c3_layout_guidance/summarize_boxdiff_seed_ab.py" \
    --run-dir "${POSITION_EVAL_DIR}" \
    --output "${POSITION_EVAL_DIR}/summary.json" \
    2>&1 | tee -a "${POSITION_EVAL_DIR}.log"

echo "Native warm-up retention evaluation completed: ${TRAIN_RUN_DIR}"
