#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/Show-o/show-o2
WORKSPACE=/root/autodl-tmp
PYTHON=${PYTHON:-${WORKSPACE}/venvs/showo2-eval/bin/python}
RUN_ROOT=${RUN_ROOT:?RUN_ROOT is required}
MASKLOGITS_CHECKPOINT=${MASKLOGITS_CHECKPOINT:-${WORKSPACE}/runs/showo2_mask2former_rollback/masklogits-step500/checkpoint.pt}
B1_CHECKPOINT=${B1_CHECKPOINT:-${WORKSPACE}/runs/showo2_bprime/bprime-full-v2-20260808T030439Z/b1/checkpoints/step-000200.pt}
MANIFEST=${MANIFEST:-${WORKSPACE}/data/showo2_layout/position_unseen72_v1.jsonl}

mkdir -p "${RUN_ROOT}/vsr" "${RUN_ROOT}/full_position" "${RUN_ROOT}/logs"
export HF_HOME=${HF_HOME:-${WORKSPACE}/cache/huggingface}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

for required in "${MASKLOGITS_CHECKPOINT}" "${B1_CHECKPOINT}" "${MANIFEST}" "${ROOT}/Wan2.1_VAE.pth"; do
    [[ -f "${required}" ]] || { echo "missing required file: ${required}" >&2; exit 1; }
done

VARIANT_CONFIG="${RUN_ROOT}/variants.json"
CHECKPOINT_MAP="${RUN_ROOT}/checkpoint_map.json"
export VARIANT_CONFIG CHECKPOINT_MAP MASKLOGITS_CHECKPOINT B1_CHECKPOINT
"${PYTHON}" - <<PY
import json
import os
from pathlib import Path

mapping = {
    "b1_200": str(Path(os.environ["B1_CHECKPOINT"]).resolve()),
    "masklogits_0500": str(Path(os.environ["MASKLOGITS_CHECKPOINT"]).resolve()),
}
variants = {
    name: {
        "guidance_steps": 0,
        "inner_steps": 1,
        "energy_step_size": 1.0,
        "layers": [7],
        "attention_source": "qwen",
        "max_grad_rms": 0.0,
        "drsa_checkpoint": checkpoint,
    }
    for name, checkpoint in mapping.items()
}
Path(os.environ["VARIANT_CONFIG"]).write_text(
    json.dumps(variants, indent=2, sort_keys=True) + "\n"
)
Path(os.environ["CHECKPOINT_MAP"]).write_text(
    json.dumps(mapping, indent=2, sort_keys=True) + "\n"
)
PY

on_exit() {
    code=$?
    if [[ ${code} -ne 0 ]]; then
        echo "phase=failed status=failed exit_code=${code}" | tee "${RUN_ROOT}/benchmark.status"
    fi
}
trap on_exit EXIT

echo "phase=vsr status=running" | tee "${RUN_ROOT}/benchmark.status"
"${PYTHON}" -u "${ROOT}/evaluation/c4_drsa_adapter/eval_drsa_vsr.py"     --run-dir "${RUN_ROOT}/vsr"     --variant-config-json "${VARIANT_CONFIG}"     --variants "baseline,b1_200,masklogits_0500"     --model showlab/show-o2-1.5B     --vae-path "${ROOT}/Wan2.1_VAE.pth"     --data-root "${WORKSPACE}/data/drsa"     2>&1 | tee "${RUN_ROOT}/logs/vsr.log"

echo "phase=full_position status=running" | tee "${RUN_ROOT}/benchmark.status"
"${PYTHON}" -u "${ROOT}/evaluation/c3_layout_guidance/run_quick8_v2.py"     --run-dir "${RUN_ROOT}/full_position"     --protocol-version "c6-masklogits-step500-full-position72-v1"     --manifest "${MANIFEST}"     --config "${ROOT}/configs/showo2_1.5b_demo_432x432.yaml"     --model showlab/show-o2-1.5B     --vae-path "${ROOT}/Wan2.1_VAE.pth"     --variant-config-json "${VARIANT_CONFIG}"     --variants "baseline,b1_200,masklogits_0500"     --samples-per-prompt 4     --sample-batch-size 4     --base-seed 12026     --guidance-scale 5.0     --num-inference-steps 12     --guided-guidance-steps 0     --inside-weight 1.0     --cross-weight 0.0     --relation-weight 0.0     --limit-prompts 72     --geneval-python "${WORKSPACE}/venvs/geneval-score/bin/python"     --geneval-root "${WORKSPACE}/geneval"     --geneval-model-config "${WORKSPACE}/mmdetection-2.28.2/mmdet/.mim/configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py"     --geneval-model-path "${WORKSPACE}/models/geneval"     --geneval-clip-path "${WORKSPACE}/models/geneval/clip/ViT-L-14.state_dict.pt"     2>&1 | tee "${RUN_ROOT}/logs/full_position.log"

"${PYTHON}" "${ROOT}/evaluation/c3_layout_guidance/summarize_boxdiff_seed_ab.py"     --run-dir "${RUN_ROOT}/full_position"     --output "${RUN_ROOT}/full_position/summary.json"     2>&1 | tee -a "${RUN_ROOT}/logs/full_position.log"

"${PYTHON}" "${ROOT}/evaluation/c6_mask2former_alignment/control_gate.py"     --run-dir "${RUN_ROOT}/full_position"     --control b1_200     --candidates masklogits_0500     --checkpoint-map "${CHECKPOINT_MAP}"     --output "${RUN_ROOT}/full_gate.json"     --apply-gate     2>&1 | tee "${RUN_ROOT}/logs/full_gate.log"

echo "phase=completed status=completed" | tee "${RUN_ROOT}/benchmark.status"
touch "${RUN_ROOT}/BENCHMARK_COMPLETED"
trap - EXIT
echo "Mask-logits step-500 full benchmark completed: ${RUN_ROOT}"
