#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/Show-o/show-o2
WORKSPACE=/root/autodl-tmp
PYTHON=${PYTHON:-${WORKSPACE}/venvs/showo2-eval/bin/python}
: "${RUN_ROOT:?RUN_ROOT is required}"
TRAIN_ROOT=${TRAIN_ROOT:-${WORKSPACE}/runs/showo2_mask2former_alignment/mask2former-online-5k-v2-20260808T104000Z}
MANIFEST=${MANIFEST:-${WORKSPACE}/data/showo2_layout/position_unseen72_v1.jsonl}
B1=${B1:-${WORKSPACE}/runs/showo2_bprime/bprime-full-v2-20260808T030439Z/b1/checkpoints/step-000200.pt}

mkdir -p "${RUN_ROOT}/quick_position" "${RUN_ROOT}/full_position" "${RUN_ROOT}/logs"
export HF_HOME=${WORKSPACE}/cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

declare -A CHECKPOINTS=(
  [b1_200]="${B1}"
  [c6_0500]="${TRAIN_ROOT}/checkpoints/step-000500.pt"
  [c6_1000]="${TRAIN_ROOT}/checkpoints/step-001000.pt"
  [c6_2500]="${TRAIN_ROOT}/checkpoints/step-002500.pt"
  [c6_5000]="${TRAIN_ROOT}/checkpoints/step-005000.pt"
)
for variant in "${!CHECKPOINTS[@]}"; do
  [[ -f "${CHECKPOINTS[$variant]}" ]] || { echo "missing ${variant}: ${CHECKPOINTS[$variant]}" >&2; exit 1; }
done

CHECKPOINT_MAP="${RUN_ROOT}/checkpoint_map.json"
QUICK_CONFIG="${RUN_ROOT}/quick_variants.json"
export CHECKPOINT_MAP QUICK_CONFIG B1 TRAIN_ROOT
"${PYTHON}" - <<'PY'
import json, os
from pathlib import Path
train = Path(os.environ["TRAIN_ROOT"])
mapping = {
    "b1_200": Path(os.environ["B1"]),
    "c6_0500": train / "checkpoints/step-000500.pt",
    "c6_1000": train / "checkpoints/step-001000.pt",
    "c6_2500": train / "checkpoints/step-002500.pt",
    "c6_5000": train / "checkpoints/step-005000.pt",
}
config = {
    name: {
        "guidance_steps": 0,
        "inner_steps": 1,
        "energy_step_size": 1.0,
        "layers": [7],
        "attention_source": "qwen",
        "max_grad_rms": 0.0,
        "drsa_checkpoint": str(path.resolve()),
    }
    for name, path in mapping.items()
}
Path(os.environ["CHECKPOINT_MAP"]).write_text(json.dumps({k: str(v.resolve()) for k,v in mapping.items()}, indent=2, sort_keys=True) + "\n")
Path(os.environ["QUICK_CONFIG"]).write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
PY

echo "phase=quick_position status=running" | tee "${RUN_ROOT}/benchmark.status"
"${PYTHON}" -u "${ROOT}/evaluation/c3_layout_guidance/run_quick8_v2.py" \
  --run-dir "${RUN_ROOT}/quick_position" \
  --protocol-version "c6-mask2former-position8-gate-v1" \
  --manifest "${MANIFEST}" \
  --config "${ROOT}/configs/showo2_1.5b_demo_432x432.yaml" \
  --model showlab/show-o2-1.5B \
  --vae-path "${ROOT}/Wan2.1_VAE.pth" \
  --variant-config-json "${QUICK_CONFIG}" \
  --variants "baseline,b1_200,c6_0500,c6_1000,c6_2500,c6_5000" \
  --samples-per-prompt 4 --sample-batch-size 4 --base-seed 12026 \
  --guidance-scale 5.0 --num-inference-steps 12 --guided-guidance-steps 0 \
  --inside-weight 1.0 --cross-weight 0.0 --relation-weight 0.0 --limit-prompts 8 \
  --geneval-python "${WORKSPACE}/venvs/geneval-score/bin/python" \
  --geneval-root "${WORKSPACE}/geneval" \
  --geneval-model-config "${WORKSPACE}/mmdetection-2.28.2/mmdet/.mim/configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py" \
  --geneval-model-path "${WORKSPACE}/models/geneval" \
  --geneval-clip-path "${WORKSPACE}/models/geneval/clip/ViT-L-14.state_dict.pt" \
  2>&1 | tee "${RUN_ROOT}/logs/quick_position.log"
"${PYTHON}" "${ROOT}/evaluation/c3_layout_guidance/summarize_boxdiff_seed_ab.py" \
  --run-dir "${RUN_ROOT}/quick_position" --output "${RUN_ROOT}/quick_position/summary.json"
"${PYTHON}" "${ROOT}/evaluation/c6_mask2former_alignment/control_gate.py" \
  --run-dir "${RUN_ROOT}/quick_position" --control b1_200 \
  --candidates c6_0500,c6_1000,c6_2500,c6_5000 \
  --checkpoint-map "${CHECKPOINT_MAP}" --output "${RUN_ROOT}/quick_gate.json" --apply-gate \
  2>&1 | tee "${RUN_ROOT}/logs/quick_gate.log"

SELECTED=$("${PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("selected_variant") or "")' "${RUN_ROOT}/quick_gate.json")
if [[ -z "${SELECTED}" ]]; then
  echo "phase=completed status=gate_rejected" | tee "${RUN_ROOT}/benchmark.status"
  touch "${RUN_ROOT}/GATE_REJECTED"
  exit 0
fi
SELECTED_CHECKPOINT=$("${PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_checkpoint"])' "${RUN_ROOT}/quick_gate.json")
FULL_CONFIG="${RUN_ROOT}/full_variants.json"
export FULL_CONFIG SELECTED_CHECKPOINT
"${PYTHON}" - <<'PY'
import json, os
from pathlib import Path
base = json.loads(Path(os.environ["QUICK_CONFIG"]).read_text())
config = {"b1_200": base["b1_200"], "c6_selected": {**base["b1_200"], "drsa_checkpoint": os.environ["SELECTED_CHECKPOINT"]}}
Path(os.environ["FULL_CONFIG"]).write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
PY

echo "phase=full_position status=running selected=${SELECTED}" | tee "${RUN_ROOT}/benchmark.status"
"${PYTHON}" -u "${ROOT}/evaluation/c3_layout_guidance/run_quick8_v2.py" \
  --run-dir "${RUN_ROOT}/full_position" \
  --protocol-version "c6-mask2former-position72-gate-v1" \
  --manifest "${MANIFEST}" \
  --config "${ROOT}/configs/showo2_1.5b_demo_432x432.yaml" \
  --model showlab/show-o2-1.5B --vae-path "${ROOT}/Wan2.1_VAE.pth" \
  --variant-config-json "${FULL_CONFIG}" --variants "baseline,b1_200,c6_selected" \
  --samples-per-prompt 4 --sample-batch-size 4 --base-seed 12026 \
  --guidance-scale 5.0 --num-inference-steps 12 --guided-guidance-steps 0 \
  --inside-weight 1.0 --cross-weight 0.0 --relation-weight 0.0 --limit-prompts 72 \
  --geneval-python "${WORKSPACE}/venvs/geneval-score/bin/python" \
  --geneval-root "${WORKSPACE}/geneval" \
  --geneval-model-config "${WORKSPACE}/mmdetection-2.28.2/mmdet/.mim/configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py" \
  --geneval-model-path "${WORKSPACE}/models/geneval" \
  --geneval-clip-path "${WORKSPACE}/models/geneval/clip/ViT-L-14.state_dict.pt" \
  2>&1 | tee "${RUN_ROOT}/logs/full_position.log"
"${PYTHON}" "${ROOT}/evaluation/c3_layout_guidance/summarize_boxdiff_seed_ab.py" \
  --run-dir "${RUN_ROOT}/full_position" --output "${RUN_ROOT}/full_position/summary.json"

echo "phase=completed status=completed selected=${SELECTED}" | tee "${RUN_ROOT}/benchmark.status"
touch "${RUN_ROOT}/BENCHMARK_COMPLETED"
