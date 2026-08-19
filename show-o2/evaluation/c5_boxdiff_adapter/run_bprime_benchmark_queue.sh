#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
WORKSPACE_ROOT=$(cd -- "${ROOT}/../.." && pwd)
PYTHON=${PYTHON:-"${WORKSPACE_ROOT}/venvs/showo2-eval/bin/python"}
TRAIN_ROOT=${TRAIN_ROOT:?Set TRAIN_ROOT to the B-prime training directory}
RUN_ROOT=${RUN_ROOT:?Set RUN_ROOT to a new benchmark directory}
OLD_B400=${OLD_B400:-"${WORKSPACE_ROOT}/runs/showo2_stage_b/stage-b-500-20260807T145350Z/checkpoints/step-000400.pt"}
MANIFEST=${MANIFEST:-"${WORKSPACE_ROOT}/data/showo2_layout/position_unseen72_v1.jsonl"}

mkdir -p "${RUN_ROOT}/vsr" "${RUN_ROOT}/quick_position" "${RUN_ROOT}/full_position" "${RUN_ROOT}/logs"
export HF_HOME=${HF_HOME:-"${WORKSPACE_ROOT}/cache/huggingface"}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

echo "phase=waiting_training status=waiting" | tee "${RUN_ROOT}/benchmark.status"
while [[ ! -f "${TRAIN_ROOT}/TRAINING_COMPLETED" ]]; do
    sleep 10
done

B1="${TRAIN_ROOT}/b1/checkpoints/step-000200.pt"
B2_100="${TRAIN_ROOT}/b2/checkpoints/step-000100.pt"
B2_200="${TRAIN_ROOT}/b2/checkpoints/step-000200.pt"
B2_300="${TRAIN_ROOT}/b2/checkpoints/step-000300.pt"
for checkpoint in "${OLD_B400}" "${B1}" "${B2_100}" "${B2_200}" "${B2_300}"; do
    [[ -f "${checkpoint}" ]] || { echo "missing checkpoint: ${checkpoint}" >&2; exit 1; }
done

QUICK_CONFIG="${RUN_ROOT}/quick_variants.json"
export QUICK_CONFIG OLD_B400 B1 B2_100 B2_200 B2_300
"${PYTHON}" - <<'PY'
import json
import os
from pathlib import Path
mapping = {
    "old_b400": os.environ["OLD_B400"],
    "b1_200": os.environ["B1"],
    "b2_100": os.environ["B2_100"],
    "b2_200": os.environ["B2_200"],
    "b2_300": os.environ["B2_300"],
}
variants = {
    name: {
        "guidance_steps": 0,
        "inner_steps": 1,
        "energy_step_size": 1.0,
        "layers": [7],
        "attention_source": "qwen",
        "max_grad_rms": 0.0,
        "drsa_checkpoint": str(Path(path).resolve()),
    }
    for name, path in mapping.items()
}
Path(os.environ["QUICK_CONFIG"]).write_text(json.dumps(variants, indent=2, sort_keys=True) + "\n")
PY

VARIANTS="baseline,old_b400,b1_200,b2_100,b2_200,b2_300"
echo "phase=vsr status=running" | tee "${RUN_ROOT}/benchmark.status"
"${PYTHON}" -u "${ROOT}/evaluation/c4_drsa_adapter/eval_drsa_vsr.py" \
    --run-dir "${RUN_ROOT}/vsr" \
    --variant-config-json "${QUICK_CONFIG}" \
    --variants "${VARIANTS}" \
    --model showlab/show-o2-1.5B \
    --vae-path "${ROOT}/Wan2.1_VAE.pth" \
    --data-root "${WORKSPACE_ROOT}/data/drsa" \
    --limit 256 \
    2>&1 | tee "${RUN_ROOT}/logs/vsr.log"

echo "phase=quick_position status=running" | tee "${RUN_ROOT}/benchmark.status"
"${PYTHON}" -u "${ROOT}/evaluation/c3_layout_guidance/run_quick8_v2.py" \
    --run-dir "${RUN_ROOT}/quick_position" \
    --protocol-version "bprime-position8-gate-v1" \
    --manifest "${MANIFEST}" \
    --config "${ROOT}/configs/showo2_1.5b_demo_432x432.yaml" \
    --model showlab/show-o2-1.5B \
    --vae-path "${ROOT}/Wan2.1_VAE.pth" \
    --variant-config-json "${QUICK_CONFIG}" \
    --variants "${VARIANTS}" \
    --samples-per-prompt 4 \
    --sample-batch-size 4 \
    --base-seed 12026 \
    --guidance-scale 5.0 \
    --num-inference-steps 12 \
    --guided-guidance-steps 0 \
    --inside-weight 1.0 \
    --cross-weight 0.0 \
    --relation-weight 0.0 \
    --limit-prompts 8 \
    --geneval-python "${WORKSPACE_ROOT}/venvs/geneval-score/bin/python" \
    --geneval-root "${WORKSPACE_ROOT}/geneval" \
    --geneval-model-config "${WORKSPACE_ROOT}/mmdetection-2.28.2/mmdet/.mim/configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py" \
    --geneval-model-path "${WORKSPACE_ROOT}/models/geneval" \
    --geneval-clip-path "${WORKSPACE_ROOT}/models/geneval/clip/ViT-L-14.state_dict.pt" \
    2>&1 | tee "${RUN_ROOT}/logs/quick_position.log"
"${PYTHON}" "${ROOT}/evaluation/c3_layout_guidance/summarize_boxdiff_seed_ab.py" \
    --run-dir "${RUN_ROOT}/quick_position" \
    --output "${RUN_ROOT}/quick_position/summary.json" \
    2>&1 | tee -a "${RUN_ROOT}/logs/quick_position.log"

export QUICK_SUMMARY="${RUN_ROOT}/quick_position/summary.json"
export SELECTED_JSON="${RUN_ROOT}/selected_checkpoint.json"
export FULL_CONFIG="${RUN_ROOT}/full_variants.json"
"${PYTHON}" - <<'PY'
import json
import os
from pathlib import Path
summary = json.loads(Path(os.environ["QUICK_SUMMARY"]).read_text())
paths = {name: os.environ[name.upper()] for name in ("b2_100", "b2_200", "b2_300")}
transitions = summary["paired_transitions_vs_baseline"]
def rank(name):
    metric = summary["variants"][name]
    return (int(metric["correct"]), -int(transitions[name]["loss"]), -int(name.split("_")[1]))
selected = max(paths, key=rank)
payload = {
    "selected_variant": selected,
    "checkpoint": str(Path(paths[selected]).resolve()),
    "selection_rule": "highest Position-8 correct, then fewer same-seed losses, then earlier step",
    "rank": list(rank(selected)),
}
Path(os.environ["SELECTED_JSON"]).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
config = {
    "bprime_selected": {
        "guidance_steps": 0,
        "inner_steps": 1,
        "energy_step_size": 1.0,
        "layers": [7],
        "attention_source": "qwen",
        "max_grad_rms": 0.0,
        "drsa_checkpoint": payload["checkpoint"],
    }
}
Path(os.environ["FULL_CONFIG"]).write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
print(json.dumps(payload, indent=2, sort_keys=True))
PY

echo "phase=full_position status=running" | tee "${RUN_ROOT}/benchmark.status"
"${PYTHON}" -u "${ROOT}/evaluation/c3_layout_guidance/run_quick8_v2.py" \
    --run-dir "${RUN_ROOT}/full_position" \
    --protocol-version "bprime-position72-v1" \
    --manifest "${MANIFEST}" \
    --config "${ROOT}/configs/showo2_1.5b_demo_432x432.yaml" \
    --model showlab/show-o2-1.5B \
    --vae-path "${ROOT}/Wan2.1_VAE.pth" \
    --variant-config-json "${FULL_CONFIG}" \
    --variants "baseline,bprime_selected" \
    --samples-per-prompt 4 \
    --sample-batch-size 4 \
    --base-seed 12026 \
    --guidance-scale 5.0 \
    --num-inference-steps 12 \
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
    2>&1 | tee "${RUN_ROOT}/logs/full_position.log"
"${PYTHON}" "${ROOT}/evaluation/c3_layout_guidance/summarize_boxdiff_seed_ab.py" \
    --run-dir "${RUN_ROOT}/full_position" \
    --output "${RUN_ROOT}/full_position/summary.json" \
    2>&1 | tee -a "${RUN_ROOT}/logs/full_position.log"

echo "phase=completed status=completed" | tee "${RUN_ROOT}/benchmark.status"
touch "${RUN_ROOT}/BENCHMARK_COMPLETED"
echo "B-prime benchmark completed: ${RUN_ROOT}"
