#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/autodl-tmp/Show-o/show-o2}"
PYTHON="${PYTHON:-/root/autodl-tmp/venvs/showo2-eval/bin/python}"
RUN_DIR="${RUN_DIR:?RUN_DIR is required}"
MAX_STEPS="${MAX_STEPS:-6000}"
POSITION_LIMIT_PROMPTS="${POSITION_LIMIT_PROMPTS:-18}"
POSITION_SAMPLES="${POSITION_SAMPLES:-1}"
POSITION_STEPS="${POSITION_STEPS:-12}"
VSR_LIMIT="${VSR_LIMIT:-256}"
EVAL_DIR="$RUN_DIR/evaluation"
MANIFEST="$EVAL_DIR/position_unseen72.jsonl"
VARIANT_CONFIG="$EVAL_DIR/variants.json"

mkdir -p "$EVAL_DIR"
for step in $(seq "$MAX_STEPS" -1000 1000); do
  checkpoint="$RUN_DIR/checkpoints/step-$(printf '%06d' "$step").pt"
  if [[ ! -f "$checkpoint" ]]; then
    echo "missing checkpoint: $checkpoint" >&2
    exit 1
  fi
done

"$PYTHON" "$ROOT/evaluation/c3_layout_guidance/build_position_unseen72.py" \
  --output "$MANIFEST"

"$PYTHON" - "$RUN_DIR" "$VARIANT_CONFIG" "$MAX_STEPS" <<'PY'
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
output = Path(sys.argv[2])
max_steps = int(sys.argv[3])
payload = {}
for step in range(1000, max_steps + 1, 1000):
    payload[f"vsr_{step:04d}"] = {
        "guidance_steps": 0,
        "inner_steps": 1,
        "energy_step_size": 1.0,
        "layers": [7],
        "attention_source": "qwen",
        "max_grad_rms": 0.0,
        "drsa_checkpoint": str(run_dir / "checkpoints" / f"step-{step:06d}.pt"),
    }
output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY

VARIANTS="baseline"
for step in $(seq 1000 1000 "$MAX_STEPS"); do
  VARIANTS="$VARIANTS,vsr_$(printf '%04d' "$step")"
done

"$PYTHON" -u "$ROOT/evaluation/c3_layout_guidance/run_quick8_v2.py" \
  --run-dir "$EVAL_DIR/position" \
  --protocol-version "vsr-only-${MAX_STEPS}-position-checkpoints-v1" \
  --manifest "$MANIFEST" \
  --config "$ROOT/configs/showo2_1.5b_demo_432x432.yaml" \
  --model showlab/show-o2-1.5B \
  --vae-path "$ROOT/Wan2.1_VAE.pth" \
  --variant-config-json "$VARIANT_CONFIG" \
  --variants "$VARIANTS" \
  --samples-per-prompt "$POSITION_SAMPLES" \
  --sample-batch-size "$POSITION_SAMPLES" \
  --base-seed 12026 \
  --guidance-scale 5.0 \
  --num-inference-steps "$POSITION_STEPS" \
  --guided-guidance-steps 0 \
  --inside-weight 1.0 \
  --relation-weight 0.0 \
  --cross-weight 0.0 \
  --limit-prompts "$POSITION_LIMIT_PROMPTS" \
  2>&1 | tee -a "$EVAL_DIR/position.log"

"$PYTHON" "$ROOT/evaluation/c3_layout_guidance/summarize_boxdiff_seed_ab.py" \
  --run-dir "$EVAL_DIR/position" \
  --output "$EVAL_DIR/position/summary.json" \
  2>&1 | tee -a "$EVAL_DIR/position.log"

"$PYTHON" -u "$ROOT/evaluation/c4_drsa_adapter/eval_drsa_vsr.py" \
  --run-dir "$EVAL_DIR/vsr" \
  --variant-config-json "$VARIANT_CONFIG" \
  --variants "$VARIANTS" \
  --limit "$VSR_LIMIT" \
  2>&1 | tee "$EVAL_DIR/vsr.log"

echo "VSR-only checkpoint sweep completed: $RUN_DIR"
