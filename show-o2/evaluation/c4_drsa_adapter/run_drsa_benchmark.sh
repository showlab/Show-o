#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/autodl-tmp/Show-o/show-o2}"
PYTHON="${PYTHON:-/root/autodl-tmp/venvs/showo2-eval/bin/python}"
MODE="${MODE:-train}"
: "${RUN_DIR:?RUN_DIR is required}"

MANIFEST="${MANIFEST:-/root/autodl-tmp/data/showo2_layout/position_unseen72_v1.jsonl}"
VARIANT_CONFIG="${VARIANT_CONFIG:-$ROOT/evaluation/c4_drsa_adapter/drsa_sweep_variants.json}"
ALL_VARIANTS="baseline,drsa_0500,drsa_1000,drsa_1500,drsa_2000,drsa_2500,drsa_3000"

case "$MODE" in
  smoke)
    VARIANTS="${VARIANTS:-baseline,drsa_3000}"
    LIMIT_ARGS=(--limit-prompts 1)
    VSR_LIMIT_ARGS=(--limit 4)
    INFERENCE_STEPS="${INFERENCE_STEPS:-4}"
    ;;
  train)
    VARIANTS="${VARIANTS:-$ALL_VARIANTS}"
    LIMIT_ARGS=()
    VSR_LIMIT_ARGS=()
    INFERENCE_STEPS="${INFERENCE_STEPS:-50}"
    ;;
  *)
    echo "MODE must be smoke or train" >&2
    exit 2
    ;;
 esac

mkdir -p "$RUN_DIR"
"$PYTHON" "$ROOT/evaluation/c3_layout_guidance/build_position_unseen72.py" \
  --output "$MANIFEST"

"$PYTHON" -u "$ROOT/evaluation/c3_layout_guidance/run_quick8_v2.py" \
  --run-dir "$RUN_DIR/generation" \
  --protocol-version "drsa-${MODE}-position-unseen72-v1" \
  --manifest "$MANIFEST" \
  --config "$ROOT/configs/showo2_1.5b_demo_432x432.yaml" \
  --model showlab/show-o2-1.5B \
  --vae-path "$ROOT/Wan2.1_VAE.pth" \
  --variant-config-json "$VARIANT_CONFIG" \
  --variants "$VARIANTS" \
  --samples-per-prompt 4 \
  --sample-batch-size 4 \
  --base-seed 12026 \
  --guidance-scale 5.0 \
  --num-inference-steps "$INFERENCE_STEPS" \
  --guided-guidance-steps 0 \
  --inside-weight 1.0 \
  --relation-weight 0.0 \
  --cross-weight 0.0 \
  "${LIMIT_ARGS[@]}"

"$PYTHON" "$ROOT/evaluation/c3_layout_guidance/summarize_boxdiff_seed_ab.py" \
  --run-dir "$RUN_DIR/generation" \
  --output "$RUN_DIR/generation/summary.json"

"$PYTHON" -u "$ROOT/evaluation/c4_drsa_adapter/eval_drsa_vsr.py" \
  --run-dir "$RUN_DIR/understanding" \
  --variant-config-json "$VARIANT_CONFIG" \
  --variants "$VARIANTS" \
  "${VSR_LIMIT_ARGS[@]}"
