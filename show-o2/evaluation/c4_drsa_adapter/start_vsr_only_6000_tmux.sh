#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/autodl-tmp/Show-o/show-o2}"
PYTHON="${PYTHON:-/root/autodl-tmp/venvs/showo2-eval/bin/python}"
RUN_ROOT="${RUN_ROOT:-/root/autodl-tmp/runs/showo2_drsa_train}"
RUN_ID="${RUN_ID:-vsr-only-6000-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_DIR="${RUN_DIR:-$RUN_ROOT/$RUN_ID}"
SESSION_NAME="${SESSION_NAME:-showo2_vsr_only_6000}"
MAX_STEPS="${MAX_STEPS:-6000}"
SAVE_EVERY="${SAVE_EVERY:-1000}"
POSITION_LIMIT_PROMPTS="${POSITION_LIMIT_PROMPTS:-18}"
POSITION_SAMPLES="${POSITION_SAMPLES:-1}"
POSITION_STEPS="${POSITION_STEPS:-12}"
VSR_LIMIT="${VSR_LIMIT:-256}"

if tmux has-session -t "=$SESSION_NAME" 2>/dev/null; then
  echo "tmux session already exists: $SESSION_NAME" >&2
  exit 2
fi

mkdir -p "$RUN_DIR"
printf '%s\n' "$RUN_DIR" > "$RUN_ROOT/latest_vsr_only.txt"

COMMAND=$(printf '%q ' \
  env CUDA_VISIBLE_DEVICES=0 \
  HF_HOME=/root/autodl-tmp/cache/huggingface \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH="$ROOT" \
  "$PYTHON" -u "$ROOT/evaluation/c4_drsa_adapter/train_vsr_only.py" \
  --run-dir "$RUN_DIR" --max-steps "$MAX_STEPS" --save-every "$SAVE_EVERY" \
  --workers 0)

EVAL_COMMAND=$(printf '%q ' \
  env CUDA_VISIBLE_DEVICES=0 \
  HF_HOME=/root/autodl-tmp/cache/huggingface \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH="$ROOT" \
  RUN_DIR="$RUN_DIR" MAX_STEPS="$MAX_STEPS" \
  POSITION_LIMIT_PROMPTS="$POSITION_LIMIT_PROMPTS" POSITION_SAMPLES="$POSITION_SAMPLES" \
  POSITION_STEPS="$POSITION_STEPS" VSR_LIMIT="$VSR_LIMIT" \
  "$ROOT/evaluation/c4_drsa_adapter/evaluate_vsr_only_checkpoints.sh")

tmux new-session -d -s "$SESSION_NAME" \
  "cd $(printf '%q' "$ROOT") && ($COMMAND 2>&1 | tee -a $(printf '%q' "$RUN_DIR/run.log")) && ($EVAL_COMMAND 2>&1 | tee -a $(printf '%q' "$RUN_DIR/evaluation.log"))"

echo "session=$SESSION_NAME"
echo "run_id=$RUN_ID"
echo "run_dir=$RUN_DIR"
echo "training_log=$RUN_DIR/run.log"
echo "evaluation_log=$RUN_DIR/evaluation.log"
echo "checkpoints=$RUN_DIR/checkpoints/step-XXXXXX.pt"
echo "position_validation=prompts:$POSITION_LIMIT_PROMPTS samples:$POSITION_SAMPLES inference_steps:$POSITION_STEPS"
echo "vsr_validation=rows:$VSR_LIMIT"
