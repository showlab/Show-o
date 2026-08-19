#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/autodl-tmp/Show-o/show-o2}"
PYTHON="${PYTHON:-/root/autodl-tmp/venvs/showo2-eval/bin/python}"
RUN_ROOT="${RUN_ROOT:-/root/autodl-tmp/runs/showo2_drsa_train}"
MODE="${MODE:-train}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"

case "$MODE" in
  smoke)
    SESSION_NAME="${SESSION_NAME:-showo2_drsa_smoke}"
    MAX_STEPS="${MAX_STEPS:-3}"
    WORKERS="${WORKERS:-0}"
    SAVE_EVERY="${SAVE_EVERY:-3}"
    RETENTION_EVERY="${RETENTION_EVERY:-3}"
    ;;
  train)
    SESSION_NAME="${SESSION_NAME:-showo2_drsa_train}"
    MAX_STEPS="${MAX_STEPS:-3000}"
    WORKERS="${WORKERS:-2}"
    SAVE_EVERY="${SAVE_EVERY:-500}"
    RETENTION_EVERY="${RETENTION_EVERY:-4}"
    ;;
  *)
    echo "MODE must be smoke or train" >&2
    exit 2
    ;;
esac

if tmux has-session -t "=$SESSION_NAME" 2>/dev/null; then
  echo "tmux session already exists: $SESSION_NAME" >&2
  exit 2
fi

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_ID="${RUN_ID:-drsa-${MODE}-${STAMP}}"
RUN_DIR="${RUN_DIR:-$RUN_ROOT/$RUN_ID}"
mkdir -p "$RUN_DIR"

RESUME_ARG=""
if [[ -n "$RESUME_CHECKPOINT" ]]; then
  printf -v RESUME_ARG ' --resume-checkpoint %q' "$RESUME_CHECKPOINT"
fi
printf '%s\n' "$RUN_DIR" > "$RUN_ROOT/latest_${MODE}.txt"

COMMAND=$(printf \
  'cd %q && exec env CUDA_VISIBLE_DEVICES=0 HF_HOME=/root/autodl-tmp/cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=%q %q -u evaluation/c4_drsa_adapter/train_drsa.py --mode %q --run-dir %q --max-steps %q --workers %q --save-every %q --retention-every %q%s 2>&1 | tee -a %q' \
  "$ROOT" "$ROOT" "$PYTHON" "$MODE" "$RUN_DIR" "$MAX_STEPS" "$WORKERS" \
  "$SAVE_EVERY" "$RETENTION_EVERY" "$RESUME_ARG" "$RUN_DIR/run.log")

tmux new-session -d -s "$SESSION_NAME" "$COMMAND"
echo "session=$SESSION_NAME"
echo "run_id=$RUN_ID"
echo "run_dir=$RUN_DIR"
echo "log=$RUN_DIR/run.log"
