#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/autodl-tmp/Show-o/show-o2}"
RUN_ROOT="${RUN_ROOT:-/root/autodl-tmp/runs/showo2_drsa_benchmark}"
MODE="${MODE:-train}"
SESSION_NAME="${SESSION_NAME:-showo2_drsa_benchmark_${MODE}}"

if tmux has-session -t "=$SESSION_NAME" 2>/dev/null; then
  echo "tmux session already exists: $SESSION_NAME" >&2
  exit 2
fi

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_ID="${RUN_ID:-drsa-benchmark-${MODE}-${STAMP}}"
RUN_DIR="${RUN_DIR:-$RUN_ROOT/$RUN_ID}"
mkdir -p "$RUN_DIR"
printf '%s\n' "$RUN_DIR" > "$RUN_ROOT/latest_${MODE}.txt"

COMMAND=$(printf \
  'set -o pipefail; cd %q && exec env MODE=%q RUN_DIR=%q CUDA_VISIBLE_DEVICES=0 HF_HOME=/root/autodl-tmp/cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=%q ./evaluation/c4_drsa_adapter/run_drsa_benchmark.sh 2>&1 | tee -a %q' \
  "$ROOT" "$MODE" "$RUN_DIR" "$ROOT" "$RUN_DIR/run.log")

tmux new-session -d -s "$SESSION_NAME" "$COMMAND"
echo "session=$SESSION_NAME"
echo "run_id=$RUN_ID"
echo "run_dir=$RUN_DIR"
echo "log=$RUN_DIR/run.log"
