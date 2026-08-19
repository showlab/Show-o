#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/autodl-tmp/Show-o/show-o2}"
PYTHON="${PYTHON:-/root/autodl-tmp/venvs/showo2-eval/bin/python}"
RUN_ROOT="${RUN_ROOT:-/root/autodl-tmp/runs/showo2_drsa_benchmark}"
BASELINE_SOURCE="${BASELINE_SOURCE:-/root/autodl-tmp/runs/showo2_drsa_benchmark/drsa-benchmark-train-20260806T173937Z}"
SESSION_NAME="${SESSION_NAME:-showo2_drsa_benchmark_20k}"
RUN_ID="${RUN_ID:-drsa-benchmark-20k-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_DIR="${RUN_DIR:-$RUN_ROOT/$RUN_ID}"

if tmux has-session -t "=$SESSION_NAME" 2>/dev/null; then
  echo "tmux session already exists: $SESSION_NAME" >&2
  exit 2
fi
for required in \
  "$BASELINE_SOURCE/generation/baseline" \
  "$BASELINE_SOURCE/generation/paired_initial_latents.json" \
  "$BASELINE_SOURCE/understanding/baseline"; do
  test -e "$required" || { echo "Missing baseline artifact: $required" >&2; exit 2; }
done

mkdir -p "$RUN_DIR/generation" "$RUN_DIR/understanding"
cp -a "$BASELINE_SOURCE/generation/baseline" "$RUN_DIR/generation/"
cp -a "$BASELINE_SOURCE/generation/paired_initial_latents.json" "$RUN_DIR/generation/"
cp -a "$BASELINE_SOURCE/understanding/baseline" "$RUN_DIR/understanding/"
printf '%s\n' "$RUN_DIR" > "$RUN_ROOT/latest_train.txt"

COMMAND=$(printf \
  'set -o pipefail; cd %q && exec env MODE=train VARIANTS=baseline,drsa_20000 RUN_DIR=%q CUDA_VISIBLE_DEVICES=0 HF_HOME=/root/autodl-tmp/cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=%q ./evaluation/c4_drsa_adapter/run_drsa_benchmark.sh 2>&1 | tee -a %q' \
  "$ROOT" "$RUN_DIR" "$ROOT" "$RUN_DIR/run.log")

tmux new-session -d -s "$SESSION_NAME" "$COMMAND"
echo "session=$SESSION_NAME"
echo "run_id=$RUN_ID"
echo "run_dir=$RUN_DIR"
echo "baseline_source=$BASELINE_SOURCE"
echo "log=$RUN_DIR/run.log"
