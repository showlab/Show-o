#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 3 ]]; then
  echo "usage: $0 RUN_DIR [TRAIN_TMUX_SESSION] [INTERVAL_SECONDS]" >&2
  exit 2
fi

RUN_DIR="$1"
TRAIN_SESSION="${2:-showo2_mask2former_5k}"
INTERVAL="${3:-5}"
SHOWO2_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/venvs/geneval-score/bin/python}"

while true; do
  clear
  "${PYTHON_BIN}" \
    "${SHOWO2_ROOT}/evaluation/c6_mask2former_alignment/monitor.py" \
    "${RUN_DIR}" \
    --session "${TRAIN_SESSION}"

  STATUS="$("${PYTHON_BIN}" -c \
    'import json,sys; print(json.load(open(sys.argv[1])).get("status", "pending"))' \
    "${RUN_DIR}/status.json" 2>/dev/null || echo pending)"
  if [[ "${STATUS}" == "completed" || "${STATUS}" == "failed" ]]; then
    exit 0
  fi
  sleep "${INTERVAL}"
done

