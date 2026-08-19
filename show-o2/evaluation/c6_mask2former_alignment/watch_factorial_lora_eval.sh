#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 RUN_ROOT PURE_ROOT FACTORIAL_ROOT" >&2
  exit 2
fi

ROOT=/root/autodl-tmp/Show-o/show-o2
PYTHON=${PYTHON:-/root/autodl-tmp/venvs/geneval-score/bin/python}
RUN_ROOT=$1
PURE_ROOT=$2
FACTORIAL_ROOT=$3

while true; do
  clear
  "${PYTHON}" \
    "${ROOT}/evaluation/c6_mask2former_alignment/monitor_factorial_lora_eval.py" \
    "${RUN_ROOT}" --pure-root "${PURE_ROOT}" --factorial-root "${FACTORIAL_ROOT}"
  if [[ -f "${RUN_ROOT}/LORA_FACTORIAL_EVAL_COMPLETED" || \
        -f "${RUN_ROOT}/LORA_FACTORIAL_EVAL_FAILED" ]]; then
    exit 0
  fi
  sleep 5
done
