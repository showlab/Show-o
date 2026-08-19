#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 RUN_ROOT PURE_ROOT [LORA_EVAL_ROOT]" >&2
  exit 2
fi

ROOT=/root/autodl-tmp/Show-o/show-o2
PYTHON=${PYTHON:-/root/autodl-tmp/venvs/geneval-score/bin/python}
RUN_ROOT=$1
PURE_ROOT=$2
LORA_EVAL_ROOT=${3:-}

while true; do
  clear
  monitor_args=("${RUN_ROOT}" --pure-root "${PURE_ROOT}")
  if [[ -n "${LORA_EVAL_ROOT}" ]]; then
    monitor_args+=(--lora-eval-root "${LORA_EVAL_ROOT}")
  fi
  "${PYTHON}" \
    "${ROOT}/evaluation/c6_mask2former_alignment/monitor_mask_retention_flow_factorial.py" \
    "${monitor_args[@]}"
  if [[ -f "${RUN_ROOT}/FACTORIAL_FAILED" ]]; then
    exit 0
  fi
  if [[ -n "${LORA_EVAL_ROOT}" ]]; then
    if [[ -f "${LORA_EVAL_ROOT}/LORA_FACTORIAL_EVAL_COMPLETED" || \
          -f "${LORA_EVAL_ROOT}/LORA_FACTORIAL_EVAL_FAILED" ]]; then
      exit 0
    fi
  elif [[ -f "${RUN_ROOT}/FACTORIAL_COMPLETED" ]]; then
    exit 0
  fi
  sleep 5
done
