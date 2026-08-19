#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 FACTORIAL_ROOT PURE_ROOT BENCHMARK_ROOT" >&2
  exit 2
fi

ROOT=/root/autodl-tmp/Show-o/show-o2
PYTHON=${PYTHON:-/root/autodl-tmp/venvs/geneval-score/bin/python}
FACTORIAL_ROOT=$1
PURE_ROOT=$2
BENCHMARK_ROOT=$3

while true; do
  clear
  "${PYTHON}" \
    "${ROOT}/evaluation/c6_mask2former_alignment/monitor_factorial_checkpoint_benchmarks.py" \
    "${FACTORIAL_ROOT}" \
    --pure-root "${PURE_ROOT}" \
    --benchmark-root "${BENCHMARK_ROOT}"
  if [[ -f "${FACTORIAL_ROOT}/FACTORIAL_FAILED" || \
        -f "${BENCHMARK_ROOT}/CHECKPOINT_BENCHMARKS_COMPLETED" || \
        -f "${BENCHMARK_ROOT}/CHECKPOINT_BENCHMARKS_FAILED" ]]; then
    exit 0
  fi
  sleep 5
done
