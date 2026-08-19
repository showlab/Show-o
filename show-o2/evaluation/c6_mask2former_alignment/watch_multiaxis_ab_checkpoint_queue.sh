#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 RUN_ROOT [INTERVAL_SECONDS]" >&2
  exit 2
fi

ROOT=/root/autodl-tmp/Show-o/show-o2
PYTHON=${PYTHON:-/root/autodl-tmp/venvs/geneval-score/bin/python}
RUN_ROOT=$1
INTERVAL=${2:-5}

exec "${PYTHON}" "${ROOT}/evaluation/c6_mask2former_alignment/monitor_multiaxis_ab_checkpoint_queue.py" \
  "${RUN_ROOT}" --session showo2_multiaxis_ab_benchmarks --watch --interval "${INTERVAL}"
