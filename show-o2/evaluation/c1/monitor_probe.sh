#!/usr/bin/env bash
set -euo pipefail

SHOWO2_ROOT="${SHOWO2_ROOT:-/root/autodl-tmp/Show-o/show-o2}"
PYTHON="${SHOWO2_EVAL_PYTHON:-/root/autodl-tmp/venvs/showo2-eval/bin/python}"
RUN_DIR="${1:-}"
INTERVAL="${2:-5}"

if [[ -z "${RUN_DIR}" ]]; then
  RUN_DIR="$(cat /root/autodl-tmp/runs/showo2_c1/latest_run.txt)"
fi

while true; do
  clear 2>/dev/null || true
  "${PYTHON}" "${SHOWO2_ROOT}/evaluation/c1/monitor_probe.py" "${RUN_DIR}"
  status="$(${PYTHON} -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "${RUN_DIR}/status.json" 2>/dev/null || true)"
  if [[ "${status}" == "completed" || "${status}" == "failed" ]]; then
    break
  fi
  sleep "${INTERVAL}"
done
