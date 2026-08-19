#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/autodl-tmp/Show-o/show-o2}"
PYTHON="${PYTHON:-/root/autodl-tmp/venvs/showo2-eval/bin/python}"
exec "$PYTHON" "$ROOT/evaluation/c4_drsa_adapter/monitor_drsa_train.py" "$@"
