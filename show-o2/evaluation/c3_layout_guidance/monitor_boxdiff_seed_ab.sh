#!/usr/bin/env bash
set -euo pipefail

C3_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SHOWO2_ROOT=$(cd -- "${C3_DIR}/../.." && pwd)
WORKSPACE_ROOT=$(cd -- "${SHOWO2_ROOT}/../.." && pwd)
MARKER="${WORKSPACE_ROOT}/runs/showo2_boxdiff_seed_ab/latest.txt"

[[ -f "${MARKER}" ]] || {
    echo "ERROR: latest-run marker not found: ${MARKER}" >&2
    exit 2
}
RUN_DIR=$(tr -d '\n' < "${MARKER}")
"${WORKSPACE_ROOT}/venvs/showo2-eval/bin/python" \
    "${C3_DIR}/monitor_quick8_v2.py" "${RUN_DIR}"
