#!/usr/bin/env bash
set -euo pipefail

C4_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SHOWO2_ROOT=$(cd -- "${C4_DIR}/../.." && pwd)
WORKSPACE_ROOT=$(cd -- "${SHOWO2_ROOT}/../.." && pwd)
MARKER="${WORKSPACE_ROOT}/runs/showo2_drsa_data/latest.txt"
SESSION_NAME=${SESSION_NAME:-showo2_drsa_data}

[[ -f "${MARKER}" ]] || {
    echo "ERROR: latest DRSA data run marker not found: ${MARKER}" >&2
    exit 2
}
RUN_DIR=$(tr -d '\n' < "${MARKER}")
"${WORKSPACE_ROOT}/venvs/showo2-eval/bin/python" \
    "${C4_DIR}/monitor_drsa_data.py" "${RUN_DIR}" --session "${SESSION_NAME}"
