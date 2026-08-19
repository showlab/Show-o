#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
: "${RUN_DIR:?RUN_DIR is required}"

export MAX_STEPS=${MAX_STEPS:-2}
export SAVE_EVERY=${SAVE_EVERY:-1}
export PREVIEW_EVERY=${PREVIEW_EVERY:-1}
export WEIGHT_U_TASK=${WEIGHT_U_TASK:-1.0}
export U_CYCLE_RAMP_STEPS=${U_CYCLE_RAMP_STEPS:-0}
export U_GATE_POLICY=${U_GATE_POLICY:-none}
export PROBE_COMPONENT_GRADS=${PROBE_COMPONENT_GRADS:-1}

exec "${SCRIPT_DIR}/run_prompt_u_cycle.sh"
