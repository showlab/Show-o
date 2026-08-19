#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/root/autodl-tmp/Show-o/show-o2
WORKSPACE=/root/autodl-tmp
CURRENT_RUN=${CURRENT_RUN:-${WORKSPACE}/runs/showo2_multiaxis_ab/multiaxis-ab-five-checkpoints-20260815T115549Z}
SESSION=${SESSION:-showo2_multiaxis_ab_intermediate}
RUN_ID=${RUN_ID:-multiaxis-ab-intermediate-4000-6000-8000-$(date -u +%Y%m%dT%H%M%SZ)}
RUN_ROOT=${RUN_ROOT:-${WORKSPACE}/runs/showo2_multiaxis_ab/${RUN_ID}}

export SESSION
export RUN_ROOT
export RUN_ID
export WAIT_ROOT=${CURRENT_RUN}
export WAIT_MARKER=BENCHMARK_QUEUE_COMPLETED
export WAIT_LABEL="current 93-task GenEval/Self-cycle/VQAScore/VSR queue"
export BENCHMARK_STEPS=4000,6000,8000

exec bash "${ROOT}/evaluation/c6_mask2former_alignment/start_multiaxis_ab_checkpoint_queue.sh"
