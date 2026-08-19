#!/usr/bin/env bash
set -Eeuo pipefail

# CPU-only contract check. This deliberately never constructs Show-o2.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-/root/autodl-tmp/venvs/showo2-eval/bin/python}"
DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/data/drsa}"
RUN_ID="${RUN_ID:-drsa-cpu-preflight-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_DIR="${RUN_DIR:-/root/autodl-tmp/runs/showo2_drsa_data/${RUN_ID}}"
DRSA_ALLOW_TORCH="${DRSA_ALLOW_TORCH:-0}"
MIN_HEADROOM_MB="${MIN_HEADROOM_MB:-128}"
WAIT_INTERVAL="${WAIT_INTERVAL:-30}"

# Direct invocation is safe too: wait in this shell instead of exiting when
# the surrounding VS Code/container cgroup is temporarily full.
if [[ "${DRSA_WAIT_FOR_MEMORY:-1}" == "1" ]]; then
    while :; do
        max=$(cat /sys/fs/cgroup/memory.max 2>/dev/null || printf 'max')
        now=$(cat /sys/fs/cgroup/memory.current 2>/dev/null || printf '0')
        if [[ "$max" == "max" ]]; then
            break
        fi
        headroom=$((max - now))
        printf '[%s] waiting for memory: headroom=%s MiB (need %s MiB)\n' \
            "$(date -u +%H:%M:%S)" "$((headroom / 1024 / 1024))" "$MIN_HEADROOM_MB"
        if (( headroom >= MIN_HEADROOM_MB * 1024 * 1024 )); then
            break
        fi
        sleep "$WAIT_INTERVAL"
    done
fi

mkdir -p "$RUN_DIR"

# Keep the test process small and independent of any CUDA installation.
export CUDA_VISIBLE_DEVICES=""
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-2}"
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

status=running
write_status() {
    local final_status="$1"
    printf '{"status":"%s","run_id":"%s","data_root":"%s"}\n' \
        "$final_status" "$RUN_ID" "$DATA_ROOT" > "$RUN_DIR/status.json"
}

on_exit() {
    local code=$?
    if [[ "$code" -eq 0 ]]; then
        status=completed
        write_status completed
    else
        status=failed
        write_status failed
    fi
}
trap on_exit EXIT

write_status running
printf 'run_id=%s\nroot=%s\ndata_root=%s\n' "$RUN_ID" "$ROOT" "$DATA_ROOT" \
    | tee "$RUN_DIR/metadata.txt"

printf '\n[1/2] static source check (no imports, no model weights)\n' \
    | tee "$RUN_DIR/run.log"
"$PYTHON" -m py_compile \
    "$ROOT/models/spatial_alignment_adapter.py" \
    "$ROOT/datasets/drsa_dataset.py" \
    "$ROOT/evaluation/c4_drsa_adapter/smoke_drsa_data.py" \
    | tee "$RUN_DIR/static_check.log"

printf '\n[2/2] downloaded-data smoke (8 SPRIGHT + 16 VSR)\n' \
    | tee -a "$RUN_DIR/run.log"
"$PYTHON" -u "$ROOT/evaluation/c4_drsa_adapter/smoke_drsa_data.py" \
    --data-root "$DATA_ROOT" \
    --spright-samples 8 \
    --vsr-samples 16 \
    --require-vsr-images \
    --skip-torch-loaders \
    --output "$RUN_DIR/smoke.json" \
    2>&1 | tee "$RUN_DIR/data_smoke.log"

if [[ "$DRSA_ALLOW_TORCH" == "1" ]]; then
    printf '\n[optional] DRSA tensor tests\n' | tee -a "$RUN_DIR/run.log"
    "$PYTHON" -u -m unittest discover \
        -s "$ROOT/evaluation/c4_drsa_adapter" -p 'test_*.py' -v \
        2>&1 | tee "$RUN_DIR/unit_tests.log"
else
    printf '\nTorch tests skipped; set DRSA_ALLOW_TORCH=1 after memory upgrade.\n' \
        | tee -a "$RUN_DIR/run.log"
fi

printf '\nCPU preflight completed: %s\n' "$RUN_DIR" | tee -a "$RUN_DIR/run.log"
