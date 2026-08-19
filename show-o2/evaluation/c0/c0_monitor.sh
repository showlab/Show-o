#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

RUN_DIR=${1:-}
INTERVAL=${2:-5}
if [[ -z "${RUN_DIR}" ]]; then
    latest_file="${OUTPUT_ROOT}/latest_full_run"
    [[ -f "${latest_file}" ]] || die "No latest run pointer found: ${latest_file}"
    RUN_DIR=$(<"${latest_file}")
fi
STATE_FILE="${RUN_DIR}/status.json"

while true; do
    clear 2>/dev/null || true
    date --iso-8601=seconds
    echo
    "${PYTHON}" "${C0_DIR}/c0_monitor.py" --state "${STATE_FILE}" || true
    echo
    echo "tmux:"
    tmux list-sessions 2>/dev/null | sed -n '/showo2_c0_full/p' || true
    echo
    echo "process:"
    ps -eo pid,ppid,stat,etime,%cpu,%mem,wchan:20,cmd \
        | sed -n '/[l]mms-eval eval/p;/[c]0_full_queue.sh/p' || true
    echo
    echo "GPU:"
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total \
        --format=csv,noheader,nounits 2>/dev/null || true
    echo
    df -h /root/autodl-tmp | tail -n 1
    echo
    current_log=$(
        "${PYTHON}" -c \
            'import json,sys; print(json.load(open(sys.argv[1],encoding="utf-8")).get("current_log",""))' \
            "${STATE_FILE}" 2>/dev/null || true
    )
    if [[ -n "${current_log}" && -f "${current_log}" ]]; then
        echo "Recent log:"
        tail -n 12 "${current_log}"
    fi
    [[ "${INTERVAL}" == "0" ]] && break
    sleep "${INTERVAL}"
done
