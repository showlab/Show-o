#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

RUN_DIR=${1:-}
INTERVAL=${2:-5}
SESSION=${SESSION:-showo2_c0_generation}
if [[ -z "${RUN_DIR}" ]]; then
    latest_file="${OUTPUT_ROOT}/latest_full_run"
    [[ -f "${latest_file}" ]] || die "No latest run pointer found: ${latest_file}"
    RUN_DIR=$(<"${latest_file}")
fi

GENEVAL_SAMPLES="${RUN_DIR}/generation/geneval/samples"
GENEVAL_RESULTS="${RUN_DIR}/generation/geneval/results/results.jsonl"
DPG_SAMPLES="${RUN_DIR}/generation/dpg/samples"
DPG_RESULTS="${RUN_DIR}/generation/dpg/results/scores.txt"
GENEVAL_LOG="${RUN_DIR}/logs/geneval-rank0.log"
DPG_LOG="${RUN_DIR}/logs/dpg-rank0.log"
GENEVAL_SCORE_LOG="${RUN_DIR}/logs/geneval-score.log"
DPG_SCORE_LOG="${RUN_DIR}/logs/dpg-score.log"

count_files() {
    local root=$1
    local pattern=$2
    if [[ -d "${root}" ]]; then
        find "${root}" -type f -name "${pattern}" -printf '.\n' 2>/dev/null | wc -l
    else
        echo 0
    fi
}

count_lines() {
    local path=$1
    if [[ -f "${path}" ]]; then
        awk 'NF {count++} END {print count+0}' "${path}"
    else
        echo 0
    fi
}

percent() {
    local done=$1
    local total=$2
    local scaled=$((done * 10000 / total))
    printf '%d.%02d' "$((scaled / 100))" "$((scaled % 100))"
}

bar() {
    local done=$1
    local total=$2
    local width=40
    local filled=$((done * width / total))
    local empty=$((width - filled))
    printf '['
    printf '%*s' "${filled}" '' | tr ' ' '#'
    printf '%*s' "${empty}" '' | tr ' ' '.'
    printf ']'
}

newest_age() {
    local root=$1
    local newest now
    [[ -d "${root}" ]] || { echo 'N/A'; return; }
    newest=$(find "${root}" -type f -name '*.png' -printf '%T@\n' 2>/dev/null | sort -nr | head -n 1)
    [[ -n "${newest}" ]] || { echo 'N/A'; return; }
    newest=${newest%%.*}
    now=$(date +%s)
    echo "$((now - newest))s"
}

recent_progress() {
    local path=$1
    [[ -f "${path}" ]] || return 0
    tail -c 65536 "${path}" 2>/dev/null \
        | tr '\r' '\n' \
        | sed -r 's/\x1B\[[0-9;]*[mK]//g' \
        | sed -n '/[0-9][0-9]*%.*[0-9][0-9]*\/[0-9][0-9]*/p' \
        | tail -n 1
}

while true; do
    geneval_png=$(count_files "${GENEVAL_SAMPLES}" '*.png')
    geneval_done=$((geneval_png / 4))
    (( geneval_done > 553 )) && geneval_done=553
    dpg_done=$(count_files "${DPG_SAMPLES}" '*.png')
    (( dpg_done > 1065 )) && dpg_done=1065
    geneval_scored=$(count_lines "${GENEVAL_RESULTS}")
    (( geneval_scored > 2212 )) && geneval_scored=2212
    dpg_scored=$(count_lines "${DPG_RESULTS}")
    (( dpg_scored > 1065 )) && dpg_scored=1065

    process_args=$(ps -eo args=)
    phase='waiting or transitioning'
    active_log=''
    if grep -q '[i]nference_geneval.py' <<<"${process_args}"; then
        phase='GenEval generation'
        active_log=${GENEVAL_LOG}
    elif grep -q '[i]nference_dpg.py' <<<"${process_args}"; then
        phase='DPG-Bench generation'
        active_log=${DPG_LOG}
    elif grep -q '[e]valuate_images.py' <<<"${process_args}"; then
        phase='GenEval scoring'
        active_log=${GENEVAL_SCORE_LOG}
    elif grep -q '[c]ompute_dpg_bench.py' <<<"${process_args}"; then
        phase='DPG-Bench scoring'
        active_log=${DPG_SCORE_LOG}
    elif [[ "${geneval_scored}" -eq 2212 && "${dpg_scored}" -eq 1065 ]]; then
        phase='completed'
    fi

    clear 2>/dev/null || true
    date --iso-8601=seconds
    echo
    echo "Run:        ${RUN_DIR}"
    echo "Phase:      ${phase}"
    if tmux has-session -t "${SESSION}" 2>/dev/null; then
        echo "tmux:       ${SESSION} (alive)"
    else
        echo "tmux:       ${SESSION} (not found)"
    fi
    echo
    printf 'GenEval generation:  %4d/553 prompts  %4d/2212 PNG  %s %6s%%  last_write=%s\n' \
        "${geneval_done}" "${geneval_png}" "$(bar "${geneval_done}" 553)" \
        "$(percent "${geneval_done}" 553)" "$(newest_age "${GENEVAL_SAMPLES}")"
    printf 'DPG generation:      %4d/1065 grids   %s %6s%%  last_write=%s\n' \
        "${dpg_done}" "$(bar "${dpg_done}" 1065)" \
        "$(percent "${dpg_done}" 1065)" "$(newest_age "${DPG_SAMPLES}")"
    printf 'GenEval scoring:     %4d/2212 rows    %s %6s%%\n' \
        "${geneval_scored}" "$(bar "${geneval_scored}" 2212)" "$(percent "${geneval_scored}" 2212)"
    printf 'DPG scoring:         %4d/1065 rows    %s %6s%%\n' \
        "${dpg_scored}" "$(bar "${dpg_scored}" 1065)" "$(percent "${dpg_scored}" 1065)"
    echo
    echo 'GPU:'
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total \
        --format=csv,noheader,nounits 2>/dev/null || true
    echo
    df -h /root/autodl-tmp | tail -n 1
    if [[ -n "${active_log}" ]]; then
        progress=$(recent_progress "${active_log}")
        if [[ -n "${progress}" ]]; then
            echo
            echo "Latest model progress: ${progress}"
        fi
    fi
    [[ "${INTERVAL}" == "0" ]] && break
    sleep "${INTERVAL}"
done
