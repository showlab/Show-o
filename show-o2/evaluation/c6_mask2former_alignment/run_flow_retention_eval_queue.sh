#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/Show-o/show-o2
WORKSPACE=/root/autodl-tmp
PYTHON=${PYTHON:-${WORKSPACE}/venvs/showo2-eval/bin/python}
RUN_ROOT=${RUN_ROOT:?RUN_ROOT is required}
CHECKPOINT=${CHECKPOINT:-${WORKSPACE}/runs/showo2_mask2former_alignment/mask-flow-retention-5k-20260810T132235Z/checkpoints/step-005000.pt}
POSITION_MANIFEST=${POSITION_MANIFEST:-${WORKSPACE}/data/showo2_layout/position_unseen72_v1.jsonl}
SELF_CYCLE_DATASET=${SELF_CYCLE_DATASET:-${WORKSPACE}/data/showo2_self_cycle/unicycle/self_built.jsonl}
SELF_CYCLE_PROTOCOL=${SELF_CYCLE_PROTOCOL:-full689-v1}
SELF_CYCLE_RECORD_INDICES=${SELF_CYCLE_RECORD_INDICES:-}
SELF_CYCLE_LIMIT_PROMPTS=${SELF_CYCLE_LIMIT_PROMPTS:-}
SELF_CYCLE_NUM_INFERENCE_STEPS=${SELF_CYCLE_NUM_INFERENCE_STEPS:-50}
VARIANT=${VARIANT:-mask_flow_retention_5000_prompt_conditioned}
PROTOCOL_VERSION=${PROTOCOL_VERSION:-c6-mask-flow-retention-5000-position72-v1}
SEEDS=${SEEDS:-20260810,20260811,20260812,20260813}
IFS=',' read -r -a seed_values <<< "${SEEDS}"
TOTAL_TASKS=$((1 + ${#seed_values[@]}))

GENEVAL_DIR=${RUN_ROOT}/geneval_position72
SELF_CYCLE_ROOT=${RUN_ROOT}/self_cycle
LOG_DIR=${RUN_ROOT}/logs
VARIANT_CONFIG=${RUN_ROOT}/variants.json
QUEUE_STATUS=${RUN_ROOT}/queue.status

mkdir -p "${GENEVAL_DIR}" "${SELF_CYCLE_ROOT}" "${LOG_DIR}"
export HF_HOME=${HF_HOME:-${WORKSPACE}/cache/huggingface}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

if [[ -n "${SELF_CYCLE_RECORD_INDICES}" && -n "${SELF_CYCLE_LIMIT_PROMPTS}" ]]; then
    echo "SELF_CYCLE_RECORD_INDICES and SELF_CYCLE_LIMIT_PROMPTS are mutually exclusive" >&2
    exit 1
fi
self_cycle_subset_args=()
if [[ -n "${SELF_CYCLE_RECORD_INDICES}" ]]; then
    self_cycle_subset_args+=(--record-indices "${SELF_CYCLE_RECORD_INDICES}")
elif [[ -n "${SELF_CYCLE_LIMIT_PROMPTS}" ]]; then
    self_cycle_subset_args+=(--limit-prompts "${SELF_CYCLE_LIMIT_PROMPTS}")
fi

for required in \
    "${PYTHON}" \
    "${CHECKPOINT}" \
    "${POSITION_MANIFEST}" \
    "${SELF_CYCLE_DATASET}" \
    "${ROOT}/Wan2.1_VAE.pth"; do
    [[ -e "${required}" ]] || { echo "missing required resource: ${required}" >&2; exit 1; }
done

"${PYTHON}" - "${VARIANT_CONFIG}" "${CHECKPOINT}" "${VARIANT}" <<'PY'
import json
import sys
from pathlib import Path

output, checkpoint, variant = sys.argv[1:]
config = {
    variant: {
        "guidance_steps": 0,
        "inner_steps": 1,
        "energy_step_size": 1.0,
        "layers": [7],
        "attention_source": "qwen",
        "max_grad_rms": 0.0,
        "drsa_checkpoint": str(Path(checkpoint).resolve()),
    }
}
Path(output).write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
PY

write_status() {
    local benchmark=$1
    local seed=$2
    local completed=$3
    local state=$4
    printf 'benchmark=%s model=%s seed=%s completed=%s/%s status=%s self_cycle_protocol=%s updated_at=%s\n' \
        "${benchmark}" "${VARIANT}" "${seed}" "${completed}" "${TOTAL_TASKS}" "${state}" \
        "${SELF_CYCLE_PROTOCOL}" \
        "$(date --iso-8601=seconds)" > "${QUEUE_STATUS}"
}

on_exit() {
    local code=$?
    if [[ ${code} -ne 0 ]]; then
        if [[ -n "${ACTIVE_DIR:-}" ]]; then
            touch "${ACTIVE_DIR}/BENCHMARK_FAILED"
        fi
        touch "${RUN_ROOT}/QUEUE_FAILED"
        printf 'status=failed exit_code=%s updated_at=%s\n' \
            "${code}" "$(date --iso-8601=seconds)" >> "${QUEUE_STATUS}"
    fi
}
trap on_exit EXIT

completed=0
if [[ -f "${GENEVAL_DIR}/BENCHMARK_COMPLETED" ]]; then
    completed=$((completed + 1))
else
    ACTIVE_DIR=${GENEVAL_DIR}
    write_status geneval_position72 none "${completed}" running
    "${PYTHON}" -u "${ROOT}/evaluation/c3_layout_guidance/run_quick8_v2.py" \
        --run-dir "${GENEVAL_DIR}" \
        --protocol-version "${PROTOCOL_VERSION}" \
        --manifest "${POSITION_MANIFEST}" \
        --config "${ROOT}/configs/showo2_1.5b_demo_432x432.yaml" \
        --model showlab/show-o2-1.5B \
        --vae-path "${ROOT}/Wan2.1_VAE.pth" \
        --variant-config-json "${VARIANT_CONFIG}" \
        --variants "${VARIANT}" \
        --samples-per-prompt 4 \
        --sample-batch-size 4 \
        --base-seed 12026 \
        --guidance-scale 5.0 \
        --num-inference-steps 12 \
        --guided-guidance-steps 0 \
        --inside-weight 1.0 \
        --cross-weight 0.0 \
        --relation-weight 0.0 \
        --limit-prompts 72 \
        --geneval-python "${WORKSPACE}/venvs/geneval-score/bin/python" \
        --geneval-root "${WORKSPACE}/geneval" \
        --geneval-model-config "${WORKSPACE}/mmdetection-2.28.2/mmdet/.mim/configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py" \
        --geneval-model-path "${WORKSPACE}/models/geneval" \
        --geneval-clip-path "${WORKSPACE}/models/geneval/clip/ViT-L-14.state_dict.pt" \
        2>&1 | tee -a "${LOG_DIR}/geneval_position72.log"
    touch "${GENEVAL_DIR}/BENCHMARK_COMPLETED"
    ACTIVE_DIR=
    completed=$((completed + 1))
fi

for seed in "${seed_values[@]}"; do
    run_dir="${SELF_CYCLE_ROOT}/seed-${seed}/${VARIANT}"
    if [[ -f "${run_dir}/BENCHMARK_COMPLETED" ]]; then
        completed=$((completed + 1))
        continue
    fi
    mkdir -p "${run_dir}"
    ACTIVE_DIR=${run_dir}
    write_status self_cycle "${seed}" "${completed}" running
    "${PYTHON}" -u "${ROOT}/evaluation/c8_self_cycle/run_self_cycle.py" \
        --run-dir "${run_dir}" \
        --dataset "${SELF_CYCLE_DATASET}" \
        --config "${ROOT}/configs/showo2_1.5b_demo_432x432.yaml" \
        --model showlab/show-o2-1.5B \
        --variant-name "${VARIANT}" \
        --drsa-checkpoint "${CHECKPOINT}" \
        --vae-path "${ROOT}/Wan2.1_VAE.pth" \
        --guidance-scale 5.0 \
        --num-inference-steps "${SELF_CYCLE_NUM_INFERENCE_STEPS}" \
        --generation-batch-size 4 \
        --samples-per-prompt 1 \
        --base-seed "${seed}" \
        --max-open-answer-tokens 12 \
        "${self_cycle_subset_args[@]}" \
        2>&1 | tee -a "${run_dir}/run.log"
    touch "${run_dir}/BENCHMARK_COMPLETED"
    ACTIVE_DIR=
    completed=$((completed + 1))
done

write_status none none "${TOTAL_TASKS}" completed
touch "${RUN_ROOT}/QUEUE_COMPLETED"
trap - EXIT
echo "Flow-retention evaluation queue completed: ${RUN_ROOT}"
