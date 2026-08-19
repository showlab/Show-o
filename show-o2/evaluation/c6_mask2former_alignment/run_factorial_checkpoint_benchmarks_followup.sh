#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/root/autodl-tmp/Show-o/show-o2
WORKSPACE=/root/autodl-tmp
PYTHON=${PYTHON:-${WORKSPACE}/venvs/geneval-score/bin/python}
RUN_ROOT=${RUN_ROOT:?RUN_ROOT is required}
PURE_ROOT=${PURE_ROOT:?PURE_ROOT is required}
FACTORIAL_ROOT=${FACTORIAL_ROOT:?FACTORIAL_ROOT is required}
TRAIN_MANIFEST=${TRAIN_MANIFEST:-${WORKSPACE}/data/drsa/bprime_coco_geneval_v2/train.jsonl}
SELF_CYCLE_DATASET=${SELF_CYCLE_DATASET:-${WORKSPACE}/data/showo2_self_cycle/unicycle/self_built.jsonl}
MAX_STEPS=${MAX_STEPS:-17100}
SAVE_STEPS=${SAVE_STEPS:-100,200,500,1000,2000,4000,5000,6000,8000,10000,13000,17100}
BENCHMARK_STEPS=${BENCHMARK_STEPS:-100,500,1000,5000,17100}
EVAL_SEED=${EVAL_SEED:-20260810}
WAIT_SECONDS=${WAIT_SECONDS:-30}
QUEUE_STATUS=${RUN_ROOT}/queue.status
COMPACT_REPORT=${RUN_ROOT}/compact96_stratified_v1.json
PREFLIGHT=${RUN_ROOT}/training_preflight.json
SUMMARY=${RUN_ROOT}/factorial_checkpoint_benchmark_summary.json

VARIANTS=(mask_logits mask_flow mask_teacher_replay mask_flow_teacher_replay)
ROUTES=(g_adapter lora)
IFS=',' read -r -a benchmark_steps <<< "${BENCHMARK_STEPS}"
TOTAL_TASKS=$((${#VARIANTS[@]} * ${#ROUTES[@]} * ${#benchmark_steps[@]}))

mkdir -p "${RUN_ROOT}"

write_status() {
  local state=$1
  local completed=$2
  local active=${3:-none}
  local detail=${4:-none}
  local temporary=${QUEUE_STATUS}.tmp
  printf 'status=%s\ncompleted=%s/%s\nactive=%s\ndetail=%s\nupdated_at=%s\n' \
    "${state}" "${completed}" "${TOTAL_TASKS}" "${active}" "${detail}" \
    "$(date --iso-8601=seconds)" > "${temporary}"
  mv "${temporary}" "${QUEUE_STATUS}"
}

completed=0
active=wait_for_training
on_exit() {
  local code=$?
  if [[ ${code} -ne 0 ]]; then
    write_status failed "${completed}" "${active}" "exit_code=${code}"
    touch "${RUN_ROOT}/CHECKPOINT_BENCHMARKS_FAILED"
  fi
}
trap on_exit EXIT

while [[ ! -f "${FACTORIAL_ROOT}/FACTORIAL_COMPLETED" ]]; do
  if [[ -f "${FACTORIAL_ROOT}/FACTORIAL_FAILED" ]]; then
    echo "factorial training failed: ${FACTORIAL_ROOT}" >&2
    exit 1
  fi
  detail=$(tr '\n' ' ' < "${FACTORIAL_ROOT}/queue.status" 2>/dev/null || true)
  write_status waiting 0 "${active}" "${detail:-factorial_training_pending}"
  sleep "${WAIT_SECONDS}"
done

SELF_CYCLE_RECORD_INDICES=$(
  "${PYTHON}" "${ROOT}/evaluation/c6_mask2former_alignment/select_compact_self_cycle.py" \
    --dataset "${SELF_CYCLE_DATASET}" \
    --size 96 \
    --seed "${EVAL_SEED}" \
    --report "${COMPACT_REPORT}" \
    --print-csv
)

validator_args=(
  --pure-root "${PURE_ROOT}"
  --factorial-root "${FACTORIAL_ROOT}"
  --benchmark-root "${RUN_ROOT}"
  --train-manifest "${TRAIN_MANIFEST}"
  --self-cycle-dataset "${SELF_CYCLE_DATASET}"
  --compact-report "${COMPACT_REPORT}"
  --max-steps "${MAX_STEPS}"
  --save-steps "${SAVE_STEPS}"
  --benchmark-steps "${BENCHMARK_STEPS}"
  --seed "${EVAL_SEED}"
)

active=preflight
write_status running 0 "${active}" validating_training_and_checkpoints
"${PYTHON}" \
  "${ROOT}/evaluation/c6_mask2former_alignment/validate_factorial_checkpoint_benchmarks.py" \
  "${validator_args[@]}" --preflight-only --output "${PREFLIGHT}" \
  > "${PREFLIGHT}.log"

for step in "${benchmark_steps[@]}"; do
  for variant in "${VARIANTS[@]}"; do
    if [[ "${variant}" == mask_logits ]]; then
      pair_root=${PURE_ROOT}
    else
      pair_root=${FACTORIAL_ROOT}/${variant}
    fi
    for route in "${ROUTES[@]}"; do
      checkpoint=${pair_root}/${route}/checkpoints/step-$(printf '%06d' "${step}").pt
      eval_root=${RUN_ROOT}/${variant}/${route}/step-$(printf '%06d' "${step}")
      eval_variant=${variant}_${route}_step_$(printf '%06d' "${step}")
      active=${variant}/${route}/step-${step}
      if [[ -f "${eval_root}/QUEUE_COMPLETED" ]]; then
        completed=$((completed + 1))
        write_status running "${completed}" "${active}" already_completed
        continue
      fi
      [[ -f "${checkpoint}" ]] || {
        echo "missing benchmark checkpoint: ${checkpoint}" >&2
        exit 1
      }
      write_status running "${completed}" "${active}" \
        "geneval_position72+self_cycle_compact96"
      env \
        RUN_ROOT="${eval_root}" \
        CHECKPOINT="${checkpoint}" \
        VARIANT="${eval_variant}" \
        PROTOCOL_VERSION="c6-factorial-${variant}-${route}-step${step}-v1" \
        SEEDS="${EVAL_SEED}" \
        SELF_CYCLE_PROTOCOL=self-cycle-compact96-stratified-v1 \
        SELF_CYCLE_RECORD_INDICES="${SELF_CYCLE_RECORD_INDICES}" \
        SELF_CYCLE_NUM_INFERENCE_STEPS=50 \
        "${ROOT}/evaluation/c6_mask2former_alignment/run_flow_retention_eval_queue.sh" \
        2>&1 | tee -a "${RUN_ROOT}/benchmark.log"
      completed=$((completed + 1))
      write_status running "${completed}" "${active}" completed
    done
  done
done

active=aggregate
write_status running "${completed}" "${active}" validating_all_outputs
"${PYTHON}" \
  "${ROOT}/evaluation/c6_mask2former_alignment/validate_factorial_checkpoint_benchmarks.py" \
  "${validator_args[@]}" --output "${SUMMARY}" > "${SUMMARY}.log"

active=none
write_status completed "${TOTAL_TASKS}" none all_outputs_validated
touch "${RUN_ROOT}/CHECKPOINT_BENCHMARKS_COMPLETED"
trap - EXIT
echo "Factorial checkpoint benchmarks completed: ${RUN_ROOT}"
