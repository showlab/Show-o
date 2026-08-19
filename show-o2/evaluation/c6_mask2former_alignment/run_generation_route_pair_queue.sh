#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/root/autodl-tmp/Show-o/show-o2
WORKSPACE=/root/autodl-tmp
RUN_ID=${RUN_ID:-generation-route-pair-$(date -u +%Y%m%dT%H%M%SZ)}
RUN_ROOT=${RUN_ROOT:-${WORKSPACE}/runs/showo2_generation_route_pair/${RUN_ID}}
PYTHON=${PYTHON:-${WORKSPACE}/venvs/geneval-score/bin/python}
MAX_STEPS=${MAX_STEPS:-5000}
WEIGHT_U_TASK=${WEIGHT_U_TASK:-1.0}
WEIGHT_FLOW=${WEIGHT_FLOW:-0.25}
WEIGHT_TEACHER_RETENTION=${WEIGHT_TEACHER_RETENTION:-0.20}
ROLLOUT_STEPS=${ROLLOUT_STEPS:-4}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-zero_init_direct_cycle}
TRAIN_MANIFEST=${TRAIN_MANIFEST:-${WORKSPACE}/data/drsa/mask2former_alignment_5k/train.jsonl}
SAVE_STEPS=${SAVE_STEPS:-}
WAIT_FOR_BENCHMARK_ROOT=${WAIT_FOR_BENCHMARK_ROOT:-}
WAIT_SECONDS=${WAIT_SECONDS:-30}
EVAL_G_ADAPTER=${EVAL_G_ADAPTER:-0}
EVAL_SEED=${EVAL_SEED:-20260810}
TRAIN_SEED=${TRAIN_SEED:-20260808}
EVAL_VARIANT=${EVAL_VARIANT:-${EXPERIMENT_NAME}_g_adapter_${MAX_STEPS}}
EVAL_ROOT=${RUN_ROOT}/g_adapter/evaluation/step-${MAX_STEPS}
ZERO_INIT_COMPARISON=${RUN_ROOT}/g_adapter/${EXPERIMENT_NAME}_comparison.json
QUEUE_STATUS=${RUN_ROOT}/queue.status

if "${PYTHON}" - "${WEIGHT_U_TASK}" <<'PY'
import sys
raise SystemExit(0 if float(sys.argv[1]) > 0.0 else 1)
PY
then
  EXPECTED_U_CYCLE=on
else
  EXPECTED_U_CYCLE=off
fi

mkdir -p "${RUN_ROOT}"

"${PYTHON}" - "${RUN_ROOT}/queue_config.json" \
  "${MAX_STEPS}" "${SAVE_STEPS}" "${TRAIN_MANIFEST}" "${EXPERIMENT_NAME}" \
  "${WEIGHT_U_TASK}" "${WEIGHT_FLOW}" "${WEIGHT_TEACHER_RETENTION}" \
  "${ROLLOUT_STEPS}" "${EVAL_SEED}" "${EVAL_G_ADAPTER}" "${TRAIN_SEED}" <<'PY'
import json
import sys
from pathlib import Path

(
    output,
    max_steps,
    save_steps,
    manifest,
    experiment,
    weight_u,
    weight_flow,
    weight_retention,
    rollout_steps,
    eval_seed,
    eval_g_adapter,
    training_seed,
) = sys.argv[1:]
payload = {
    "status": "configured",
    "max_steps": int(max_steps),
    "save_steps": [int(value) for value in save_steps.split(",") if value],
    "manifest": str(Path(manifest).resolve()),
    "experiment": experiment,
    "weights": {
        "u_task": float(weight_u),
        "flow": float(weight_flow),
        "teacher_retention": float(weight_retention),
    },
    "teacher_replay": {
        "enabled": float(weight_retention) > 0.0,
        "implementation": "frozen Show-o2 velocity retention on original_caption",
        "weight": float(weight_retention),
    },
    "rollout_steps": int(rollout_steps),
    "eval_seed": int(eval_seed),
    "eval_g_adapter": bool(int(eval_g_adapter)),
    "training_seed": int(training_seed),
    "routes": ["g_adapter", "lora"],
}
path = Path(output)
temporary = path.with_suffix(path.suffix + ".tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
temporary.replace(path)
PY

write_status() {
  local route=$1
  local state=$2
  local detail=${3:-none}
  local temporary=${QUEUE_STATUS}.tmp
  printf 'route=%s\nstatus=%s\ndetail=%s\nupdated_at=%s\n' \
    "${route}" "${state}" "${detail}" "$(date --iso-8601=seconds)" > "${temporary}"
  mv "${temporary}" "${QUEUE_STATUS}"
}

on_exit() {
  local code=$?
  if [[ ${code} -ne 0 ]]; then
    write_status "${active_route:-none}" failed "exit_code=${code}"
    touch "${RUN_ROOT}/QUEUE_FAILED"
  fi
}
trap on_exit EXIT

if [[ "${EVAL_G_ADAPTER}" == 1 && -z "${WAIT_FOR_BENCHMARK_ROOT}" ]]; then
  echo "EVAL_G_ADAPTER=1 requires WAIT_FOR_BENCHMARK_ROOT for comparison" >&2
  exit 1
fi

if [[ -n "${WAIT_FOR_BENCHMARK_ROOT}" ]]; then
  active_route=upstream_benchmarks
  while [[ ! -f "${WAIT_FOR_BENCHMARK_ROOT}/BENCHMARK_QUEUE_COMPLETED" ]]; do
    if [[ -f "${WAIT_FOR_BENCHMARK_ROOT}/BENCHMARK_QUEUE_FAILED" ]]; then
      echo "upstream benchmark queue failed: ${WAIT_FOR_BENCHMARK_ROOT}" >&2
      exit 1
    fi
    upstream_detail=$("${PYTHON}" - "${WAIT_FOR_BENCHMARK_ROOT}/queue_status.json" <<'PY'
import json
import sys

try:
    data = json.load(open(sys.argv[1], encoding="utf-8"))
except (OSError, json.JSONDecodeError):
    data = {}
if data.get("status") == "failed":
    raise SystemExit(2)
print(
    f"completed={data.get('completed_tasks', 0)}/{data.get('total_tasks', '?')} "
    f"current={data.get('current_task', 'pending')}"
)
PY
    ) || {
      echo "upstream benchmark queue reported failure" >&2
      exit 1
    }
    write_status upstream_benchmarks waiting "${upstream_detail}"
    sleep "${WAIT_SECONDS}"
  done
  [[ -f "${WAIT_FOR_BENCHMARK_ROOT}/comparison_summary.json" ]] || {
    echo "upstream completed without comparison_summary.json" >&2
    exit 1
  }
  write_status upstream_benchmarks completed all_artifacts_validated
fi

for route in g_adapter lora; do
  active_route=${route}
  route_dir=${RUN_ROOT}/${route}
  mkdir -p "${route_dir}"
  if [[ -s "${route_dir}/status.json" ]] && \
     "${PYTHON}" - "${route_dir}/status.json" "${route}" <<'PY'
import json
import sys

status = json.load(open(sys.argv[1], encoding="utf-8"))
raise SystemExit(0 if (
    status.get("status") == "completed"
    and status.get("adaptation", {}).get("route") == sys.argv[2]
) else 1)
PY
  then
    write_status "${route}" completed "already_completed"
  else
    write_status "${route}" running "step=0/${MAX_STEPS}"
    resume_checkpoint=
    if [[ -s "${route_dir}/latest_checkpoint.txt" ]]; then
      resume_checkpoint=$(<"${route_dir}/latest_checkpoint.txt")
    fi
    env \
      RUN_DIR="${route_dir}" \
      MANIFEST="${TRAIN_MANIFEST}" \
      SEED="${TRAIN_SEED}" \
      ADAPTATION_ROUTE="${route}" \
      MAX_STEPS="${MAX_STEPS}" \
      SAVE_STEPS="${SAVE_STEPS}" \
      WEIGHT_U_TASK="${WEIGHT_U_TASK}" \
      WEIGHT_FLOW="${WEIGHT_FLOW}" \
      WEIGHT_TEACHER_RETENTION="${WEIGHT_TEACHER_RETENTION}" \
      PARAMETER_MATCH_TOLERANCE=0.00001 \
      RESUME_CHECKPOINT="${resume_checkpoint}" \
      ROLLOUT_STEPS="${ROLLOUT_STEPS}" \
      U_TASK_MODE=relation_choice \
      U_GATE_POLICY=strict \
      U_GATE_MARGIN=0.0 \
      U_CYCLE_RAMP_STEPS=100 \
      "${ROOT}/evaluation/c6_mask2former_alignment/run_prompt_u_cycle.sh" \
      2>&1 | tee -a "${route_dir}/run.log"
    "${PYTHON}" - "${route_dir}/status.json" "${route}" <<'PY'
import json
import sys

status = json.load(open(sys.argv[1], encoding="utf-8"))
route = sys.argv[2]
if status.get("status") != "completed":
    raise SystemExit(f"{route} did not complete: {status.get('status')}")
actual = status.get("adaptation", {}).get("route")
if actual != route:
    raise SystemExit(f"route mismatch: expected={route} actual={actual}")
PY
    write_status "${route}" completed "step=${MAX_STEPS}/${MAX_STEPS}"
  fi

  if [[ "${route}" == g_adapter && "${EVAL_G_ADAPTER}" == 1 ]]; then
    active_route=g_adapter_evaluation
    checkpoint=${route_dir}/checkpoints/step-$(printf '%06d' "${MAX_STEPS}").pt
    validation=${route_dir}/${EXPERIMENT_NAME}_validation.json
    if ! "${PYTHON}" \
      "${ROOT}/evaluation/c6_mask2former_alignment/validate_zero_init_direct_cycle.py" \
      --train-root "${route_dir}" \
      --eval-root "${EVAL_ROOT}" \
      --variant "${EVAL_VARIANT}" \
      --step "${MAX_STEPS}" \
      --seed "${EVAL_SEED}" \
      --warmup-comparison "${WAIT_FOR_BENCHMARK_ROOT}/comparison_summary.json" \
      --experiment-name "${EXPERIMENT_NAME}" \
      --expected-u-cycle "${EXPECTED_U_CYCLE}" \
      --expected-rollout-steps "${ROLLOUT_STEPS}" \
      --expected-flow-weight "${WEIGHT_FLOW}" \
      --expected-retention-weight "${WEIGHT_TEACHER_RETENTION}" \
      --expected-manifest "${TRAIN_MANIFEST}" \
      --expected-manifest-rows "${MAX_STEPS}" \
      --expected-save-steps "${SAVE_STEPS}" \
      --output "${ZERO_INIT_COMPARISON}" \
      > "${validation}.tmp" 2>/dev/null; then
      rm -f "${validation}.tmp"
      write_status g_adapter_evaluation running \
        "geneval_position72+self_cycle_seed_${EVAL_SEED}"
      env \
        RUN_ROOT="${EVAL_ROOT}" \
        CHECKPOINT="${checkpoint}" \
        VARIANT="${EVAL_VARIANT}" \
        PROTOCOL_VERSION="c6-${EXPERIMENT_NAME}-${MAX_STEPS}-position72-v1" \
        SEEDS="${EVAL_SEED}" \
        "${ROOT}/evaluation/c6_mask2former_alignment/run_flow_retention_eval_queue.sh" \
        2>&1 | tee -a "${route_dir}/${EXPERIMENT_NAME}_eval.log"
      "${PYTHON}" \
        "${ROOT}/evaluation/c6_mask2former_alignment/validate_zero_init_direct_cycle.py" \
        --train-root "${route_dir}" \
        --eval-root "${EVAL_ROOT}" \
        --variant "${EVAL_VARIANT}" \
        --step "${MAX_STEPS}" \
        --seed "${EVAL_SEED}" \
        --warmup-comparison "${WAIT_FOR_BENCHMARK_ROOT}/comparison_summary.json" \
        --experiment-name "${EXPERIMENT_NAME}" \
        --expected-u-cycle "${EXPECTED_U_CYCLE}" \
        --expected-rollout-steps "${ROLLOUT_STEPS}" \
        --expected-flow-weight "${WEIGHT_FLOW}" \
        --expected-retention-weight "${WEIGHT_TEACHER_RETENTION}" \
        --expected-manifest "${TRAIN_MANIFEST}" \
        --expected-manifest-rows "${MAX_STEPS}" \
        --expected-save-steps "${SAVE_STEPS}" \
        --output "${ZERO_INIT_COMPARISON}" \
        > "${validation}.tmp"
    fi
    mv "${validation}.tmp" "${validation}"
    touch "${route_dir}/ZERO_INIT_EVAL_COMPLETED"
    write_status g_adapter_evaluation completed comparison_validated
  fi
done

"${PYTHON}" - "${RUN_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = {}
for route in ("g_adapter", "lora"):
    status = json.load(open(root / route / "status.json", encoding="utf-8"))
    rows[route] = status["adaptation"]
difference = abs(
    rows["g_adapter"]["trainable_parameters"]
    - rows["lora"]["trainable_parameters"]
)
relative = difference / rows["g_adapter"]["trainable_parameters"]
if difference > 12 or relative > 0.00001:
    raise SystemExit(
        f"parameter mismatch after training: difference={difference} relative={relative:.6%}"
    )
summary = {
    "status": "completed",
    "routes": rows,
    "absolute_parameter_difference": difference,
    "relative_parameter_difference": relative,
}
comparisons = sorted((root / "g_adapter").glob("zero_init*_comparison.json"))
if comparisons:
    summary["zero_init_experiment"] = json.load(open(comparisons[-1], encoding="utf-8"))
(root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
PY

write_status all completed both_routes_validated
touch "${RUN_ROOT}/QUEUE_COMPLETED"
trap - EXIT
echo "Generation route pair completed: ${RUN_ROOT}"
