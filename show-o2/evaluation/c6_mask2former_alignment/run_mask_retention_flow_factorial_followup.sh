#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/root/autodl-tmp/Show-o/show-o2
WORKSPACE=/root/autodl-tmp
PYTHON=${PYTHON:-${WORKSPACE}/venvs/geneval-score/bin/python}
RUN_ROOT=${RUN_ROOT:?RUN_ROOT is required}
PURE_ROOT=${PURE_ROOT:?PURE_ROOT is required}
UPSTREAM_BENCHMARK_ROOT=${UPSTREAM_BENCHMARK_ROOT:?UPSTREAM_BENCHMARK_ROOT is required}
TRAIN_MANIFEST=${TRAIN_MANIFEST:-${WORKSPACE}/data/drsa/bprime_coco_geneval_v2/train.jsonl}
MAX_STEPS=${MAX_STEPS:-17100}
SAVE_STEPS=${SAVE_STEPS:-100,200,500,1000,2000,4000,5000,6000,8000,10000,13000,17100}
WAIT_SECONDS=${WAIT_SECONDS:-30}
EVAL_SEED=${EVAL_SEED:-20260810}
QUEUE_STATUS=${RUN_ROOT}/queue.status
ACTIVE_VARIANT=wait_for_mask_logits

VARIANTS=(mask_flow mask_teacher_replay mask_flow_teacher_replay)
FLOW_WEIGHTS=(0.25 0.0 0.25)
RETENTION_WEIGHTS=(0.0 0.20 0.20)

mkdir -p "${RUN_ROOT}"

write_status() {
  local state=$1
  local completed=$2
  local detail=${3:-none}
  local temporary=${QUEUE_STATUS}.tmp
  printf 'status=%s\nactive_variant=%s\ncompleted_variants=%s/3\ndetail=%s\nupdated_at=%s\n' \
    "${state}" "${ACTIVE_VARIANT}" "${completed}" "${detail}" \
    "$(date --iso-8601=seconds)" > "${temporary}"
  mv "${temporary}" "${QUEUE_STATUS}"
}

on_exit() {
  local code=$?
  if [[ ${code} -ne 0 ]]; then
    write_status failed "${completed_variants:-0}" "exit_code=${code}"
    touch "${RUN_ROOT}/FACTORIAL_FAILED"
  fi
}
trap on_exit EXIT

completed_variants=0
while [[ ! -f "${PURE_ROOT}/QUEUE_COMPLETED" ]]; do
  if [[ -f "${PURE_ROOT}/QUEUE_FAILED" ]]; then
    echo "pure Mask-logits queue failed: ${PURE_ROOT}" >&2
    exit 1
  fi
  detail=$("${PYTHON}" - "${PURE_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for route in ("g_adapter", "lora"):
    try:
        status = json.load(open(root / route / "status.json", encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        status = {}
    rows.append(
        f"{route}={status.get('step', 0)}/{status.get('total_steps', 17100)}"
    )
print(" ".join(rows))
PY
  )
  write_status waiting 0 "${detail}"
  sleep "${WAIT_SECONDS}"
done

for index in "${!VARIANTS[@]}"; do
  variant=${VARIANTS[index]}
  flow=${FLOW_WEIGHTS[index]}
  retention=${RETENTION_WEIGHTS[index]}
  child_root=${RUN_ROOT}/${variant}
  mkdir -p "${child_root}"
  ACTIVE_VARIANT=${variant}
  if [[ -f "${child_root}/QUEUE_COMPLETED" ]]; then
    completed_variants=$((completed_variants + 1))
    write_status running "${completed_variants}" already_completed
    continue
  fi
  write_status running "${completed_variants}" \
    "flow=${flow} retention=${retention}"
  env \
    RUN_ID="${variant}" \
    RUN_ROOT="${child_root}" \
    WAIT_FOR_BENCHMARK_ROOT="${UPSTREAM_BENCHMARK_ROOT}" \
    TRAIN_MANIFEST="${TRAIN_MANIFEST}" \
    MAX_STEPS="${MAX_STEPS}" \
    SAVE_STEPS="${SAVE_STEPS}" \
    EVAL_G_ADAPTER=0 \
    EVAL_SEED="${EVAL_SEED}" \
    WEIGHT_U_TASK=0 \
    WEIGHT_FLOW="${flow}" \
    WEIGHT_TEACHER_RETENTION="${retention}" \
    ROLLOUT_STEPS=1 \
    EXPERIMENT_NAME="zero_init_${variant}_full${MAX_STEPS}" \
    "${ROOT}/evaluation/c6_mask2former_alignment/run_generation_route_pair_queue.sh" \
    2>&1 | tee -a "${child_root}/factorial-child.log"
  [[ -f "${child_root}/QUEUE_COMPLETED" ]] || {
    echo "factorial child did not validate: ${child_root}" >&2
    exit 1
  }
  completed_variants=$((completed_variants + 1))
  write_status running "${completed_variants}" validated
done

ACTIVE_VARIANT=aggregate
"${PYTHON}" - "${RUN_ROOT}" "${PURE_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root, pure = map(Path, sys.argv[1:])
variants = {"mask_logits": pure}
variants.update({name: root / name for name in (
    "mask_flow", "mask_teacher_replay", "mask_flow_teacher_replay"
)})
output = {
    "status": "completed",
    "design": "Mask-logits with Flow x frozen-teacher replay 2x2",
    "teacher_replay_implementation": "frozen Show-o2 velocity retention",
}
output["variants"] = {}
for name, path in variants.items():
    summary_path = path / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    summary = json.load(open(summary_path, encoding="utf-8"))
    if summary.get("status") != "completed":
        raise RuntimeError(f"incomplete variant: {name}")
    output["variants"][name] = {
        "run_root": str(path.resolve()),
        "summary": summary,
    }
temporary = root / "factorial_summary.json.tmp"
temporary.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
temporary.replace(root / "factorial_summary.json")
PY

ACTIVE_VARIANT=none
write_status completed 3 all_variants_validated
touch "${RUN_ROOT}/FACTORIAL_COMPLETED"
trap - EXIT
echo "Mask/Flow/Retention factorial completed: ${RUN_ROOT}"
