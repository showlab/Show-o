#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/root/autodl-tmp/Show-o/show-o2
WORKSPACE=/root/autodl-tmp
PYTHON=${PYTHON:-${WORKSPACE}/venvs/geneval-score/bin/python}
RUN_ROOT=${RUN_ROOT:?RUN_ROOT is required}
FACTORIAL_ROOT=${FACTORIAL_ROOT:?FACTORIAL_ROOT is required}
PURE_ROOT=${PURE_ROOT:?PURE_ROOT is required}
UPSTREAM_BENCHMARK_ROOT=${UPSTREAM_BENCHMARK_ROOT:?UPSTREAM_BENCHMARK_ROOT is required}
TRAIN_MANIFEST=${TRAIN_MANIFEST:-${WORKSPACE}/data/drsa/bprime_coco_geneval_v2/train.jsonl}
MAX_STEPS=${MAX_STEPS:-17100}
SAVE_STEPS=${SAVE_STEPS:-100,200,500,1000,2000,4000,6000,8000,10000,13000,17100}
WAIT_SECONDS=${WAIT_SECONDS:-30}
EVAL_SEED=${EVAL_SEED:-20260810}
QUEUE_STATUS=${RUN_ROOT}/queue.status
ACTIVE_VARIANT=wait_for_factorial

VARIANTS=(mask_logits mask_flow mask_retention mask_flow_retention)
EXPERIMENTS=(
  zero_init_mask_logits_full17100
  zero_init_mask_flow_full17100
  zero_init_mask_retention_full17100
  zero_init_mask_flow_retention_full17100
)
FLOW_WEIGHTS=(0.0 0.25 0.0 0.25)
RETENTION_WEIGHTS=(0.0 0.0 0.20 0.20)

mkdir -p "${RUN_ROOT}"

write_status() {
  local state=$1
  local completed=$2
  local detail=${3:-none}
  local temporary=${QUEUE_STATUS}.tmp
  printf 'status=%s\nactive_variant=%s\ncompleted_evaluations=%s/4\ndetail=%s\nupdated_at=%s\n' \
    "${state}" "${ACTIVE_VARIANT}" "${completed}" "${detail}" \
    "$(date --iso-8601=seconds)" > "${temporary}"
  mv "${temporary}" "${QUEUE_STATUS}"
}

completed=0
on_exit() {
  local code=$?
  if [[ ${code} -ne 0 ]]; then
    write_status failed "${completed}" "exit_code=${code}"
    touch "${RUN_ROOT}/LORA_FACTORIAL_EVAL_FAILED"
  fi
}
trap on_exit EXIT

while [[ ! -f "${FACTORIAL_ROOT}/FACTORIAL_COMPLETED" ]]; do
  if [[ -f "${FACTORIAL_ROOT}/FACTORIAL_FAILED" ]]; then
    echo "factorial queue failed: ${FACTORIAL_ROOT}" >&2
    exit 1
  fi
  detail=$(tr '\n' ' ' < "${FACTORIAL_ROOT}/queue.status" 2>/dev/null || true)
  write_status waiting 0 "${detail:-factorial_pending}"
  sleep "${WAIT_SECONDS}"
done

for index in "${!VARIANTS[@]}"; do
  variant=${VARIANTS[index]}
  experiment=${EXPERIMENTS[index]}
  flow=${FLOW_WEIGHTS[index]}
  retention=${RETENTION_WEIGHTS[index]}
  if [[ "${variant}" == mask_logits ]]; then
    pair_root=${PURE_ROOT}
  else
    pair_root=${FACTORIAL_ROOT}/${variant}
  fi
  train_root=${pair_root}/lora
  eval_root=${train_root}/evaluation/step-${MAX_STEPS}
  checkpoint=${train_root}/checkpoints/step-$(printf '%06d' "${MAX_STEPS}").pt
  eval_variant=${experiment}_lora_${MAX_STEPS}
  validation=${train_root}/${experiment}_lora_validation.json
  comparison=${train_root}/${experiment}_lora_comparison.json
  marker=${train_root}/ZERO_INIT_EVAL_COMPLETED
  ACTIVE_VARIANT=${variant}

  validation_args=(
    --train-root "${train_root}"
    --eval-root "${eval_root}"
    --variant "${eval_variant}"
    --step "${MAX_STEPS}"
    --seed "${EVAL_SEED}"
    --warmup-comparison "${UPSTREAM_BENCHMARK_ROOT}/comparison_summary.json"
    --experiment-name "${experiment}_lora"
    --expected-route lora
    --expected-u-cycle off
    --expected-rollout-steps 1
    --expected-flow-weight "${flow}"
    --expected-retention-weight "${retention}"
    --expected-manifest "${TRAIN_MANIFEST}"
    --expected-manifest-rows "${MAX_STEPS}"
    --expected-save-steps "${SAVE_STEPS}"
    --output "${comparison}"
  )

  if "${PYTHON}" \
      "${ROOT}/evaluation/c6_mask2former_alignment/validate_zero_init_direct_cycle.py" \
      "${validation_args[@]}" > "${validation}.tmp" 2>/dev/null; then
    mv "${validation}.tmp" "${validation}"
    touch "${marker}"
    completed=$((completed + 1))
    write_status running "${completed}" already_validated
    continue
  fi
  rm -f "${validation}.tmp"

  [[ -f "${checkpoint}" ]] || { echo "missing LoRA checkpoint: ${checkpoint}" >&2; exit 1; }
  write_status running "${completed}" \
    "geneval_position72+self_cycle_seed_${EVAL_SEED}"
  env \
    RUN_ROOT="${eval_root}" \
    CHECKPOINT="${checkpoint}" \
    VARIANT="${eval_variant}" \
    PROTOCOL_VERSION="c6-${experiment}-lora-${MAX_STEPS}-position72-v1" \
    SEEDS="${EVAL_SEED}" \
    "${ROOT}/evaluation/c6_mask2former_alignment/run_flow_retention_eval_queue.sh" \
    2>&1 | tee -a "${train_root}/${experiment}_lora_eval.log"

  "${PYTHON}" \
    "${ROOT}/evaluation/c6_mask2former_alignment/validate_zero_init_direct_cycle.py" \
    "${validation_args[@]}" > "${validation}.tmp"
  mv "${validation}.tmp" "${validation}"
  touch "${marker}"
  completed=$((completed + 1))
  write_status running "${completed}" validated
done

ACTIVE_VARIANT=aggregate
"${PYTHON}" - "${RUN_ROOT}" "${PURE_ROOT}" "${FACTORIAL_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

output_root, pure_root, factorial_root = map(Path, sys.argv[1:])
roots = {
    "mask_logits": pure_root,
    "mask_flow": factorial_root / "mask_flow",
    "mask_retention": factorial_root / "mask_retention",
    "mask_flow_retention": factorial_root / "mask_flow_retention",
}
payload = {
    "status": "completed",
    "design": "flow x teacher-retention 2x2 by generation adaptation route",
    "variants": {},
}
metrics = (
    "geneval_image_accuracy",
    "geneval_prompt_pass",
    "self_cycle_accuracy",
    "self_cycle_mean_token_f1",
    "self_cycle_prompt_hard_score",
    "self_cycle_prompt_soft_score",
)


def compact(comparison):
    evaluation = comparison["evaluation"]
    return {
        "geneval_image_accuracy": evaluation["geneval"]["image_accuracy"],
        "geneval_prompt_pass": evaluation["geneval"]["prompt_pass"],
        "self_cycle_accuracy": evaluation["self_cycle"]["accuracy"],
        "self_cycle_mean_token_f1": evaluation["self_cycle"]["mean_token_f1"],
        "self_cycle_prompt_hard_score": evaluation["self_cycle"]["prompt_hard_score"],
        "self_cycle_prompt_soft_score": evaluation["self_cycle"]["prompt_soft_score"],
    }


for name, root in roots.items():
    payload["variants"][name] = {"pair_root": str(root.resolve()), "routes": {}}
    for route, pattern in (
        ("g_adapter", "zero_init*_comparison.json"),
        ("lora", "*_lora_comparison.json"),
    ):
        matches = sorted((root / route).glob(pattern))
        if len(matches) != 1:
            raise RuntimeError(
                f"expected one {route} comparison for {name}, found {matches}"
            )
        comparison = json.loads(matches[0].read_text(encoding="utf-8"))
        if comparison.get("status") != "completed":
            raise RuntimeError(f"incomplete {route} comparison: {name}")
        payload["variants"][name]["routes"][route] = {
            "comparison_path": str(matches[0].resolve()),
            "metrics": compact(comparison),
        }

payload["effects"] = {}
for route in ("g_adapter", "lora"):
    values = {
        name: payload["variants"][name]["routes"][route]["metrics"]
        for name in roots
    }
    payload["effects"][route] = {}
    for metric in metrics:
        y00 = values["mask_logits"][metric]
        y10 = values["mask_flow"][metric]
        y01 = values["mask_retention"][metric]
        y11 = values["mask_flow_retention"][metric]
        payload["effects"][route][metric] = {
            "flow_main_effect": ((y10 - y00) + (y11 - y01)) / 2.0,
            "retention_main_effect": ((y01 - y00) + (y11 - y10)) / 2.0,
            "flow_x_retention_interaction": y11 - y10 - y01 + y00,
        }

payload["lora_minus_g_adapter"] = {}
for name in roots:
    g_metrics = payload["variants"][name]["routes"]["g_adapter"]["metrics"]
    lora_metrics = payload["variants"][name]["routes"]["lora"]["metrics"]
    payload["lora_minus_g_adapter"][name] = {
        metric: lora_metrics[metric] - g_metrics[metric] for metric in metrics
    }
temporary = output_root / "lora_factorial_summary.json.tmp"
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
temporary.replace(output_root / "lora_factorial_summary.json")
PY

ACTIVE_VARIANT=none
write_status completed 4 all_lora_evaluations_validated
touch "${RUN_ROOT}/LORA_FACTORIAL_EVAL_COMPLETED"
trap - EXIT
echo "Factorial LoRA evaluation completed: ${RUN_ROOT}"
