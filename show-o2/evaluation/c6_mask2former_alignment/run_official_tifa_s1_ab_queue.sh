#!/usr/bin/env bash
set -Eeuo pipefail
export HF_HOME=/root/autodl-tmp/cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
ROOT=/root/autodl-tmp/Show-o/show-o2
WORKSPACE=/root/autodl-tmp
PYTHON=${PYTHON:-${WORKSPACE}/venvs/showo2-eval/bin/python}
RUN_ID=${RUN_ID:-official-tifa-s1-ab-20260819T000000Z}
RUN_ROOT=${RUN_ROOT:-${WORKSPACE}/runs/showo2_tifa_official/${RUN_ID}}
DATA_MANIFEST=${DATA_MANIFEST:-${WORKSPACE}/data/showo2_self_cycle/tifa_v1/official_trainval_interleaved.manifest.json}
TIFA_ROOT=${TIFA_ROOT:-${WORKSPACE}/data/showo2_self_cycle/tifa_v1}
LOCK_LORA=${LOCK_LORA:-${TIFA_ROOT}/official_tifa_lora_starts.lock.json}
LOCK_G=${LOCK_G:-${TIFA_ROOT}/official_tifa_g_adapter_starts.lock.json}
STEPS=100,200,500,750,1000,1500,2000,3865
BENCH_STEPS=100,200,500,1000,2000,3865
mkdir -p "${RUN_ROOT}"
write_stage() {
  local stage=$1 state=$2 detail=${3:-}
  "${PYTHON}" - "${RUN_ROOT}/queue_status.json" "${stage}" "${state}" "${detail}" <<'PY'
import json,sys,time
from pathlib import Path
p=Path(sys.argv[1]); old=json.loads(p.read_text()) if p.is_file() else {}
payload={**old,'stage':sys.argv[2],'stage_status':sys.argv[3],'detail':sys.argv[4],'updated_at':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'run_root':str(p.parent.resolve()),'train_steps':[100,200,500,750,1000,1500,2000,3865],'benchmark_steps':[100,200,500,1000,2000,3865],'official_test_excluded_from_training':True}
p.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
PY
}
run_train() {
  local name=$1 lock=$2 dir=$3
  if [[ -f "${dir}/QUEUE_COMPLETED" ]]; then return; fi
  write_stage "${name}_training" running "starting two-route ${name} training"
  "${PYTHON}" -u "${ROOT}/evaluation/c6_mask2former_alignment/run_tifa12_self_cycle_queue.py" \
    --run-dir "${dir}" --lock-file "${lock}" --data-manifest "${DATA_MANIFEST}" \
    --max-steps 3865 --save-steps "${STEPS}" --seed 20260819 --learning-rate 1e-5 \
    2>&1 | tee -a "${dir}/queue.log"
  write_stage "${name}_training" completed "two-route ${name} training completed"
}
run_bench() {
  local name route train_dir bench_dir
  name=$1; route=$2; train_dir=$3; bench_dir=$4
  if [[ -f "${bench_dir}/BENCHMARK_QUEUE_COMPLETED" ]]; then return; fi
  write_stage "${name}_benchmark" running "Geneval+VQAScore+VSR-single-seed+official-TIFA-test"
  "${PYTHON}" -u "${ROOT}/evaluation/c6_mask2former_alignment/run_official_tifa_benchmark_queue.py" \
    --training-root "${train_dir}" --run-root "${bench_dir}" --route "${route}" \
    --steps "${BENCH_STEPS}" --eval-seed 20260810 \
    2>&1 | tee -a "${bench_dir}/queue.log"
  write_stage "${name}_benchmark" completed "${name} benchmarks completed"
}
write_stage preflight running "validating official TIFA manifest and locked starts"
[[ -f "${DATA_MANIFEST}" && -f "${LOCK_LORA}" && -f "${LOCK_G}" ]] || { write_stage preflight failed "missing data manifest or lock"; exit 1; }
write_stage preflight completed "official test excluded; four-source interleaved trainval ready"
run_train lora "${LOCK_LORA}" "${RUN_ROOT}/lora_training"
run_bench lora lora "${RUN_ROOT}/lora_training" "${RUN_ROOT}/lora_benchmark"
run_train g_adapter "${LOCK_G}" "${RUN_ROOT}/g_adapter_training"
run_bench g_adapter g_adapter "${RUN_ROOT}/g_adapter_training" "${RUN_ROOT}/g_adapter_benchmark"
write_stage completed completed "LoRA then G-adapter training and benchmark queues completed"
touch "${RUN_ROOT}/QUEUE_COMPLETED"
