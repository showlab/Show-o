#!/usr/bin/env bash
set -Eeuo pipefail
BENCH=/root/autodl-tmp/runs/showo2_tifa_official/official-tifa-s1-ab-20260819T000000Z/lora_benchmark
ROOT=/root/autodl-tmp/runs/showo2_tifa_official/official-tifa-s1-ab-20260819T000000Z
while [[ ! -f "${BENCH}/BENCHMARK_QUEUE_COMPLETED" ]]; do sleep 10; done
/root/autodl-tmp/venvs/showo2-eval/bin/python - <<'PY2'
import json,time
from pathlib import Path
p=Path("/root/autodl-tmp/runs/showo2_tifa_official/official-tifa-s1-ab-20260819T000000Z/queue_status.json")
d=json.loads(p.read_text()) if p.is_file() else {}
d.update({"stage":"mixed_replay_lora_training","stage_status":"waiting","detail":"TIFA CE + Mask replay + low-weight Flow replay + COCO frozen-U teacher consistency; G-adapter discarded","updated_at":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),"next_queue":"/root/autodl-tmp/runs/showo2_tifa_mixed_replay/official-tifa-mixed-s1-lora-20260819T000000Z"})
p.write_text(json.dumps(d,indent=2,sort_keys=True)+"\n")
PY2
kill -KILL 423616 2>/dev/null || true
echo "[switch-guard] benchmark completed; obsolete G-adapter stage terminated; mixed LoRA queue is active/waiting"
exec bash
