#!/usr/bin/env python3
"""Monitor the Mask/Flow/teacher-retention 2x2 factorial queue."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from monitor_factorial_lora_eval import evaluation_progress as lora_eval_progress
from monitor_generation_route_pair import direct_eval_progress


VARIANTS = ("mask_flow", "mask_teacher_replay", "mask_flow_teacher_replay")
ROUTES = ("g_adapter", "lora")


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--pure-root", type=Path, required=True)
    parser.add_argument("--lora-eval-root", type=Path)
    args = parser.parse_args()

    specs = [("mask_logits", args.pure_root)] + [
        (name, args.run_root / name) for name in VARIANTS
    ]
    rows = []
    eval_rows = []
    lora_eval_rows = []
    for name, root in specs:
        config = read_json(root / "queue_config.json")
        total = max(1, int(config.get("max_steps", 17100)))
        for route in ROUTES:
            status = read_json(root / route / "status.json")
            step = max(0, min(total, int(status.get("step", 0))))
            state = str(status.get("status", "waiting"))
            if state == "completed":
                step = total
            rows.append((name, route, state, step, total, status))
        eval_state, eval_fraction, eval_eta = direct_eval_progress(root)
        eval_rows.append((name, eval_state, eval_fraction, eval_eta))
        if args.lora_eval_root is not None:
            state, fraction, eta = lora_eval_progress(root)
            lora_eval_rows.append((name, state, fraction, eta))

    work = sum(step / total for _, _, _, step, total, _ in rows)
    work += sum(fraction for _, _, fraction, _ in eval_rows)
    work += sum(fraction for _, _, fraction, _ in lora_eval_rows)
    stages = len(rows) + len(eval_rows) + len(lora_eval_rows)
    percent = 100.0 * work / stages
    print(
        f"factorial [{int(percent // 4) * '#':<25}] {percent:6.2f}% "
        "stages_completed="
        f"{sum(state == 'completed' for _, _, state, *_ in rows) + sum(state == 'completed' for _, state, *_ in eval_rows) + sum(state == 'completed' for _, state, *_ in lora_eval_rows)}/{stages}"
    )
    by_variant = {name: [] for name, _ in specs}
    for row in rows:
        by_variant[row[0]].append(row)
    eval_by_variant = {row[0]: row for row in eval_rows}
    lora_eval_by_variant = {row[0]: row for row in lora_eval_rows}
    for name, _ in specs:
        for _, route, state, step, total, status in by_variant[name]:
            eta = status.get("eta_seconds")
            eta_text = f"{float(eta) / 3600:.2f}h" if eta is not None else "n/a"
            print(
                f"{name:<21} {route:<9} status={state:<10} "
                f"step={step:5d}/{total:<5d} progress={100.0 * step / total:6.2f}% "
                f"eta={eta_text}"
            )
            if route == "g_adapter":
                _, eval_state, eval_fraction, eval_eta = eval_by_variant[name]
                eval_eta_text = f"{eval_eta / 3600:.2f}h" if eval_eta is not None else "n/a"
                print(
                    f"{name:<21} {'evaluation':<9} status={eval_state:<10} "
                    f"progress={100.0 * eval_fraction:6.2f}% eta={eval_eta_text}"
                )
            if route == "lora" and name in lora_eval_by_variant:
                _, eval_state, eval_fraction, eval_eta = lora_eval_by_variant[name]
                eval_eta_text = (
                    f"{eval_eta / 3600:.2f}h" if eval_eta is not None else "n/a"
                )
                print(
                    f"{name:<21} {'lora_eval':<9} status={eval_state:<10} "
                    f"progress={100.0 * eval_fraction:6.2f}% eta={eval_eta_text}"
                )
    if (args.run_root / "queue.status").is_file():
        print((args.run_root / "queue.status").read_text(encoding="utf-8").strip())
    if args.lora_eval_root is not None:
        lora_status = args.lora_eval_root / "queue.status"
        if lora_status.is_file():
            print(lora_status.read_text(encoding="utf-8").strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
