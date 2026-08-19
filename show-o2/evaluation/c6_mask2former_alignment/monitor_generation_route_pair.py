#!/usr/bin/env python3
"""Show combined percent/ETA for the paired G-adapter and LoRA queue."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROUTES = ("g_adapter", "lora")
EVAL_GENEVAL_WEIGHT = 420.0
EVAL_SELF_CYCLE_WEIGHT = 3758.0


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def progress(status: dict[str, Any], completed_key: str, total_key: str) -> float:
    if status.get("status") == "completed":
        return 1.0
    completed = float(status.get(completed_key, 0.0) or 0.0)
    total = float(status.get(total_key, 0.0) or 0.0)
    return min(1.0, max(0.0, completed / total)) if total > 0 else 0.0


def direct_eval_progress(
    run_root: Path, route: str = "g_adapter"
) -> tuple[str, float, float | None]:
    root = run_root / route / "evaluation"
    geneval_candidates = sorted(root.glob("step-*/geneval_position72/status.json"))
    geneval_status = read_json(geneval_candidates[-1]) if geneval_candidates else {}
    self_candidates = sorted(root.glob("step-*/self_cycle/seed-*/**/status.json"))
    self_status = read_json(self_candidates[-1]) if self_candidates else {}
    geneval = progress(geneval_status, "completed_images", "total_images")
    self_cycle = progress(self_status, "completed_questions", "total_questions")
    if self_status and self_status.get("phase") == "generation":
        self_cycle = progress(self_status, "completed_images", "total_images") * 0.90
    elif self_status and self_status.get("phase") in {"qa", "scoring"}:
        self_cycle = 0.90 + 0.10 * progress(
            self_status, "completed_questions", "total_questions"
        )
    weighted = (
        EVAL_GENEVAL_WEIGHT * geneval + EVAL_SELF_CYCLE_WEIGHT * self_cycle
    ) / (EVAL_GENEVAL_WEIGHT + EVAL_SELF_CYCLE_WEIGHT)
    marker = run_root / route / "ZERO_INIT_EVAL_COMPLETED"
    if marker.is_file():
        return "completed", 1.0, 0.0
    state = "running" if geneval_status or self_status else "waiting"
    eta_values = [
        float(value["eta_seconds"])
        for value in (geneval_status, self_status)
        if value.get("eta_seconds") is not None
    ]
    return state, weighted, sum(eta_values) if eta_values else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    args = parser.parse_args()
    config = read_json(args.run_root / "queue_config.json")
    configured_total = max(1, int(config.get("max_steps", 5000)))
    rows = []
    for route in ROUTES:
        status = read_json(args.run_root / route / "status.json")
        total = max(1, int(status.get("total_steps", configured_total)))
        step = min(total, max(0, int(status.get("step", 0))))
        state = str(status.get("status", "waiting"))
        if state == "completed":
            step = total
        rows.append((route, state, step, total, status))
    eval_enabled = bool(config.get("eval_g_adapter", False))
    if eval_enabled:
        eval_state, eval_fraction, eval_eta = direct_eval_progress(args.run_root)
    else:
        eval_state, eval_fraction, eval_eta = "disabled", 0.0, 0.0
    route_work = sum(step / total for _, _, step, total, _ in rows)
    total_units = len(rows) + int(eval_enabled)
    work = route_work + (eval_fraction if eval_enabled else 0.0)
    percent = 100.0 * work / total_units
    completed_units = sum(state == "completed" for _, state, *_ in rows)
    completed_units += int(eval_enabled and eval_state == "completed")

    reference_total = next((
        float(status["elapsed_seconds"]) / step * total
        for _, state, step, total, status in rows
        if state == "running" and step > 0 and status.get("elapsed_seconds")
    ), None)
    remaining_seconds = []
    eta_known = True
    for _, state, _, _, status in rows:
        if state == "completed":
            remaining_seconds.append(0.0)
        elif status.get("eta_seconds") is not None:
            remaining_seconds.append(float(status["eta_seconds"]))
        elif state == "waiting" and reference_total is not None:
            remaining_seconds.append(reference_total)
        else:
            eta_known = False
    if eval_enabled and eval_state != "completed":
        if eval_eta is None:
            eta_known = False
        else:
            remaining_seconds.append(float(eval_eta))
    total_eta = sum(remaining_seconds) if eta_known else None
    total_eta_text = f"{total_eta / 3600:.2f}h" if total_eta is not None else "n/a"
    print(
        f"total [{int(percent // 4) * '#':<25}] {percent:6.2f}% "
        f"completed={completed_units}/{total_units} eta~={total_eta_text}"
    )
    for route, state, step, total, status in rows:
        route_percent = 100.0 * step / total
        eta = status.get("eta_seconds")
        eta_text = f"{float(eta) / 3600:.2f}h" if eta is not None else "n/a"
        adaptation = status.get("adaptation", {})
        parameters = adaptation.get("trainable_parameters", "pending")
        print(
            f"{route:<10} status={state:<12} step={step:4d}/{total:<4d} "
            f"progress={route_percent:6.2f}% eta={eta_text:<8} params={parameters}"
        )
        if route == "g_adapter" and eval_enabled:
            eval_eta_text = f"{eval_eta / 3600:.2f}h" if eval_eta is not None else "n/a"
            print(
                f"zero_init_eval status={eval_state:<12} "
                f"progress={100.0 * eval_fraction:6.2f}% eta={eval_eta_text} "
                "benchmarks=GenEval+Self-cycle(seed=20260810)"
            )
    queue = args.run_root / "queue.status"
    if queue.is_file():
        print(queue.read_text(encoding="utf-8").strip())
    if config:
        print(
            f"config experiment={config.get('experiment')} manifest={config.get('manifest')}"
        )
        print(
            f"save_steps={','.join(str(value) for value in config.get('save_steps', []))}"
        )
        print(
            f"training_seed={config.get('training_seed')} eval_enabled={eval_enabled}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
