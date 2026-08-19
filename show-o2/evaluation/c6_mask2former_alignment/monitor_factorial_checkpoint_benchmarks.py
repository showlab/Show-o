#!/usr/bin/env python3
"""Monitor eight factorial training routes and forty checkpoint benchmarks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


VARIANTS = (
    "mask_logits",
    "mask_flow",
    "mask_teacher_replay",
    "mask_flow_teacher_replay",
)
ROUTES = ("g_adapter", "lora")
GENEVAL_WEIGHT = 420.0
SELF_CYCLE_COMPACT96_WEIGHT = 3758.0 * 96.0 / 689.0


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def read_status(path: Path) -> dict[str, str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}
    return {
        key: value
        for line in lines
        if "=" in line
        for key, value in (line.split("=", 1),)
    }


def fraction(status: dict[str, Any], completed: str, total: str) -> float:
    if status.get("status") == "completed":
        return 1.0
    denominator = float(status.get(total, 0) or 0)
    numerator = float(status.get(completed, 0) or 0)
    return min(1.0, max(0.0, numerator / denominator)) if denominator else 0.0


def benchmark_progress(eval_root: Path) -> tuple[str, float, float | None]:
    if (eval_root / "QUEUE_COMPLETED").is_file():
        return "completed", 1.0, 0.0
    geneval = read_json(eval_root / "geneval_position72/status.json")
    self_paths = sorted(eval_root.glob("self_cycle/seed-*/**/status.json"))
    self_cycle = read_json(self_paths[-1]) if self_paths else {}
    geneval_fraction = fraction(geneval, "completed_images", "total_images")
    if self_cycle.get("phase") == "generation":
        self_fraction = 0.90 * fraction(
            self_cycle, "completed_images", "total_images"
        )
    elif self_cycle.get("phase") in {"qa", "scoring"}:
        self_fraction = 0.90 + 0.10 * fraction(
            self_cycle, "completed_questions", "total_questions"
        )
    else:
        self_fraction = fraction(
            self_cycle, "completed_questions", "total_questions"
        )
    progress = (
        GENEVAL_WEIGHT * geneval_fraction
        + SELF_CYCLE_COMPACT96_WEIGHT * self_fraction
    ) / (GENEVAL_WEIGHT + SELF_CYCLE_COMPACT96_WEIGHT)
    state = "running" if geneval or self_cycle else "waiting"
    eta_values = [
        float(status["eta_seconds"])
        for status in (geneval, self_cycle)
        if status.get("eta_seconds") is not None
    ]
    return state, progress, sum(eta_values) if eta_values else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("factorial_root", type=Path)
    parser.add_argument("--pure-root", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument(
        "--benchmark-steps", default="100,500,1000,5000,17100"
    )
    args = parser.parse_args()
    steps = tuple(int(value) for value in args.benchmark_steps.split(",") if value)
    roots = {
        "mask_logits": args.pure_root,
        **{
            name: args.factorial_root / name
            for name in VARIANTS
            if name != "mask_logits"
        },
    }
    training_rows = []
    for name, root in roots.items():
        config = read_json(root / "queue_config.json")
        configured_total = max(1, int(config.get("max_steps", 17100)))
        for route in ROUTES:
            status = read_json(root / route / "status.json")
            total = max(1, int(status.get("total_steps", configured_total)))
            step = min(total, max(0, int(status.get("step", 0))))
            state = str(status.get("status", "waiting"))
            if state == "completed":
                step = total
            training_rows.append((name, route, state, step, total, status))

    benchmark_rows = []
    for step in steps:
        for name in VARIANTS:
            for route in ROUTES:
                eval_root = (
                    args.benchmark_root
                    / name
                    / route
                    / f"step-{step:06d}"
                )
                state, progress, eta = benchmark_progress(eval_root)
                benchmark_rows.append((step, name, route, state, progress, eta))

    training_work = sum(
        step / total for _, _, _, step, total, _ in training_rows
    )
    benchmark_work = sum(row[4] for row in benchmark_rows)
    stages = len(training_rows) + len(benchmark_rows)
    overall = (training_work + benchmark_work) / stages
    completed = sum(row[2] == "completed" for row in training_rows) + sum(
        row[3] == "completed" for row in benchmark_rows
    )
    print(
        f"factorial [{int(overall * 28) * '#':<28}] {100.0 * overall:6.2f}% "
        f"stages_completed={completed}/{stages}"
    )
    print("training:")
    for name, route, state, step, total, status in training_rows:
        eta = status.get("eta_seconds")
        eta_text = f"{float(eta) / 3600:.2f}h" if eta is not None else "n/a"
        print(
            f"  {name:<25} {route:<9} status={state:<9} "
            f"step={step:5d}/{total:<5d} progress={100.0 * step / total:6.2f}% "
            f"eta={eta_text}"
        )
    print("benchmarks (GenEval Position-72 + Self-cycle compact96):")
    for checkpoint_step in steps:
        rows = [row for row in benchmark_rows if row[0] == checkpoint_step]
        checkpoint_progress = sum(row[4] for row in rows) / len(rows)
        checkpoint_completed = sum(row[3] == "completed" for row in rows)
        active = next((row for row in rows if row[3] == "running"), None)
        active_text = "none"
        if active is not None:
            eta_text = f"{active[5] / 3600:.2f}h" if active[5] is not None else "n/a"
            active_text = (
                f"{active[1]}/{active[2]} "
                f"{100.0 * active[4]:.2f}% eta={eta_text}"
            )
        print(
            f"  step={checkpoint_step:<5d} completed={checkpoint_completed}/8 "
            f"progress={100.0 * checkpoint_progress:6.2f}% active={active_text}"
        )
    factorial_status = read_status(args.factorial_root / "queue.status")
    benchmark_status = read_status(args.benchmark_root / "queue.status")
    if factorial_status:
        print(
            "training_queue "
            f"status={factorial_status.get('status', 'unknown')} "
            f"active={factorial_status.get('active_variant', 'unknown')} "
            f"detail={factorial_status.get('detail', 'none')}"
        )
    if benchmark_status:
        print(
            "benchmark_queue "
            f"status={benchmark_status.get('status', 'unknown')} "
            f"completed={benchmark_status.get('completed', '0/40')} "
            f"active={benchmark_status.get('active', 'unknown')} "
            f"detail={benchmark_status.get('detail', 'none')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
