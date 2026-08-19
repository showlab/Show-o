#!/usr/bin/env python3
"""Print percentage and ETA for the formal four-route TIFA-12 queue."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=5.0)
    return parser.parse_args()


def duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0 or seconds == float("inf"):
        return "unknown"
    value = int(seconds)
    hours, remainder = divmod(value, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def render(run_dir: Path) -> bool:
    queue_path = run_dir / "queue_status.json"
    if not queue_path.is_file():
        print(f"queue status pending: {queue_path}")
        return False
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    tasks = queue["tasks"]
    maximum = int(queue["max_steps_per_route"])
    total_units = maximum * len(tasks)
    completed_units = 0
    current_eta: float | None = None
    observed_rate: float | None = None
    rows = []
    for index, task in enumerate(tasks, 1):
        label = task["label"]
        path = run_dir / label / "status.json"
        detail = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        state = str(detail.get("status", task["status"]))
        step = int(detail.get("step", maximum if state == "completed" else 0))
        if state == "completed":
            step = maximum
        completed_units += min(step, maximum)
        percent = 100.0 * step / maximum
        eta = detail.get("eta_seconds")
        if state == "running":
            current_eta = float(eta) if eta is not None else None
            rate = detail.get("throughput_steps_per_second")
            observed_rate = float(rate) if rate else None
        marker = {"completed": "DONE", "running": "RUN ", "failed": "FAIL"}.get(state, "WAIT")
        rows.append(
            f"[{marker}] {index}/4 {label:22s} step={step:5d}/{maximum} "
            f"progress={percent:7.3f}% eta={duration(float(eta)) if eta is not None else 'unknown'}"
        )
    remaining_after_current = sum(
        maximum for task in tasks
        if task["status"] == "waiting"
    )
    overall_eta = current_eta
    if overall_eta is not None and observed_rate:
        overall_eta += remaining_after_current / observed_rate
    overall = 100.0 * completed_units / total_units
    print(time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    print(f"run={run_dir.resolve()}")
    print(
        f"queue_status={queue['status']} current={queue.get('current')} "
        f"overall={overall:.3f}% eta~={duration(overall_eta)}"
    )
    print("\n".join(rows))
    return queue["status"] in ("completed", "failed")


def main() -> int:
    args = parse_args()
    while True:
        terminal = render(args.run_dir)
        if not args.watch or terminal:
            return 0
        print(flush=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
