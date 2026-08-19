#!/usr/bin/env python3
"""Create and update the machine-readable C0 benchmark status."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--state", required=True)
    result.add_argument("--run-id")
    result.add_argument("--status")
    result.add_argument("--task")
    result.add_argument("--task-index", type=int)
    result.add_argument("--task-total", type=int)
    result.add_argument("--expected-samples", type=int)
    result.add_argument("--completed-samples", type=int)
    result.add_argument("--completed-before", type=int)
    result.add_argument("--overall-total", type=int)
    result.add_argument("--pid", type=int)
    result.add_argument("--log")
    result.add_argument("--error")
    result.add_argument("--tasks-json")
    result.add_argument("--finish", action="store_true")
    return result


def read(path: Path) -> dict:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(state, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def main() -> None:
    args = parser().parse_args()
    path = Path(args.state)
    state = read(path)
    now = int(time.time())
    if not state:
        state["started_at"] = now
    if args.task is not None and args.task != state.get("current_task"):
        state["task_completed_offset"] = 0
    mapping = {
        "run_id": args.run_id,
        "status": args.status,
        "current_task": args.task,
        "task_index": args.task_index,
        "task_total": args.task_total,
        "expected_samples": args.expected_samples,
        "completed_samples": args.completed_samples,
        "completed_before": args.completed_before,
        "overall_total": args.overall_total,
        "pid": args.pid,
        "current_log": args.log,
        "error": args.error,
    }
    for key, value in mapping.items():
        if value is not None:
            state[key] = value
    if args.status in {"resolving_data", "preflight", "running"}:
        state.pop("finished_at", None)
        if args.error is None:
            state["error"] = ""
    if args.tasks_json:
        state["tasks"] = json.loads(args.tasks_json)
    if args.completed_samples is not None or args.expected_samples is not None:
        completed = int(state.get("completed_samples", 0))
        expected = int(state.get("expected_samples", 0))
        state["task_percent"] = round(100.0 * completed / expected, 2) if expected else 0.0
    completed_before = int(state.get("completed_before", 0))
    completed = int(state.get("completed_samples", 0))
    overall_total = int(state.get("overall_total", 0))
    state["overall_percent"] = (
        round(100.0 * (completed_before + completed) / overall_total, 2)
        if overall_total
        else 0.0
    )
    if args.finish:
        state["finished_at"] = now
    if state.get("status") == "completed":
        state["overall_percent"] = 100.0
    state["updated_at"] = now
    write(path, state)


if __name__ == "__main__":
    main()
