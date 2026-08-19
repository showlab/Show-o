#!/usr/bin/env python3
"""Render one snapshot of C0 benchmark progress without extra dependencies."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def bar(percent: float, width: int = 40) -> str:
    percent = max(0.0, min(100.0, percent))
    filled = int(round(width * percent / 100.0))
    return "[" + "#" * filled + "." * (width - filled) + f"] {percent:6.2f}%"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", required=True)
    args = parser.parse_args()
    path = Path(args.state)
    if not path.is_file():
        raise SystemExit(f"Status file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        state = json.load(handle)
    updated = int(state.get("updated_at", 0))
    age = max(0, int(time.time()) - updated) if updated else -1
    task_index = int(state.get("task_index", 0))
    task_total = int(state.get("task_total", 0))
    completed = int(state.get("completed_samples", 0))
    expected = int(state.get("expected_samples", 0))
    overall_total = int(state.get("overall_total", 0))
    if state.get("status") == "completed":
        overall_done = overall_total
    else:
        overall_done = int(state.get("completed_before", 0)) + completed
    print(f"Run:       {state.get('run_id', 'unknown')}")
    print(f"Status:    {state.get('status', 'unknown')}  PID={state.get('pid', 'unknown')}  update_age={age}s")
    print(f"Task:      {task_index}/{task_total} {state.get('current_task', '-')}")
    print(f"Task rows: {completed}/{expected}")
    print("Task:      " + bar(float(state.get("task_percent", 0.0))))
    print(f"All rows:  {overall_done}/{overall_total}")
    print("Overall:   " + bar(float(state.get("overall_percent", 0.0))))
    print(f"Log:       {state.get('current_log', '-')}")
    error = state.get("error")
    if error:
        print(f"Error:     {error}")


if __name__ == "__main__":
    main()
