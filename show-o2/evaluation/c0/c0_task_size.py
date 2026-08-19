#!/usr/bin/env python3
"""Resolve one lmms-eval task and print its authoritative evaluation size."""

from __future__ import annotations

import argparse
import json

import lmms_eval.tasks
from lmms_eval.tasks import TaskManager


MARKER = "SHOWO2_TASK_SIZE_JSON="


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--include-path", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manager = TaskManager("INFO", include_path=args.include_path)
    matched = manager.match_tasks([args.task])
    if args.task not in matched:
        raise ValueError(f"Task is not registered: {args.task}")
    task_dict = lmms_eval.tasks.get_task_dict([args.task], manager)
    sizes: dict[str, int] = {}
    for name, task_value in task_dict.items():
        task = task_value[1] if isinstance(task_value, tuple) else task_value
        if task is None or not hasattr(task, "eval_docs"):
            continue
        sizes[name] = len(task.eval_docs)
    if not sizes:
        raise ValueError(f"No evaluation documents resolved for task: {args.task}")
    payload = {
        "task": args.task,
        "size": sum(sizes.values()),
        "subtasks": sizes,
    }
    print(MARKER + json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
