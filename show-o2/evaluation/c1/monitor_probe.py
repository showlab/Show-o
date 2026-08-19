#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    return parser.parse_args()


def bar(percent: float, width: int = 40) -> str:
    filled = min(width, max(0, round(width * percent / 100.0)))
    return "[" + "#" * filled + "." * (width - filled) + "]"


def main() -> None:
    args = parse_args()
    status_path = args.run_dir / "status.json"
    if not status_path.is_file():
        print(f"Status not available yet: {status_path}")
        return
    status = json.loads(status_path.read_text(encoding="utf-8"))
    completed = int(status.get("completed", 0))
    total = int(status.get("total", 0))
    percent = 100.0 * completed / total if total else 0.0
    age = max(0, int(time.time()) - int(status.get("updated_at", 0)))
    average = None
    samples_path = args.run_dir / "samples.jsonl"
    if samples_path.is_file():
        runtimes = []
        with samples_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    runtimes.append(float(json.loads(line)["runtime_seconds"]))
        if runtimes:
            average = sum(runtimes) / len(runtimes)
    eta = (total - completed) * average if average is not None else None
    print(time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    print(f"Run:      {args.run_dir.name}")
    print(f"Status:   {status.get('status', 'unknown')}  PID={status.get('pid', 'N/A')}  update_age={age}s")
    print(f"Sample:   {completed}/{total}  {status.get('current_sample') or '-'}")
    print(f"Progress: {bar(percent)} {percent:.2f}%")
    if average is not None:
        print(f"Average:  {average:.1f}s/sample")
        print(f"ETA:      {eta / 60.0:.1f} minutes")
    if status.get("error"):
        print(f"Error:    {status['error']}")
    print(f"Log:      {args.run_dir / 'logs/probe.log'}")


if __name__ == "__main__":
    main()
