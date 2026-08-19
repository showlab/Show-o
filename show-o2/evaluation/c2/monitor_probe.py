#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    return parser.parse_args()


def bar(percent: float, width: int = 40) -> str:
    filled = min(width, max(0, round(width * percent / 100.0)))
    return "[" + "#" * filled + "." * (width - filled) + "]"


def duration(seconds: float | None) -> str:
    if seconds is None:
        return "N/A"
    minutes, seconds = divmod(max(0, int(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:d}h {minutes:02d}m {seconds:02d}s" if hours else f"{minutes:d}m {seconds:02d}s"


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
    elapsed = max(0, int(time.time()) - int(status.get("started_at", 0)))
    throughput = completed / elapsed if completed > 0 and elapsed > 0 else None
    eta = (total - completed) / throughput if throughput is not None else None
    last_forward = status.get("last_runtime_seconds")
    pid = int(status.get("pid", 0) or 0)
    process_alive = pid > 0 and Path(f"/proc/{pid}").exists()
    print(time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    print(f"Run:       {args.run_dir.name}")
    print(f"Status:    {status.get('status', 'unknown')}  PID={pid or 'N/A'}  alive={process_alive}  update_age={age}s")
    print(f"Record:    {completed}/{total}  {status.get('current_record') or '-'}")
    print(f"Progress:  {bar(percent)} {percent:.2f}%")
    if throughput is not None:
        print(f"Elapsed:   {duration(elapsed)}")
        forward_text = f"  last_forward={float(last_forward):.3f}s" if last_forward is not None else ""
        print(f"Rate:      {throughput:.3f} records/s{forward_text}")
        print(f"ETA:       {duration(eta)}")
    if status.get("error"):
        print(f"Error:     {status['error']}")
    print(f"Log:       {args.run_dir / 'logs/probe.log'}")


if __name__ == "__main__":
    main()
