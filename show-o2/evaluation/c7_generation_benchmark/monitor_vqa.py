#!/usr/bin/env python3
"""Print explicit progress for the latest five-way C7 VQAScore run."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_ROOT = Path("/root/autodl-tmp/runs/showo2_c7")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", nargs="?", type=Path)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=5.0)
    return parser.parse_args()


def latest_run() -> Path:
    candidates = [
        path
        for path in DEFAULT_ROOT.glob("c7-vqascore-fiveway-*")
        if path.is_dir()
    ]
    if not candidates:
        raise FileNotFoundError("no c7-vqascore-fiveway run found")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def format_seconds(value: object) -> str:
    if value is None:
        return "unknown"
    seconds = max(0, int(float(value)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def render(run_dir: Path) -> tuple[str, str]:
    status_path = run_dir / "vqascore" / "status.json"
    if not status_path.is_file():
        queue = run_dir / "queue.status"
        state = queue.read_text().strip() if queue.is_file() else "initializing"
        return f"run={run_dir.name} {state} percent=0.00%", "initializing"
    data = json.loads(status_path.read_text(encoding="utf-8"))
    completed = int(data.get("completed_images", 0))
    total = int(data.get("total_images", 0))
    percent = float(data.get("percent", 100.0 * completed / total if total else 0.0))
    queries = int(data.get("completed_vqa_queries", completed * 2))
    total_queries = int(data.get("total_vqa_queries", total * 2))
    rate = data.get("images_per_second")
    rate_text = "unknown" if rate is None else f"{float(rate):.3f} images/s"
    state = str(data.get("status", "unknown"))
    line = (
        f"{datetime.now(timezone.utc).isoformat()} run={run_dir.name} "
        f"status={state} images={completed}/{total} queries={queries}/{total_queries} "
        f"percent={percent:.2f}% rate={rate_text} "
        f"eta={format_seconds(data.get('eta_seconds'))}"
    )
    if data.get("error"):
        line += f" error={data['error']}"
    return line, state


def main() -> int:
    args = parse_args()
    if args.interval <= 0:
        raise ValueError("interval must be positive")
    run_dir = args.run_dir.resolve() if args.run_dir else latest_run()
    while True:
        line, state = render(run_dir)
        print(line, flush=True)
        if not args.watch or state in {"completed", "failed"}:
            return 0 if state != "failed" else 1
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
