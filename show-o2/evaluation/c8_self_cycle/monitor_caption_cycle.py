#!/usr/bin/env python3
"""Display explicit progress for the strict Show-o2 caption cycle."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_ROOT = Path("/root/autodl-tmp/runs/showo2_c8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", nargs="?", type=Path)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=5.0)
    return parser.parse_args()


def latest_run() -> Path:
    marker = DEFAULT_ROOT / "latest_caption_cycle.txt"
    if marker.is_file():
        value = Path(marker.read_text(encoding="utf-8").strip())
        if value.is_dir():
            return value
    candidates = [
        path
        for path in DEFAULT_ROOT.glob("c8-caption-cycle-baseline-*")
        if path.is_dir()
    ]
    if not candidates:
        raise FileNotFoundError("no C8 caption-cycle run found")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def duration(value: object) -> str:
    if value is None:
        return "unknown"
    seconds = max(0, int(float(value)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def bar(percent: float, width: int = 28) -> str:
    filled = min(width, max(0, round(width * percent / 100.0)))
    return "[" + "#" * filled + "." * (width - filled) + "]"


def render(run_dir: Path) -> tuple[str, str]:
    path = run_dir / "status.json"
    if not path.is_file():
        return f"run={run_dir.name} status=initializing percent=0.00%", "initializing"
    data = json.loads(path.read_text(encoding="utf-8"))
    state = str(data.get("status", "unknown"))
    percent = float(data.get("percent", 0.0))
    completed = int(data.get("completed_captions", 0))
    total = int(data.get("total_captions", 0))
    line = (
        f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} "
        f"run={run_dir.name} status={state} phase={data.get('phase', 'unknown')}\n"
        f"caption-cycle {bar(percent)} {percent:6.2f}% "
        f"captions={completed}/{total} eta={duration(data.get('eta_seconds'))}"
    )
    if data.get("current_record_index") is not None:
        line += f" record={data['current_record_index']}"
    if data.get("error"):
        line += f"\nerror={data['error']}"
    return line, state


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve() if args.run_dir else latest_run()
    while True:
        line, state = render(run_dir)
        print(line, flush=True)
        if not args.watch or state in {"completed", "failed"}:
            return 0 if state != "failed" else 1
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
