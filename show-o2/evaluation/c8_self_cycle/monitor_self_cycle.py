#!/usr/bin/env python3
"""Display explicit generation, QA, and overall self-cycle progress."""

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
    latest_file = DEFAULT_ROOT / "latest_baseline.txt"
    if latest_file.is_file():
        path = Path(latest_file.read_text(encoding="utf-8").strip())
        if path.is_dir():
            return path
    candidates = [
        path
        for path in DEFAULT_ROOT.glob("c8-self-cycle-baseline-*")
        if path.is_dir()
    ]
    if not candidates:
        raise FileNotFoundError("no C8 self-cycle baseline run found")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def duration(value: object) -> str:
    if value is None:
        return "unknown"
    seconds = max(0, int(float(value)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def bar(percent: float, width: int = 24) -> str:
    filled = min(width, max(0, round(width * percent / 100.0)))
    return "[" + "#" * filled + "." * (width - filled) + "]"


def render(run_dir: Path) -> tuple[str, str]:
    status_path = run_dir / "status.json"
    if not status_path.is_file():
        return f"run={run_dir.name} status=initializing overall=0.00%", "initializing"
    data = json.loads(status_path.read_text(encoding="utf-8"))
    overall = float(data.get("percent", 0.0))
    generation = float(data.get("generation_percent", 0.0))
    qa = float(data.get("qa_percent", 0.0))
    completed_images = int(data.get("completed_images", 0))
    total_images = int(data.get("total_images", 0))
    completed_questions = int(data.get("completed_questions", 0))
    total_questions = int(data.get("total_questions", 0))
    state = str(data.get("status", "unknown"))
    phase = str(data.get("phase", "unknown"))
    sampler_step = int(data.get("sampler_step", 0))
    sampler_steps = int(data.get("sampler_steps", 0))
    line = (
        f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} "
        f"run={run_dir.name} status={state} phase={phase}\n"
        f"overall    {bar(overall)} {overall:6.2f}% "
        f"eta={duration(data.get('eta_seconds'))}\n"
        f"generation {bar(generation)} {generation:6.2f}% "
        f"images={completed_images}/{total_images} "
        f"active_step={sampler_step}/{sampler_steps}\n"
        f"structured {bar(qa)} {qa:6.2f}% "
        f"questions={completed_questions}/{total_questions}"
    )
    if data.get("current_answer_mode"):
        line += f" mode={data['current_answer_mode']}"
    if data.get("error"):
        line += f"\nerror={data['error']}"
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
