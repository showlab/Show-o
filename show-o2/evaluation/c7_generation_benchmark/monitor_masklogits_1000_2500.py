#!/usr/bin/env python3
"""Monitor the C7 original mask-logits 1000/2500 generation and scoring queue."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def bar(percent: float, width: int = 36) -> str:
    filled = min(width, max(0, round(width * percent / 100.0)))
    return "[" + "#" * filled + "." * (width - filled) + "]"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    args = parser.parse_args()
    root = args.run_root.resolve()
    queue_path = root / "queue.status"
    queue = queue_path.read_text(encoding="utf-8").strip() if queue_path.exists() else "pending"
    print(f"Run:   {root.name}")
    print(f"Queue: {queue}")
    generation_status = root / "generation" / "status.json"
    if generation_status.exists():
        data = json.loads(generation_status.read_text(encoding="utf-8"))
        completed = int(data.get("completed_images", 0))
        total = int(data.get("total_images", 576))
        percent = 100.0 * completed / total if total else 0.0
        print(
            f"Generation: {completed}/{total} {bar(percent)} {percent:6.2f}% "
            f"variant={data.get('variant')} phase={data.get('phase')}"
        )
    reward_status = root / "image_reward" / "status.json"
    if reward_status.exists():
        data = json.loads(reward_status.read_text(encoding="utf-8"))
        completed = int(data.get("completed_images", data.get("total_images", 0) if data.get("status") == "completed" else 0))
        total = int(data.get("total_images", 1728))
        percent = 100.0 * completed / total if total else 0.0
        print(
            f"ImageReward: {completed}/{total} {bar(percent)} {percent:6.2f}% "
            f"status={data.get('status')}"
        )
    clip_root = root / "clip"
    if clip_root.exists():
        completed = sum((path / "summary.json").exists() for path in clip_root.iterdir() if path.is_dir())
        print(f"CLIP variants completed: {completed}/6")
    if (root / "BENCHMARK_COMPLETED").exists():
        print("Status: completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
