#!/usr/bin/env python3
"""Print a compact progress snapshot for a paired Position A/B run."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

SHOWO2_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = SHOWO2_ROOT.parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", nargs="?", type=Path)
    return parser.parse_args()


def latest_run() -> Path:
    marker = WORKSPACE_ROOT / "runs/showo2_layout_tune/latest_v2.txt"
    if not marker.is_file():
        raise FileNotFoundError(f"Latest-run marker not found: {marker}")
    return Path(marker.read_text(encoding="utf-8").strip())


def progress_bar(value: int, total: int, width: int = 40) -> str:
    ratio = min(max(value / total if total else 0.0, 0.0), 1.0)
    filled = round(ratio * width)
    return "[" + "#" * filled + "." * (width - filled) + f"] {ratio:6.2%}"


def image_count(run_dir: Path, variant: str) -> int:
    return sum(
        1
        for path in (run_dir / variant / "geneval").glob("*/samples/*.png")
        if path.is_file()
    )


def result_stats(run_dir: Path, variant: str) -> tuple[int, int]:
    path = run_dir / variant / "results/results.jsonl"
    if not path.is_file():
        return 0, 0
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    return sum(bool(item["correct"]) for item in records), len(records)


def parse_time(value: str | None) -> float | None:
    if not value:
        return None
    return datetime.fromisoformat(value).timestamp()


def main() -> None:
    args = parse_args()
    run_dir = (args.run_dir or latest_run()).resolve()
    status_path = run_dir / "status.json"
    protocol_path = run_dir / "protocol.json"
    status = (
        json.loads(status_path.read_text(encoding="utf-8"))
        if status_path.is_file()
        else {}
    )
    protocol = (
        json.loads(protocol_path.read_text(encoding="utf-8"))
        if protocol_path.is_file()
        else {}
    )
    variants = tuple((protocol.get("variants") or {"baseline": {}, "guided": {}}))
    counts = {variant: image_count(run_dir, variant) for variant in variants}
    generated = sum(counts.values())
    total = int(status.get("total_images", 64))
    updated = parse_time(status.get("updated_at"))
    update_age = max(0, int(time.time() - updated)) if updated is not None else None
    variant_total = int(protocol.get("prompts", 8)) * int(
        protocol.get("samples_per_prompt", 4)
    )

    print(time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    print()
    print(f"Run:       {run_dir.name}")
    print(f"Status:    {status.get('status', 'not-started')}")
    print(f"Phase:     {status.get('phase', 'not-started')}")
    print(
        f"Update:    {update_age}s ago"
        if update_age is not None
        else "Update:    unavailable"
    )
    print()
    label_width = max(10, *(len(variant) for variant in variants))
    for variant in variants:
        print(
            f"{variant:<{label_width}} {counts[variant]:3d}/{variant_total} "
            f"{progress_bar(counts[variant], variant_total)}"
        )
    print(f"Generated: {generated:2d}/{total} {progress_bar(generated, total)}")

    if status.get("phase") == "generation":
        print()
        print(
            "Current:    "
            f"{status.get('variant')} source={status.get('source_index')} "
            f"relation={status.get('relation')} samples={status.get('sample_indices')}"
        )
        print(
            f"Euler:      {status.get('euler_step', 0)}/{status.get('euler_steps', 99)} "
            f"{progress_bar(int(status.get('euler_step', 0)), int(status.get('euler_steps', 99)))}"
        )

    print()
    for variant in variants:
        correct, scored = result_stats(run_dir, variant)
        score = f"{correct / scored:.2%}" if scored else "pending"
        print(f"Score {variant:<{label_width}} {correct}/{scored} ({score})")
    if status.get("error"):
        print()
        print(f"Error:      {status['error']}")
    print(f"Log:        {run_dir / 'run.log'}")


if __name__ == "__main__":
    main()
