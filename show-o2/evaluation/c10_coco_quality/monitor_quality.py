#!/usr/bin/env python3
"""Display live progress for the formal four-model COCO-5K quality queue."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_RUN_DIR = Path(
    "/root/autodl-tmp/runs/showo2_coco_quality/"
    "coco5k-four-model-20260812T080000Z"
)
VARIANTS = (
    ("baseline", "Show-o2 baseline"),
    ("mask_logits_500", "Mask-logits step-500"),
    ("mask_flow_500", "Mask+Flow step-500"),
    ("mask_flow_k4_5000", "K4 Mask+Flow step-5000"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument(
        "--interval",
        type=float,
        default=0.0,
        help="refresh interval in seconds; zero prints once",
    )
    parser.add_argument("--no-gpu", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def progress_from_status(status: dict[str, Any] | None) -> tuple[float, int, int]:
    if not status:
        return 0.0, 0, 5000
    completed = int(status.get("completed_images", 0))
    total = int(status.get("total_images", 5000))
    active = int(status.get("active_images", 0))
    step = int(status.get("sampler_step", 0))
    steps = int(status.get("sampler_steps", 0))
    partial = active * step / steps if steps else 0.0
    return min(float(total), completed + partial), completed, total


def bar(percent: float, width: int = 28) -> str:
    filled = min(width, max(0, round(width * percent / 100.0)))
    return "#" * filled + "." * (width - filled)


def duration(seconds: Any) -> str:
    try:
        value = max(0, int(float(seconds)))
    except (TypeError, ValueError):
        return "n/a"
    hours, remainder = divmod(value, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def scoring_line(label: str, path: Path) -> str:
    status = read_json(path)
    if not status:
        return f"[WAIT] {label:<13} progress=  0.00% phase=not_started"
    state = str(status.get("status", "unknown"))
    tag = {"completed": "DONE", "running": "RUN ", "failed": "FAIL"}.get(
        state, "????"
    )
    percent = float(status.get("percent", 0.0))
    phase = status.get("phase", "unknown")
    current = status.get("current")
    suffix = f" current={current}" if current else ""
    return (
        f"[{tag}] {label:<13} progress={percent:6.2f}% "
        f"phase={phase}{suffix}"
    )


def gpu_line() -> str:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        fields = [part.strip() for part in result.stdout.splitlines()[0].split(",")]
        return (
            f"gpu={fields[0]} util={fields[1]}% "
            f"memory={fields[2]}/{fields[3]} MiB"
        )
    except (IndexError, OSError, subprocess.SubprocessError):
        return "gpu=unavailable"


def render(run_dir: Path, show_gpu: bool) -> str:
    lines = [
        datetime.now().astimezone().isoformat(timespec="seconds"),
        f"run={run_dir}",
    ]
    total_progress = 0.0
    total_completed = 0
    total_images = 0
    rows = []
    for key, label in VARIANTS:
        status = read_json(run_dir / "generation" / key / "status.json")
        progress, completed, total = progress_from_status(status)
        total_progress += progress
        total_completed += completed
        total_images += total
        if not status:
            rows.append(
                f"[WAIT] {label:<25} progress=  0.00% images=0/{total} "
                "phase=not_started"
            )
            continue
        state = str(status.get("status", "unknown"))
        tag = {"completed": "DONE", "running": "RUN ", "failed": "FAIL"}.get(
            state, "????"
        )
        percent = 100.0 * progress / total if total else 100.0
        phase = str(status.get("phase", "unknown"))
        step = int(status.get("sampler_step", 0))
        steps = int(status.get("sampler_steps", 0))
        eta = duration(status.get("eta_seconds"))
        rows.append(
            f"[{tag}] {label:<25} progress={percent:6.2f}% "
            f"images={completed}/{total} sampler={step}/{steps} "
            f"phase={phase} eta={eta}"
        )
    overall = 100.0 * total_progress / total_images if total_images else 0.0
    formal_completed = (run_dir / "FORMAL_COMPLETED").is_file()
    queue_state = "completed" if formal_completed else "running"
    lines.extend(
        [
            f"queue_status={queue_state}",
            (
                f"generation [{bar(overall)}] {overall:6.2f}% "
                f"completed_images={total_completed}/{total_images}"
            ),
            "",
            *rows,
            "",
            scoring_line("CleanFID/KID", run_dir / "scoring/fid_kid/status.json"),
            scoring_line("official CMMD", run_dir / "scoring/cmmd/status.json"),
        ]
    )
    if show_gpu:
        lines.extend(["", gpu_line()])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.interval < 0:
        raise ValueError("--interval must be non-negative")
    while True:
        if args.interval:
            print("\033[2J\033[H", end="")
        print(render(args.run_dir.resolve(), not args.no_gpu), flush=True)
        if not args.interval:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
