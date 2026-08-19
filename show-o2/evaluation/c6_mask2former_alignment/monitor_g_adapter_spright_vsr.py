#!/usr/bin/env python3
"""Print step-level progress for SPRIGHT+VSR G-adapter warmup."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--session")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=2.0)
    return parser.parse_args()


def duration(seconds) -> str:
    if seconds is None:
        return "n/a"
    seconds = max(0, int(float(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def bar(percent: float, width: int = 28) -> str:
    filled = min(width, max(0, round(width * percent / 100.0)))
    return "[" + "#" * filled + "." * (width - filled) + "]"


def command_output(command: list[str]) -> str:
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    return result.stdout.strip()


def render(args: argparse.Namespace) -> str:
    status_path = args.run_dir / "status.json"
    if not status_path.is_file():
        return f"run={args.run_dir}\nstatus=waiting_for_status_json"
    try:
        status = json.loads(status_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        return f"run={args.run_dir}\nstatus=status_read_retry error={error}"
    step = int(status.get("step", 0))
    total = int(status.get("total_steps", 0))
    percent = float(status.get("percent", 0.0))
    counts = status.get("counts", {})
    sizes = status.get("dataset_sizes", {})
    metrics = status.get("metrics", {})
    lines = [
        time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        f"run={args.run_dir}",
        (
            f"status={status.get('status', 'unknown')} phase={status.get('phase', 'unknown')} "
            f"step={step}/{total}"
        ),
        f"total {bar(percent)} {percent:7.3f}% eta={duration(status.get('eta_seconds'))}",
        (
            f"SPRIGHT {int(counts.get('spright', 0))}/{int(sizes.get('spright', 0))}  "
            f"VSR {int(counts.get('vsr', 0))}/{int(sizes.get('vsr', 0))}"
        ),
    ]
    if metrics:
        lines.append(
            f"task={metrics.get('task')} loss={float(metrics.get('loss', 0.0)):.6f} "
            f"g_grad={float(metrics.get('grad_generation_adapter', 0.0)):.4e} "
            f"lr={float(metrics.get('lr', 0.0)):.4e}"
        )
    latest = status.get("latest_checkpoint")
    if latest:
        lines.append(f"checkpoint={latest}")
    if status.get("error"):
        lines.append(f"error={status['error']}")
    gpu = command_output([
        "nvidia-smi",
        "--query-gpu=index,utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ])
    if gpu:
        lines.append(f"gpu={gpu}")
    if args.session:
        pane = command_output([
            "tmux", "list-panes", "-t", f"={args.session}",
            "-F", "#{pane_pid} #{pane_current_command} #{pane_dead}",
        ])
        lines.append(f"tmux={pane or 'not_running'}")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    if args.interval <= 0:
        raise ValueError("interval must be positive")
    while True:
        print("\033[2J\033[H" if args.watch else "", end="")
        print(render(args), flush=True)
        if not args.watch:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
