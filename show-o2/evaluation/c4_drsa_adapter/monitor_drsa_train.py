#!/usr/bin/env python3
"""Render DRSA training status, progress, ETA and process information."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path


def duration(seconds):
    if seconds is None:
        return "pending"
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def command_output(command):
    try:
        return subprocess.run(
            command, check=False, capture_output=True, text=True, timeout=3
        ).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("smoke", "train"), nargs="?", default="train")
    parser.add_argument("--run-root", type=Path, default=Path("/root/autodl-tmp/runs/showo2_drsa_train"))
    parser.add_argument("--session")
    args = parser.parse_args()
    marker = args.run_root / f"latest_{args.mode}.txt"
    if not marker.is_file():
        raise SystemExit(f"No run marker: {marker}")
    run_dir = Path(marker.read_text().strip())
    status_path = run_dir / "status.json"
    status = json.loads(status_path.read_text()) if status_path.is_file() else {}
    session = args.session or f"showo2_drsa_{args.mode}"
    step = int(status.get("step", 0))
    total = int(status.get("total_steps", 0))
    resume_step = int(status.get("resume_step", 0))
    completed_since_resume = max(0, step - resume_step)
    eta_seconds = status.get("eta_seconds")
    if completed_since_resume > 0 and total > step:
        elapsed = float(status.get("elapsed_seconds", 0.0))
        eta_seconds = elapsed / completed_since_resume * (total - step)
    percent = float(status.get("percent", 0.0))
    filled = max(0, min(40, round(percent * 40 / 100)))
    bar = "#" * filled + "." * (40 - filled)
    metrics = status.get("metrics", {})
    counts = status.get("counts", {})
    tmux_info = command_output([
        "tmux", "list-panes", "-t", f"={session}",
        "-F", "pid=#{pane_pid} command=#{pane_current_command} dead=#{pane_dead}",
    ]) or "not running"
    gpu = command_output([
        "nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]) or "unavailable"
    print(time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    print()
    print(f"Run:       {run_dir.name}")
    print(f"Status:    {status.get('status', 'starting')}")
    print(f"Phase:     {status.get('phase', 'initializing')}")
    print(f"Step:      {step}/{total}")
    print(f"Progress:  [{bar}] {percent:6.2f}%")
    print(f"Elapsed:   {duration(status.get('elapsed_seconds'))}")
    print(f"ETA:       {duration(eta_seconds)}")
    if metrics:
        print(f"Loss:      {metrics.get('loss', float('nan')):.6f}  LR={metrics.get('lr', float('nan')):.3e}")
        print(
            "Grad:      "
            f"U={metrics.get('grad_u', 0.0):.3e}  "
            f"G={metrics.get('grad_g', 0.0):.3e}  "
            f"Relation={metrics.get('grad_relation', 0.0):.3e}"
        )
    print(f"Counts:    SPRIGHT={counts.get('spright', 0)} VSR={counts.get('vsr', 0)}")
    print(f"tmux:      {tmux_info}")
    print(f"GPU:       {gpu}")
    print(f"Log:       {run_dir / 'run.log'}")
    if status.get("error"):
        print(f"Error:     {status['error']}")
    log_path = run_dir / "run.log"
    if log_path.is_file():
        lines = log_path.read_text(errors="replace").splitlines()[-8:]
        if lines:
            print("\nRecent log:")
            print("\n".join(lines))


if __name__ == "__main__":
    main()
