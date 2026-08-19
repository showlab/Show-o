#!/usr/bin/env python3
"""Print one compact B-prime training progress snapshot."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path


def load(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def bar(percent: float, width: int = 40) -> str:
    filled = min(width, max(0, round(width * percent / 100.0)))
    return "[" + "#" * filled + "." * (width - filled) + "]"


def duration(seconds) -> str:
    if seconds is None:
        return "unknown"
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--session", default="showo2_bprime_train")
    args = parser.parse_args()
    b1, b2 = load(args.run_root / "b1/status.json"), load(args.run_root / "b2/status.json")
    b1_step = int(b1.get("step", 0))
    b2_step = int(b2.get("step", 0))
    total = int(b1.get("total_steps", 200) or 200) + int(b2.get("total_steps", 300) or 300)
    completed = b1_step + b2_step
    percent = 100.0 * completed / max(total, 1)
    if b2:
        phase, active = "B2 position-only", b2
    elif b1:
        phase, active = "B1 native-anchor", b1
    else:
        phase, active = "initializing", {}
    print(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    print(f"Run:       {args.run_root.name}")
    print(f"Phase:     {phase}")
    print(f"B1:        {b1_step}/{b1.get('total_steps', 200)} status={b1.get('status', 'pending')}")
    print(f"B2:        {b2_step}/{b2.get('total_steps', 300)} status={b2.get('status', 'pending')}")
    print(f"Overall:   {bar(percent)} {percent:6.2f}% ({completed}/{total})")
    active_step = int(active.get("step", 0) or 0)
    resume_step = int(active.get("resume_step", 0) or 0)
    elapsed = active.get("elapsed_seconds")
    if elapsed is not None and active_step > resume_step:
        observed_steps = active_step - resume_step
        remaining_steps = max(0, int(active.get("total_steps", active_step)) - active_step)
        eta_seconds = float(elapsed) / observed_steps * remaining_steps
    else:
        eta_seconds = active.get("eta_seconds")
    print(f"ETA phase: {duration(eta_seconds)}")
    metrics = active.get("metrics", {})
    if metrics:
        loss = metrics.get("loss")
        grad = metrics.get("grad_g_spatial", metrics.get("grad_g"))
        update = metrics.get("update_g_spatial", metrics.get("update_g"))
        print(f"Loss:      {loss}  G-grad={grad}  G-update={update}")
    try:
        tmux = subprocess.run(
            ["tmux", "list-panes", "-t", f"={args.session}", "-F", "#{pane_pid} #{pane_current_command} #{pane_dead}"],
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()
    except subprocess.CalledProcessError:
        tmux = "no active tmux pane"
    print(f"tmux:      {tmux}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
