#!/usr/bin/env python3
"""Print one compact B-prime benchmark progress snapshot."""

import argparse
import json
import subprocess
from pathlib import Path


def load(path):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError, IsADirectoryError):
        return {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--session", default="showo2_bprime_benchmark")
    args = parser.parse_args()
    marker = args.run_root / "benchmark.status"
    marker_text = marker.read_text().strip() if marker.exists() else "phase=initializing status=pending"
    phase = dict(item.split("=", 1) for item in marker_text.split()).get("phase", "initializing")
    paths = {
        "vsr": args.run_root / "vsr/status.json",
        "quick_position": args.run_root / "quick_position/status.json",
        "full_position": args.run_root / "full_position/status.json",
    }
    status = load(paths.get(phase, ""))
    if phase == "vsr":
        done, total = status.get("completed_predictions", 0), status.get("total_predictions", 1536)
    elif phase in {"quick_position", "full_position"}:
        done, total = status.get("completed_images", 0), status.get("total_images", 1)
    else:
        done, total = (1, 1) if phase == "completed" else (0, 1)
    percent = 100.0 * done / max(total, 1)
    width, filled = 40, round(40 * percent / 100.0)
    print(f"Phase:   {phase}")
    print(f"Progress:[{'#' * filled}{'.' * (width - filled)}] {percent:6.2f}% ({done}/{total})")
    try:
        pane = subprocess.run(
            ["tmux", "list-panes", "-t", f"={args.session}", "-F", "#{pane_pid} #{pane_current_command} #{pane_dead}"],
            text=True, capture_output=True, check=True,
        ).stdout.strip()
    except subprocess.CalledProcessError:
        pane = "no active tmux pane"
    print(f"tmux:    {pane}")


if __name__ == "__main__":
    main()
