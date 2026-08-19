#!/usr/bin/env python3
"""Print deterministic progress for a DRSA data preparation run."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path


def _bar(value: int, total: int, width: int = 40) -> str:
    ratio = 0.0 if total <= 0 else min(1.0, value / total)
    filled = int(round(width * ratio))
    return "[" + "#" * filled + "." * (width - filled) + f"] {ratio * 100:6.2f}%"


def _size_for(entry: dict) -> int:
    final = Path(entry["destination"])
    part = final.with_suffix(final.suffix + ".part")
    if final.is_file():
        return final.stat().st_size
    if part.is_file():
        return part.stat().st_size
    return int(entry.get("downloaded", 0))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--session", default="showo2_drsa_data")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_path = args.run_dir / "manifest.json"
    status_path = args.run_dir / "status.json"
    if not manifest_path.is_file():
        print(f"Run:       {args.run_dir.name}")
        print("Status:    initializing (fetching remote manifest)")
        try:
            output = subprocess.check_output(
                ["tmux", "list-panes", "-t", args.session, "-F", "#{pane_pid} #{pane_current_command}"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            print(f"tmux:      {output or 'session exists'}")
        except (FileNotFoundError, subprocess.CalledProcessError):
            print("tmux:      session not found")
        log_path = args.run_dir / "run.log"
        if log_path.is_file():
            print(f"Log:       {log_path}")
        return 0
    manifest = json.loads(manifest_path.read_text())
    status = json.loads(status_path.read_text()) if status_path.is_file() else {}
    status_files = status.get("files", {})
    entries = []
    for spec in manifest["files"]:
        state = status_files.get(spec["key"], {})
        entry = {**spec, **state}
        entry["current_size"] = _size_for(entry)
        entries.append(entry)
    total = sum(int(entry["expected_size"]) for entry in entries)
    current = sum(min(entry["current_size"], int(entry["expected_size"])) for entry in entries)
    complete = sum(entry["current_size"] == int(entry["expected_size"]) for entry in entries)
    print(f"Run:       {args.run_dir.name}")
    print(f"Status:    {status.get('status', 'initializing')}")
    print(f"Files:     {complete}/{len(entries)}")
    print(f"Bytes:     {current / 2**30:.2f}/{total / 2**30:.2f} GiB")
    print(f"Progress:  {_bar(current, total)}")
    active = [
        entry
        for entry in entries
        if entry["current_size"] < int(entry["expected_size"])
        and entry.get("status") in {"downloading", "retrying"}
    ]
    for entry in active:
        print(
            "Active:    "
            f"{Path(entry['destination']).name} "
            f"{entry['current_size'] / 2**20:.1f}/"
            f"{int(entry['expected_size']) / 2**20:.1f} MiB "
            f"({entry.get('status')}, attempt {entry.get('attempt', 1)})"
        )
    try:
        output = subprocess.check_output(
            ["tmux", "list-panes", "-t", args.session, "-F", "#{pane_pid} #{pane_current_command}"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        print(f"tmux:      {output or 'session exists'}")
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("tmux:      session not found")
    log_path = args.run_dir / "run.log"
    if log_path.is_file():
        print(f"Log:       {log_path}")
    smoke_path = args.run_dir / "smoke.json"
    if smoke_path.is_file():
        smoke = json.loads(smoke_path.read_text())
        print(
            "Smoke:     "
            f"{smoke.get('status')} "
            f"SPRIGHT={smoke.get('spright_coco', {}).get('samples_checked')} "
            f"VSR={smoke.get('vsr', {}).get('rows_checked')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
