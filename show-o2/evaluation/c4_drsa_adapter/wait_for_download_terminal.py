#!/usr/bin/env python3
"""Wait until another DRSA downloader reaches a terminal status."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


TERMINAL_STATES = {"completed", "failed"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("status_path", type=Path)
    parser.add_argument("--interval", type=float, default=30.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.interval <= 0:
        raise ValueError("--interval must be positive")
    last_reported: str | None = None
    while True:
        state = "status-file-pending"
        error = None
        if args.status_path.is_file():
            try:
                payload = json.loads(args.status_path.read_text())
                state = str(payload.get("status", "unknown"))
                error = payload.get("error")
            except (OSError, json.JSONDecodeError) as exception:
                state = f"status-read-retry:{type(exception).__name__}"
        if state != last_reported:
            print(
                f"Prerequisite download status: {state} ({args.status_path})",
                flush=True,
            )
            last_reported = state
        if state in TERMINAL_STATES:
            if error:
                print(f"Prerequisite download error: {error}", flush=True)
            print(
                "Prerequisite downloader reached a terminal state; "
                "continuing the full resumable download regardless of smoke outcome.",
                flush=True,
            )
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
