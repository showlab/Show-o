#!/usr/bin/env python3
"""Display C7 resource download and environment preparation progress."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def human(value: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    size = float(value)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    raise AssertionError


def bar(percent: float, width: int = 32) -> str:
    filled = min(width, max(0, round(width * percent / 100.0)))
    return "[" + "#" * filled + "." * (width - filled) + "]"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    status_path = args.run_dir.resolve() / "status.json"
    if not status_path.is_file():
        print(f"status pending: {status_path}")
        return 1
    data = json.loads(status_path.read_text(encoding="utf-8"))
    print(f"Status: {data['status']}  resources={data['completed_resources']}/{data['total_resources']}")
    print(f"Updated: {data['updated_at']}")
    print()
    for name, item in data["resources"].items():
        expected = int(item.get("expected_bytes", 0))
        downloaded = int(item.get("downloaded_bytes", 0))
        percent = 100.0 if item["status"] == "completed" else float(item.get("percent", 100.0 * downloaded / expected if expected else 0.0))
        print(
            f"{name:20s} {item['status']:10s} {bar(percent)} "
            f"{percent:6.2f}%  {human(downloaded)}/{human(expected)}"
        )
        if item.get("error"):
            print(f"  error: {item['error']}")
    usage = shutil.disk_usage("/root/autodl-tmp")
    print()
    print(
        f"Disk: used={human(usage.used)} free={human(usage.free)} "
        f"total={human(usage.total)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
