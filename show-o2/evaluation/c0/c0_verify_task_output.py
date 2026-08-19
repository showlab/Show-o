#!/usr/bin/env python3
"""Verify lmms-eval emitted both aggregate results and all sample rows."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected", type=int, required=True)
    args = parser.parse_args()
    output = Path(args.output)
    results = sorted(output.rglob("*_results.json"))
    samples = sorted(output.rglob("*_samples_*.jsonl"))
    if not results:
        raise SystemExit(f"No results JSON found below {output}")
    if not samples:
        raise SystemExit(f"No samples JSONL found below {output}")
    rows = 0
    for path in samples:
        with path.open("r", encoding="utf-8") as handle:
            rows += sum(1 for line in handle if line.strip())
    if rows != args.expected:
        raise SystemExit(f"Expected {args.expected} sample rows, found {rows} below {output}")
    aggregate = None
    for path in reversed(results):
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("results"):
            aggregate = path
            break
    if aggregate is None:
        raise SystemExit(f"No aggregate results JSON with task metrics found below {output}")
    print(json.dumps({"status": "ok", "rows": rows, "results": str(aggregate)}))


if __name__ == "__main__":
    main()
