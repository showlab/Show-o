#!/usr/bin/env python3
"""Derive a fixed COCO-500 pretest manifest from the paired COCO-5K protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-reference-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=500)
    args = parser.parse_args()
    if args.count < 2:
        raise ValueError("count must be at least two")
    rows = [
        json.loads(line) for line in args.source_manifest.read_text().splitlines()
        if line.strip()
    ]
    if len(rows) < args.count:
        raise ValueError(f"source manifest has only {len(rows)} rows")
    selected = rows[: args.count]
    for index, row in enumerate(selected):
        if int(row["index"]) != index or row["reference_filename"] != f"{index:05d}.jpg":
            raise ValueError(f"source manifest is not prefix-compatible at row {index}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reference_dir = args.output_dir / "references"
    reference_dir.mkdir(parents=True, exist_ok=True)
    expected_names = set()
    for row in selected:
        name = str(row["reference_filename"])
        expected_names.add(name)
        source = args.source_reference_dir / name
        destination = reference_dir / name
        if not source.is_file():
            raise FileNotFoundError(source)
        if not destination.exists():
            os.link(source, destination)
        if sha256(destination) != str(row["reference_sha256"]):
            raise RuntimeError(f"reference hash mismatch: {destination}")
    extras = {
        path.name for path in reference_dir.iterdir()
        if path.is_file() and path.name not in expected_names
    }
    if extras:
        raise RuntimeError(f"derived reference directory has unexpected files: {sorted(extras)[:5]}")
    manifest = args.output_dir / "manifest.jsonl"
    payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in selected)
    if manifest.exists() and manifest.read_text() != payload:
        raise RuntimeError(f"refusing to replace different manifest: {manifest}")
    manifest.write_text(payload)
    protocol = {
        "protocol_version": "showo2-coco500-quality-pretest-v1",
        "purpose": "checkpoint screening only; not comparable to COCO-5K metrics",
        "selection": f"first_{args.count}_rows_of_fixed_coco5k_v1",
        "count": args.count,
        "source_manifest": str(args.source_manifest.resolve()),
        "source_manifest_sha256": sha256(args.source_manifest),
        "manifest": str(manifest.resolve()),
        "manifest_sha256": sha256(manifest),
        "source_reference_dir": str(args.source_reference_dir.resolve()),
        "reference_dir": str(reference_dir.resolve()),
        "seed_policy": "unchanged per-row seed from parent manifest",
    }
    protocol_path = args.output_dir / "protocol.json"
    text = json.dumps(protocol, indent=2, sort_keys=True) + "\n"
    if protocol_path.exists() and protocol_path.read_text() != text:
        raise RuntimeError(f"refusing to replace different protocol: {protocol_path}")
    protocol_path.write_text(text)
    print(json.dumps(protocol, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
