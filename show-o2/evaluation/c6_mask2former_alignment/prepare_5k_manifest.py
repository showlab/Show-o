#!/usr/bin/env python3
"""Select a deterministic direction-balanced 5k detector-alignment split."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
from pathlib import Path


RELATIONS = ("left", "right", "above", "below")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260808)
    args = parser.parse_args()
    if args.samples < 4 or args.samples % 4:
        raise ValueError("samples must be positive and divisible by four")
    grouped = {relation: [] for relation in RELATIONS}
    with args.input.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            grouped[str(row["relation"])].append(row)
    take = args.samples // 4
    selected = []
    for index, relation in enumerate(RELATIONS):
        random.Random(args.seed + index * 1009).shuffle(grouped[relation])
        if len(grouped[relation]) < take:
            raise RuntimeError(f"{relation} has only {len(grouped[relation])} rows")
        selected.extend(grouped[relation][:take])
    random.Random(args.seed + 7919).shuffle(selected)
    if len({row["sample_id"] for row in selected}) != len(selected):
        raise AssertionError("sample_id duplication detected")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    digest = hashlib.sha256()
    with temporary.open("w", encoding="utf-8") as handle:
        for row in selected:
            encoded = (json.dumps(row, sort_keys=True) + "\n").encode()
            handle.write(encoded.decode())
            digest.update(encoded)
    temporary.replace(args.output)
    audit = {
        "protocol": "mask2former-online-alignment-balanced-5k-v1",
        "source": str(args.input.resolve()),
        "output": str(args.output.resolve()),
        "samples": len(selected),
        "seed": args.seed,
        "direction_counts": dict(collections.Counter(row["relation"] for row in selected)),
        "unique_images": len({int(row["image_id"]) for row in selected}),
        "minimum_source_geneval_score": min(
            float(row["geneval_supervision"]["source_box_score"]) for row in selected
        ),
        "sha256": digest.hexdigest(),
    }
    audit_path = args.output.with_suffix(".audit.json")
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
