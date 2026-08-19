#!/usr/bin/env python3
"""Select a deterministic type-stratified subset of the Self-cycle dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"empty dataset: {path}")
    return rows


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_type_seed(seed: int, record_type: str) -> int:
    value = hashlib.sha256(f"{seed}:{record_type}".encode("utf-8")).digest()
    return int.from_bytes(value[:8], "big")


def select_indices(
    rows: list[dict[str, Any]], size: int, seed: int
) -> tuple[list[int], dict[str, int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        record_type = str(row.get("type", "")).strip()
        if not record_type:
            raise ValueError(f"record {index} has no type")
        groups[record_type].append(index)
    if size < len(groups):
        raise ValueError(
            f"subset size {size} is smaller than the {len(groups)} type strata"
        )
    base, remainder = divmod(size, len(groups))
    extra_types = {
        name
        for name, _ in sorted(
            groups.items(), key=lambda item: (-len(item[1]), item[0])
        )[:remainder]
    }
    allocation = {
        name: base + int(name in extra_types) for name in sorted(groups)
    }
    selected: list[int] = []
    for name, indices in sorted(groups.items()):
        quota = allocation[name]
        if len(indices) < quota:
            raise ValueError(
                f"type {name!r} has {len(indices)} rows but needs {quota}"
            )
        rng = random.Random(stable_type_seed(seed, name))
        selected.extend(rng.sample(indices, quota))
    selected.sort()
    if len(selected) != size or len(set(selected)) != size:
        raise RuntimeError("compact selection is not unique or has the wrong size")
    return selected, allocation


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--size", type=int, default=96)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--print-csv", action="store_true")
    args = parser.parse_args()
    if args.size < 1:
        raise ValueError("size must be positive")
    rows = load_jsonl(args.dataset)
    indices, allocation = select_indices(rows, args.size, args.seed)
    selected = [rows[index] for index in indices]
    type_counts = Counter(str(row["type"]) for row in selected)
    question_counts = Counter()
    for row in selected:
        question_counts[str(row["type"])] += len(row.get("qa_pairs", []))
    payload = {
        "protocol": "self-cycle-compact96-stratified-v1",
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": file_sha256(args.dataset),
        "source_records": len(rows),
        "selected_records": len(indices),
        "selected_questions": sum(question_counts.values()),
        "seed": args.seed,
        "selection": "equal-per-type with remainder assigned to largest strata",
        "allocation": allocation,
        "selected_type_counts": dict(sorted(type_counts.items())),
        "selected_question_counts": dict(sorted(question_counts.items())),
        "record_indices": indices,
        "record_indices_csv": ",".join(str(index) for index in indices),
    }
    if args.report:
        atomic_json(args.report, payload)
    if args.print_csv:
        print(payload["record_indices_csv"])
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
