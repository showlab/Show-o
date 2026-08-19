#!/usr/bin/env python3
"""Filter existing B-prime manifests with GenEval's exact position rule."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.c5_boxdiff_adapter.geneval_geometry import (
    POSITION_THRESHOLD,
    RELATION_THRESHOLD,
    RELATIONS,
    geneval_relation_score,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--position-threshold", type=float, default=POSITION_THRESHOLD)
    parser.add_argument("--relation-threshold", type=float, default=RELATION_THRESHOLD)
    parser.add_argument("--relation-margin", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260808)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def filter_and_balance(rows, *, split, args):
    grouped = {relation: [] for relation in RELATIONS}
    failures = collections.Counter()
    scores = {relation: [] for relation in RELATIONS}
    minimum_score = args.relation_threshold + args.relation_margin
    for row in rows:
        relation = str(row["relation"])
        if relation not in grouped:
            failures["unsupported_relation"] += 1
            continue
        objects = row.get("objects", [])
        if len(objects) != 2:
            failures["object_count"] += 1
            continue
        score = geneval_relation_score(
            objects[0]["box"],
            objects[1]["box"],
            relation,
            position_threshold=args.position_threshold,
        )
        scores[relation].append(score)
        if score < minimum_score:
            failures[f"below_margin_{relation}"] += 1
            continue
        item = dict(row)
        item["geneval_supervision"] = {
            "expected_relation": relation,
            "source_box_score": score,
            "position_threshold": args.position_threshold,
            "evaluation_threshold": args.relation_threshold,
            "training_margin": args.relation_margin,
            "training_min_score": minimum_score,
        }
        grouped[relation].append(item)

    before_balance = {relation: len(items) for relation, items in grouped.items()}
    take = min(before_balance.values())
    if take < 1:
        raise RuntimeError(f"cannot balance {split}: {before_balance}")
    selected = []
    for index, relation in enumerate(RELATIONS):
        random.Random(args.seed + index + (10000 if split == "validation" else 0)).shuffle(
            grouped[relation]
        )
        selected.extend(grouped[relation][:take])
    random.Random(args.seed + (20000 if split == "validation" else 10000)).shuffle(selected)
    return selected, {
        "input_rows": len(rows),
        "score_minimum": minimum_score,
        "direction_counts_after_formula_filter": before_balance,
        "rows_per_direction_after_balance": take,
        "output_rows": len(selected),
        "failures": dict(failures),
        "score_ranges": {
            relation: {
                "min": min(values) if values else None,
                "max": max(values) if values else None,
            }
            for relation, values in scores.items()
        },
    }


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit = {
        "protocol": "bprime-geneval-aligned-manifest-v2",
        "config": {
            "input_dir": str(args.input_dir.resolve()),
            "output_dir": str(args.output_dir.resolve()),
            "position_threshold": args.position_threshold,
            "relation_threshold": args.relation_threshold,
            "relation_margin": args.relation_margin,
            "seed": args.seed,
        },
        "splits": {},
    }
    image_ids = {}
    for split in ("train", "validation"):
        input_path = args.input_dir / f"{split}.jsonl"
        output_path = args.output_dir / f"{split}.jsonl"
        selected, split_audit = filter_and_balance(
            load_jsonl(input_path), split=split, args=args
        )
        write_jsonl(output_path, selected)
        image_ids[split] = {int(row["image_id"]) for row in selected}
        split_audit["unique_images"] = len(image_ids[split])
        split_audit["sha256"] = sha256(output_path)
        audit["splits"][split] = split_audit
    overlap = image_ids["train"] & image_ids["validation"]
    audit["train_validation_image_overlap"] = len(overlap)
    if overlap:
        raise AssertionError("train/validation image leakage detected")
    audit_path = args.output_dir / "audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
