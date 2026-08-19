#!/usr/bin/env python3
"""Build a deterministic Position holdout disjoint from the quick tuning sets."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

SHOWO2_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = SHOWO2_ROOT.parents[1]
RELATIONS = ("left of", "right of", "above", "below")
SALT = "showo2-position-holdout20-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-metadata",
        type=Path,
        default=WORKSPACE_ROOT / "geneval/prompts/evaluation_metadata.jsonl",
    )
    parser.add_argument(
        "--exclude-manifest",
        type=Path,
        action="append",
        default=[
            WORKSPACE_ROOT / "data/showo2_layout/position_quick8.jsonl",
            WORKSPACE_ROOT / "data/showo2_layout/position_quick8_v2.jsonl",
        ],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=WORKSPACE_ROOT / "data/showo2_layout/position_holdout20_v1.jsonl",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def source_index(record: dict) -> int:
    for key in ("source_index", "_source_index"):
        if key in record:
            return int(record[key])
    raise KeyError(f"No source index in record: {record}")


def relation(metadata: dict) -> str:
    positioned = [item for item in metadata["include"] if "position" in item]
    if len(positioned) != 1:
        raise ValueError(f"Expected one positioned object, got {positioned}")
    return str(positioned[0]["position"][0])


def rank_key(index: int) -> str:
    return hashlib.sha256(f"{SALT}:{index}".encode("ascii")).hexdigest()


def atomic_write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    excluded = {
        source_index(record)
        for path in args.exclude_manifest
        for record in read_jsonl(path)
    }
    candidates: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    for index, metadata in enumerate(read_jsonl(args.source_metadata)):
        if metadata.get("tag") != "position" or index in excluded:
            continue
        item_relation = relation(metadata)
        if item_relation not in RELATIONS:
            raise ValueError(f"Unsupported Position relation: {item_relation}")
        candidates[item_relation].append((index, metadata))

    selected = []
    for item_relation in RELATIONS:
        ranked = sorted(candidates[item_relation], key=lambda item: rank_key(item[0]))
        if len(ranked) < 5:
            raise ValueError(
                f"Not enough {item_relation!r} candidates after exclusions"
            )
        selected.extend(
            (index, item_relation, metadata) for index, metadata in ranked[:5]
        )
    selected.sort(key=lambda item: (RELATIONS.index(item[1]), rank_key(item[0])))

    records = []
    for local_index, (index, item_relation, metadata) in enumerate(selected):
        records.append(
            {
                "local_index": local_index,
                "source_index": index,
                "relation": item_relation,
                "prompt": metadata["prompt"],
                "metadata": metadata,
                "selection": {
                    "salt": SALT,
                    "rank_sha256": rank_key(index),
                    "excluded_source_indices": sorted(excluded),
                },
            }
        )

    counts = Counter(record["relation"] for record in records)
    expected = Counter({item: 5 for item in RELATIONS})
    if counts != expected:
        raise RuntimeError(f"Unbalanced holdout: {counts}")
    if {record["source_index"] for record in records} & excluded:
        raise RuntimeError("Holdout overlaps a quick tuning manifest")

    atomic_write_jsonl(args.output, records)
    print(
        json.dumps(
            {
                "status": "ok",
                "output": str(args.output.resolve()),
                "prompts": len(records),
                "relations": counts,
                "source_indices": [record["source_index"] for record in records],
                "excluded_source_indices": sorted(excluded),
                "selection_salt": SALT,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
