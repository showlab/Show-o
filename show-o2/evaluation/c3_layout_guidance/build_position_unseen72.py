#!/usr/bin/env python3
"""Build every GenEval Position prompt unseen by quick tuning and holdout20."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


SHOWO2_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = SHOWO2_ROOT.parents[1]
RELATIONS = ("left of", "right of", "above", "below")
PROTOCOL = "showo2-position-unseen72-v1"


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
            WORKSPACE_ROOT / "data/showo2_layout/position_quick8_v2.jsonl",
            WORKSPACE_ROOT / "data/showo2_layout/position_holdout20_v1.jsonl",
        ],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=WORKSPACE_ROOT / "data/showo2_layout/position_unseen72_v1.jsonl",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def source_index(row: dict[str, Any]) -> int:
    for key in ("source_index", "_source_index"):
        if key in row:
            return int(row[key])
    raise KeyError(f"No source index in record: {row}")


def relation(metadata: dict[str, Any]) -> str:
    positioned = [item for item in metadata["include"] if "position" in item]
    if len(positioned) != 1:
        raise ValueError(f"Expected one positioned object, got {positioned}")
    value = str(positioned[0]["position"][0])
    if value not in RELATIONS:
        raise ValueError(f"Unsupported Position relation: {value}")
    return value


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    excluded = {
        source_index(row)
        for path in args.exclude_manifest
        for row in read_jsonl(path)
    }
    source = read_jsonl(args.source_metadata)
    records = []
    for index, metadata in enumerate(source):
        if metadata.get("tag") != "position" or index in excluded:
            continue
        records.append(
            {
                "local_index": len(records),
                "source_index": index,
                "relation": relation(metadata),
                "prompt": metadata["prompt"],
                "metadata": metadata,
                "selection": {
                    "protocol": PROTOCOL,
                    "source_metadata_sha256": hashlib.sha256(
                        args.source_metadata.read_bytes()
                    ).hexdigest(),
                    "excluded_source_indices": sorted(excluded),
                },
            }
        )
    counts = Counter(row["relation"] for row in records)
    if len(records) != 72:
        raise RuntimeError(f"Expected 72 unseen Position prompts, got {len(records)}")
    if set(counts) != set(RELATIONS):
        raise RuntimeError(f"Missing Position directions: {counts}")
    indices = {row["source_index"] for row in records}
    if indices & excluded:
        raise RuntimeError("Unseen manifest overlaps an excluded source index")
    write_jsonl(args.output, records)
    print(
        json.dumps(
            {
                "status": "ok",
                "output": str(args.output.resolve()),
                "prompts": len(records),
                "relations": dict(counts),
                "excluded": len(excluded),
                "overlap": len(indices & excluded),
                "protocol": PROTOCOL,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
