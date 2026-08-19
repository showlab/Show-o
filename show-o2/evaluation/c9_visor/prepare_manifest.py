#!/usr/bin/env python3
"""Prepare deterministic pair-complete SR2D/VISOR evaluation manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SHOWO2_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = SHOWO2_ROOT.parents[1]
OFFICIAL_COMMIT = "b9e368b73422aad655cc7121648beaca5c94563b"
OFFICIAL_JSON_GIT_BLOB = "e8c8d38ba166e69dac361a51b2053042335b42ea"
RELATIONS = ("to the left of", "to the right of", "above", "below")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--official-json",
        type=Path,
        default=WORKSPACE_ROOT / "visor/text_spatial_rel_phrases.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--pair-count",
        type=int,
        default=25,
        help="Unordered object pairs; zero selects all 3,160 pairs.",
    )
    parser.add_argument("--seed", type=int, default=20260811)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_blob_sha1(path: Path) -> str:
    data = path.read_bytes()
    header = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(header + data).hexdigest()


def read_official(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    if git_blob_sha1(path) != OFFICIAL_JSON_GIT_BLOB:
        raise RuntimeError(
            "official SR2D JSON does not match the pinned VISOR commit "
            f"{OFFICIAL_COMMIT}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or len(payload) != 31680:
        raise ValueError("pinned SR2D JSON must contain exactly 31,680 rows")
    return payload


def relation_groups(
    records: list[dict[str, Any]],
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        relation = str(record["rel_type"])
        if relation not in RELATIONS:
            continue
        object_a = str(record["obj_1_attributes"][0])
        object_b = str(record["obj_2_attributes"][0])
        if not object_a or not object_b or object_a == object_b:
            raise ValueError(f"invalid SR2D object pair: {record}")
        groups[tuple(sorted((object_a, object_b)))].append(record)
    for pair, rows in groups.items():
        directions = Counter(str(row["rel_type"]) for row in rows)
        orders = Counter(
            (str(row["obj_1_attributes"][0]), str(row["obj_2_attributes"][0]))
            for row in rows
        )
        if len(rows) != 8 or set(directions.values()) != {2} or set(orders.values()) != {4}:
            raise ValueError(f"SR2D pair is not an 8-row complete group: {pair}")
    if len(groups) != 3160:
        raise ValueError(f"expected 3,160 SR2D object pairs, got {len(groups)}")
    return dict(groups)


def select_records(
    groups: dict[tuple[str, str], list[dict[str, Any]]], pair_count: int, seed: int
) -> list[dict[str, Any]]:
    if pair_count < 0 or pair_count > len(groups):
        raise ValueError(f"pair-count must be between 0 and {len(groups)}")
    pairs = sorted(groups)
    if pair_count:
        random.Random(seed).shuffle(pairs)
        pairs = sorted(pairs[:pair_count])
    selected = [row for pair in pairs for row in groups[pair]]
    return sorted(selected, key=lambda row: int(row["unique_id"]))


def manifest_row(record: dict[str, Any], manifest_index: int) -> dict[str, Any]:
    object_a = str(record["obj_1_attributes"][0])
    object_b = str(record["obj_2_attributes"][0])
    relation = str(record["rel_type"])
    return {
        "manifest_index": manifest_index,
        "unique_id": int(record["unique_id"]),
        "prompt": str(record["text"]),
        "object_a": object_a,
        "object_b": object_b,
        "pair_key": sorted((object_a, object_b)),
        "relation": relation,
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    records = read_official(args.official_json)
    selected = select_records(relation_groups(records), args.pair_count, args.seed)
    rows = [manifest_row(record, index) for index, record in enumerate(selected)]
    relation_counts = Counter(row["relation"] for row in rows)
    object_counts = Counter(
        obj for row in rows for obj in (row["object_a"], row["object_b"])
    )
    if len(set(relation_counts.values())) != 1:
        raise RuntimeError("selected VISOR manifest is not direction-balanced")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
    temporary.replace(args.output)
    summary = {
        "protocol": "sr2d-visor-pair-complete-v1",
        "official_commit": OFFICIAL_COMMIT,
        "official_json": str(args.official_json.resolve()),
        "official_json_git_blob": OFFICIAL_JSON_GIT_BLOB,
        "official_json_sha256": file_sha256(args.official_json),
        "manifest": str(args.output.resolve()),
        "manifest_sha256": file_sha256(args.output),
        "seed": args.seed,
        "pair_count_requested": args.pair_count,
        "pair_count": len(rows) // 8,
        "prompts": len(rows),
        "relation_counts": dict(sorted(relation_counts.items())),
        "object_count": len(object_counts),
        "object_occurrences": dict(sorted(object_counts.items())),
    }
    write_json(args.output.with_suffix(".audit.json"), summary)
    return summary


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

