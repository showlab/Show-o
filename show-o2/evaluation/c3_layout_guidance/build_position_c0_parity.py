#!/usr/bin/env python3
"""Attach C0 rewritten prompts to a selected GenEval Position manifest."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selection-manifest",
        type=Path,
        default=WORKSPACE_ROOT / "data/showo2_layout/position_quick8_v2.jsonl",
    )
    parser.add_argument(
        "--source-metadata",
        type=Path,
        default=WORKSPACE_ROOT / "geneval/prompts/evaluation_metadata.jsonl",
    )
    parser.add_argument(
        "--rewritten-metadata",
        type=Path,
        default=SHOWO2_ROOT / "prompts/geneval_meta_data_rewritten.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=WORKSPACE_ROOT / "data/showo2_layout/position_quick8_c0_v2.jsonl",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optionally keep the first N selected prompts for a smoke test.",
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


def relation(metadata: dict[str, Any]) -> str:
    positioned = [item for item in metadata["include"] if "position" in item]
    if len(positioned) != 1:
        raise ValueError(f"Expected one positioned object, got {positioned}")
    value = str(positioned[0]["position"][0])
    if value not in RELATIONS:
        raise ValueError(f"Unsupported Position relation: {value}")
    return value


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    selection = read_jsonl(args.selection_manifest)
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        selection = selection[: args.limit]
    source = read_jsonl(args.source_metadata)
    rewritten = read_jsonl(args.rewritten_metadata)
    if len(source) != len(rewritten):
        raise RuntimeError(
            f"Source/rewritten metadata length mismatch: {len(source)} != "
            f"{len(rewritten)}"
        )

    records = []
    seen_indices = set()
    for local_index, selected in enumerate(selection):
        index = int(selected["source_index"])
        if index in seen_indices:
            raise ValueError(f"Duplicate source index in selection: {index}")
        seen_indices.add(index)
        original = source[index]
        c0 = rewritten[index]
        if original.get("tag") != "position" or c0.get("tag") != "position":
            raise ValueError(f"Source index {index} is not Position metadata")
        if original.get("include") != c0.get("include"):
            raise ValueError(f"C0 rewrite changed scoring constraints at index {index}")
        current_relation = relation(original)
        if relation(c0) != current_relation:
            raise ValueError(f"C0 rewrite changed relation at index {index}")
        if str(selected["relation"]) != current_relation:
            raise ValueError(
                f"Selection relation mismatch at {index}: "
                f"{selected['relation']} != {current_relation}"
            )
        original_prompt = str(original["prompt"]).strip()
        generation_prompt = str(c0["prompt"]).strip()
        if not generation_prompt.lower().lstrip().startswith(original_prompt.lower()):
            raise ValueError(
                f"C0 prompt at {index} does not preserve the original prompt prefix"
            )
        scoring_metadata = dict(c0)
        scoring_metadata["prompt"] = generation_prompt
        records.append(
            {
                "local_index": local_index,
                "source_index": index,
                "relation": current_relation,
                "prompt": original_prompt,
                "generation_prompt": generation_prompt,
                "metadata": scoring_metadata,
                "selection": {
                    "protocol": "showo2-position-c0-parity-v2",
                    "selection_manifest": str(args.selection_manifest.resolve()),
                    "selection_manifest_sha256": hashlib.sha256(
                        args.selection_manifest.read_bytes()
                    ).hexdigest(),
                    "source_metadata_sha256": hashlib.sha256(
                        args.source_metadata.read_bytes()
                    ).hexdigest(),
                    "rewritten_metadata_sha256": hashlib.sha256(
                        args.rewritten_metadata.read_bytes()
                    ).hexdigest(),
                },
            }
        )

    counts = Counter(record["relation"] for record in records)
    write_jsonl(args.output, records)
    print(
        json.dumps(
            {
                "status": "ok",
                "output": str(args.output.resolve()),
                "prompts": len(records),
                "relations": dict(counts),
                "generation_prompts_differ": sum(
                    record["prompt"] != record["generation_prompt"]
                    for record in records
                ),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
