#!/usr/bin/env python3
"""Build a balanced, C0-anchored Position manifest for the quick A/B test."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


SHOWO2_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = SHOWO2_ROOT.parents[1]
SELECTED_SOURCE_INDICES = (363, 378, 365, 420, 367, 390, 371, 416)
EXPECTED_RELATIONS = {
    363: "left of",
    378: "left of",
    365: "right of",
    420: "right of",
    367: "above",
    390: "above",
    371: "below",
    416: "below",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-metadata",
        type=Path,
        default=WORKSPACE_ROOT / "geneval/prompts/evaluation_metadata.jsonl",
    )
    parser.add_argument(
        "--c0-results",
        type=Path,
        default=(
            WORKSPACE_ROOT
            / "runs/showo2_c0/c0-full-20260802T125630Z/generation/geneval/results/results.jsonl"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=WORKSPACE_ROOT / "data/showo2_layout/position_quick8_v2.jsonl",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def source_index(result: dict) -> int:
    return int(Path(result["filename"]).parents[1].name)


def position_relation(metadata: dict) -> str:
    positioned = [item for item in metadata["include"] if "position" in item]
    if len(positioned) != 1:
        raise ValueError(f"Expected one positioned object, got {positioned}")
    return str(positioned[0]["position"][0])


def atomic_write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    source = read_jsonl(args.source_metadata)
    c0_by_index: dict[int, list[dict]] = defaultdict(list)
    for result in read_jsonl(args.c0_results):
        c0_by_index[source_index(result)].append(result)

    records = []
    for local_index, index in enumerate(SELECTED_SOURCE_INDICES):
        metadata = dict(source[index])
        if metadata.get("tag") != "position":
            raise ValueError(f"Source index {index} is not a Position prompt")
        relation = position_relation(metadata)
        if relation != EXPECTED_RELATIONS[index]:
            raise ValueError(f"Unexpected relation for {index}: {relation}")

        results = c0_by_index[index]
        if len(results) != 4:
            raise ValueError(f"Expected four C0 images for {index}, got {len(results)}")
        rewritten_prompts = {item["prompt"] for item in results}
        if len(rewritten_prompts) != 1:
            raise ValueError(f"C0 prompt is not stable for source index {index}")
        prompt = rewritten_prompts.pop().strip()
        first_sentence = prompt.split(".", 1)[0].lower()
        if relation not in first_sentence:
            raise ValueError(
                f"Rewritten prompt {index} does not state {relation!r} in its first sentence"
            )

        c0_correct = sum(bool(item["correct"]) for item in results)
        if not 0 < c0_correct < 4:
            raise ValueError(
                f"Source index {index} must be a moderate C0 case, got {c0_correct}/4"
            )
        metadata["prompt"] = prompt
        records.append(
            {
                "local_index": local_index,
                "source_index": index,
                "relation": relation,
                "c0_correct": c0_correct,
                "c0_total": len(results),
                "original_prompt": source[index]["prompt"],
                "prompt": prompt,
                "metadata": metadata,
            }
        )

    relation_counts = Counter(item["relation"] for item in records)
    expected_counts = Counter({relation: 2 for relation in ("left of", "right of", "above", "below")})
    if relation_counts != expected_counts:
        raise ValueError(f"Unbalanced relation set: {relation_counts}")

    atomic_write_jsonl(args.output, records)
    print(json.dumps({
        "status": "ok",
        "output": str(args.output.resolve()),
        "prompts": len(records),
        "relations": relation_counts,
        "c0_correct": sum(item["c0_correct"] for item in records),
        "c0_total": sum(item["c0_total"] for item in records),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
