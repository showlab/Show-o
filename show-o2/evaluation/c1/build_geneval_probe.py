#!/usr/bin/env python3
"""Build a deterministic, stratified GenEval manifest for gradient probing."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


DEFAULT_TAGS = (
    "single_object",
    "two_object",
    "counting",
    "colors",
    "position",
    "color_attr",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--base-prompts", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-tag", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--tags", nargs="+", default=list(DEFAULT_TAGS))
    return parser.parse_args()


def prompt_id(row: dict) -> str:
    return Path(row["filename"]).parent.parent.name


def sample_bucket(rows: list[dict], count: int, rng: random.Random, used_prompts: set[str]) -> list[dict]:
    candidates = rows[:]
    rng.shuffle(candidates)
    unique = [row for row in candidates if prompt_id(row) not in used_prompts]
    repeated = [row for row in candidates if prompt_id(row) in used_prompts]
    selected = (unique + repeated)[:count]
    used_prompts.update(prompt_id(row) for row in selected)
    return selected


def select_tag(rows: list[dict], count: int, rng: random.Random) -> list[dict]:
    by_correct = defaultdict(list)
    for row in rows:
        by_correct[bool(row["correct"])].append(row)

    used_prompts: set[str] = set()
    false_target = count // 2
    selected = sample_bucket(by_correct[False], min(false_target, len(by_correct[False])), rng, used_prompts)
    selected += sample_bucket(by_correct[True], min(count - len(selected), len(by_correct[True])), rng, used_prompts)

    if len(selected) < count:
        selected_names = {row["filename"] for row in selected}
        remaining = [row for row in rows if row["filename"] not in selected_names]
        selected += sample_bucket(remaining, count - len(selected), rng, used_prompts)
    rng.shuffle(selected)
    return selected


def main() -> None:
    args = parse_args()
    if args.per_tag < 1:
        raise ValueError("--per-tag must be positive")
    if not args.results.is_file():
        raise FileNotFoundError(args.results)

    base_prompts = {}
    if args.base_prompts is not None:
        if not args.base_prompts.is_file():
            raise FileNotFoundError(args.base_prompts)
        with args.base_prompts.open("r", encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                if line.strip():
                    base_prompts[f"{index:05d}"] = json.loads(line)["prompt"]

    rows_by_tag = defaultdict(list)
    with args.results.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("tag") not in args.tags:
                continue
            image_path = Path(row["filename"])
            if not image_path.is_file():
                raise FileNotFoundError(f"Missing GenEval image at line {line_number}: {image_path}")
            rows_by_tag[row["tag"]].append(row)

    missing = [tag for tag in args.tags if not rows_by_tag[tag]]
    if missing:
        raise RuntimeError(f"No scored GenEval rows for tags: {', '.join(missing)}")

    rng = random.Random(args.seed)
    selected_by_tag = {
        tag: select_tag(rows_by_tag[tag], args.per_tag, rng)
        for tag in args.tags
    }

    interleaved = []
    for offset in range(args.per_tag):
        for tag in args.tags:
            if offset < len(selected_by_tag[tag]):
                interleaved.append(selected_by_tag[tag][offset])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(interleaved):
            identifier = prompt_id(row)
            metadata = json.loads(row.get("metadata") or "{}")
            item = {
                "index": index,
                "sample_id": f"{prompt_id(row)}-{Path(row['filename']).stem}",
                "tag": row["tag"],
                "correct": bool(row["correct"]),
                "reason": row.get("reason", ""),
                "image": str(Path(row["filename"]).resolve()),
                "prompt": row["prompt"],
                "base_prompt": base_prompts.get(identifier, metadata.get("prompt", row["prompt"])),
            }
            handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")

    counts = defaultdict(lambda: {"correct": 0, "incorrect": 0})
    for row in interleaved:
        key = "correct" if row["correct"] else "incorrect"
        counts[row["tag"]][key] += 1
    print(json.dumps({"output": str(args.output), "samples": len(interleaved), "counts": counts}, default=dict))


if __name__ == "__main__":
    main()
