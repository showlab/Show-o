#!/usr/bin/env python3
"""Audit TIFA self-cycle splits, semantic diversity, and prompt token lengths."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.c6_mask2former_alignment.tifa_cycle import TifaChoiceQuestion
from models.misc import get_text_tokenizer
from utils import path_to_llm_name


def parse_args() -> argparse.Namespace:
    workspace = ROOT.parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=workspace / "data/showo2_self_cycle/tifa_v1",
    )
    parser.add_argument("--family", default="coco_multidim")
    parser.add_argument("--max-text-tokens", type=int, default=192)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
    return rows


def percentile(values: list[int], fraction: float) -> int:
    if not values:
        return 0
    index = round((len(values) - 1) * fraction)
    return sorted(values)[index]


def main() -> int:
    args = parse_args()
    if args.max_text_tokens < 1:
        raise ValueError("max-text-tokens must be positive")
    llm_path = "Qwen/Qwen2.5-1.5B-Instruct"
    tokenizer = get_text_tokenizer(
        llm_path,
        add_showo_tokens=False,
        return_showo_token_ids=False,
        llm_name=path_to_llm_name[llm_path],
    )
    split_rows = {
        split: read_jsonl(args.data_dir / f"{args.family}_{split}.jsonl")
        for split in ("train", "validation", "test")
    }
    split_keys: dict[str, set[tuple[str, str]]] = {}
    for split, rows in split_rows.items():
        keys = {(str(row["sample_id"]), str(row["caption"])) for row in rows}
        if len(keys) != len(rows):
            raise ValueError(f"{split} contains duplicate prompt keys")
        split_keys[split] = keys
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = split_keys[left] & split_keys[right]
        if overlap:
            raise ValueError(f"prompt leakage between {left} and {right}: {len(overlap)}")

    dimension_counts: Counter[str] = Counter()
    option_counts: Counter[int] = Counter()
    target_counts: Counter[int] = Counter()
    dimensions_per_prompt: list[int] = []
    questions_per_prompt: list[int] = []
    token_lengths: list[int] = []
    truncated = []
    caption_leaks = []
    for split, rows in split_rows.items():
        for row in rows:
            dimensions = {str(value).casefold() for value in row["dimensions"]}
            dimensions_per_prompt.append(len(dimensions))
            questions_per_prompt.append(len(row["qa_pairs"]))
            for index, qa in enumerate(row["qa_pairs"]):
                dimension = str(qa["element_type"]).casefold()
                dimension_counts[dimension] += 1
                option_counts[len(qa["choices"])] += 1
                target_counts[int(qa["target_index"])] += 1
                question = TifaChoiceQuestion(
                    name=f"{split}_{row['sample_id']}_{index}",
                    question=str(qa["question"]),
                    choices=tuple(str(value) for value in qa["choices"]),
                    target=int(qa["target_index"]),
                    dimension=dimension,
                )
                length = len(
                    tokenizer(question.statement, add_special_tokens=False)["input_ids"]
                )
                token_lengths.append(length)
                if length > args.max_text_tokens:
                    truncated.append({"sample_id": row["sample_id"], "index": index, "tokens": length})
                caption = " ".join(str(row["caption"]).casefold().split())
                statement = " ".join(question.statement.casefold().split())
                if len(caption) >= 20 and caption in statement:
                    caption_leaks.append({"sample_id": row["sample_id"], "index": index})

    question_total = sum(dimension_counts.values())
    spatial_questions = sum(
        count for name, count in dimension_counts.items() if "spatial" in name
    )
    report = {
        "status": "ready",
        "family": args.family,
        "split_counts": {
            split: {
                "prompts": len(rows),
                "questions": sum(len(row["qa_pairs"]) for row in rows),
            }
            for split, rows in split_rows.items()
        },
        "semantic_dimensions": dict(sorted(dimension_counts.items())),
        "semantic_dimension_count": len(dimension_counts),
        "spatial_question_fraction": spatial_questions / question_total,
        "option_counts": dict(sorted(option_counts.items())),
        "target_position_counts": dict(sorted(target_counts.items())),
        "dimensions_per_prompt": {
            "min": min(dimensions_per_prompt),
            "mean": statistics.mean(dimensions_per_prompt),
            "max": max(dimensions_per_prompt),
        },
        "questions_per_prompt": {
            "min": min(questions_per_prompt),
            "mean": statistics.mean(questions_per_prompt),
            "max": max(questions_per_prompt),
        },
        "prompt_token_lengths": {
            "max_text_tokens": args.max_text_tokens,
            "p50": percentile(token_lengths, 0.50),
            "p95": percentile(token_lengths, 0.95),
            "p99": percentile(token_lengths, 0.99),
            "max": max(token_lengths),
            "over_limit": len(truncated),
        },
        "caption_leak_count": len(caption_leaks),
        "leakage": "none",
    }
    if len(dimension_counts) < 6:
        raise RuntimeError(f"insufficient TIFA semantic diversity: {dimension_counts}")
    if report["spatial_question_fraction"] > 0.35:
        raise RuntimeError("TIFA cycle data is still dominated by spatial questions")
    if min(dimensions_per_prompt) < 2:
        raise RuntimeError("multidimensional family contains a single-dimension prompt")
    if truncated:
        report["status"] = "not_ready_prompt_truncation"
    if caption_leaks:
        report["status"] = "not_ready_caption_leak"
    output = args.output or args.data_dir / f"{args.family}_audit.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
