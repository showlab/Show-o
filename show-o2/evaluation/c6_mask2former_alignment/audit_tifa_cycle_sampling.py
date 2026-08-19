#!/usr/bin/env python3
"""Validate the final filtered and option-balanced TIFA sampling protocol."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from evaluation.c6_mask2former_alignment.tifa_cycle_sampling import (
    filter_caption_copy_questions,
    prepare_cycle_questions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--base-seed", type=int, default=20260816)
    parser.add_argument("--max-questions", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    positions: Counter[tuple[int, int]] = Counter()
    dimensions: Counter[str] = Counter()
    split_counts = {}
    removed_questions = 0
    sampled_questions = 0
    global_step = 0
    for split in ("train", "validation", "test"):
        rows = [
            json.loads(line)
            for line in open(
                args.data_dir / f"coco_multidim_{split}.jsonl", encoding="utf-8"
            )
        ]
        split_sampled = 0
        for row in rows:
            global_step += 1
            filtered = filter_caption_copy_questions(
                row["caption"], row["qa_pairs"]
            )
            removed_questions += len(row["qa_pairs"]) - len(filtered)
            selected = prepare_cycle_questions(
                row,
                max_questions=args.max_questions,
                base_seed=args.base_seed,
                cycle_step=global_step,
            )
            repeated = prepare_cycle_questions(
                row,
                max_questions=args.max_questions,
                base_seed=args.base_seed,
                cycle_step=global_step,
            )
            if selected != repeated:
                raise RuntimeError(f"non-deterministic question sampling: {row['sample_id']}")
            if len({qa["element_type"] for qa in selected}) < 2:
                raise RuntimeError(f"single-dimensional sample: {row['sample_id']}")
            for qa in selected:
                positions[(len(qa["choices"]), int(qa["target_index"]))] += 1
                dimensions[str(qa["element_type"])] += 1
            split_sampled += len(selected)
        sampled_questions += split_sampled
        split_counts[split] = {"prompts": len(rows), "sampled_questions": split_sampled}

    def fractions(choice_count: int) -> list[float]:
        values = [positions[(choice_count, index)] for index in range(choice_count)]
        total = sum(values)
        return [value / total for value in values] if total else []

    two_choice = fractions(2)
    four_choice = fractions(4)
    position_balanced = (
        all(0.45 <= value <= 0.55 for value in two_choice)
        and all(0.20 <= value <= 0.30 for value in four_choice)
    )
    spatial = sum(
        count for name, count in dimensions.items() if "spatial" in name
    )
    report = {
        "status": "ready" if position_balanced else "not_ready_position_bias",
        "protocol": "tifa-coco-multidim-filtered-shuffled-v1",
        "base_seed": args.base_seed,
        "max_questions_per_prompt": args.max_questions,
        "split_counts": split_counts,
        "sampled_questions": sampled_questions,
        "caption_copy_questions_filtered": removed_questions,
        "semantic_dimension_count": len(dimensions),
        "semantic_dimensions": dict(sorted(dimensions.items())),
        "spatial_question_fraction": spatial / sampled_questions,
        "target_position_counts": {
            f"{count}_choice_target_{target}": value
            for (count, target), value in sorted(positions.items())
        },
        "target_position_fractions": {
            "2_choice": two_choice,
            "4_choice": four_choice,
        },
        "route_pairing": (
            "seed depends only on base_seed, sample_id, and cycle_step; "
            "G-adapter and LoRA receive identical questions and permutations"
        ),
        "understanding_input": "predicted_image+question+choices_only",
        "generation_caption_in_u": False,
    }
    output = args.data_dir / "coco_multidim_sampling_audit.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
