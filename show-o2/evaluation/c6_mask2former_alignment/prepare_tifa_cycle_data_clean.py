#!/usr/bin/env python3
"""Prepare TIFA cycle data with an explicit audit of upstream choice defects."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from evaluation.c6_mask2former_alignment import prepare_tifa_cycle_data as base


STRICT_VALIDATE = base.validate_question
CLEANING_ROWS: list[dict[str, Any]] = []


def clean_validate_question(raw: Any, index: int) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return STRICT_VALIDATE(raw, index)
    cleaned = dict(raw)
    original = list(raw.get("choices", []))
    choices = []
    seen = set()
    for choice in original:
        key = base.normalized(choice)
        if key and key not in seen:
            choices.append(choice)
            seen.add(key)
    answer_key = base.normalized(raw.get("answer", ""))
    operations = []
    if len(choices) != len(original):
        operations.append("deduplicate_choices_keep_first")
    if answer_key and answer_key not in {base.normalized(value) for value in choices}:
        if len(choices) < 4:
            choices.append(raw["answer"])
            operations.append("append_missing_answer")
        else:
            choices[-1] = raw["answer"]
            operations.append("replace_last_distractor_with_missing_answer")
    cleaned["choices"] = choices
    result = STRICT_VALIDATE(cleaned, index)
    if operations:
        CLEANING_ROWS.append({
            "source_index": index,
            "sample_id": raw.get("id"),
            "question": raw.get("question"),
            "answer": raw.get("answer"),
            "operations": operations,
            "original_choices": original,
            "cleaned_choices": choices,
        })
    return result


def output_dir_from_argv() -> Path:
    if "--output-dir" in sys.argv:
        return Path(sys.argv[sys.argv.index("--output-dir") + 1]).expanduser()
    return Path(__file__).resolve().parents[4] / "data/showo2_self_cycle/tifa_v1"


if __name__ == "__main__":
    CLEANING_ROWS.clear()
    base.validate_question = clean_validate_question
    result = base.main()
    output_dir = output_dir_from_argv()
    report = {
        "status": "completed",
        "policy": {
            "duplicate_choices": "keep first normalized occurrence",
            "missing_answer_under_four_choices": "append official answer",
            "missing_answer_at_four_choices": "replace final distractor with official answer",
            "unchanged_fields": ["id", "caption", "question", "answer", "element_type", "element"],
        },
        "changed_questions": len(CLEANING_ROWS),
        "operation_counts": dict(sorted(Counter(
            operation for row in CLEANING_ROWS for operation in row["operations"]
        ).items())),
        "changes": CLEANING_ROWS,
    }
    base.atomic_write(
        output_dir / "cleaning_report.json",
        (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    raise SystemExit(result)
