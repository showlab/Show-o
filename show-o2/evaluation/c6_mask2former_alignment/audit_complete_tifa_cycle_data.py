#!/usr/bin/env python3
"""Audit complete paired and unpaired TIFA cycle manifests."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.c6_mask2former_alignment.prepare_tifa_cycle_data import (
    EXPECTED_SOURCE_PROMPTS,
    EXPECTED_SOURCE_QUESTIONS,
    atomic_write,
    normalized,
    question_axis_directions,
    sha256_bytes,
)


SPLITS = ("train", "validation", "test")
SUBSET_PREDICATES: dict[str, Callable[[dict[str, Any]], bool]] = {
    "full_cycle_multidim": lambda row: len(row["dimensions"]) >= 2,
    "full_cycle_spatial": lambda row: row["spatial_question_count"] > 0,
    "full_cycle_directional": lambda row: bool(row["axis_directions"]),
    "full_cycle_unpaired_directional": lambda row: (
        not row["has_reference_image"] and bool(row["axis_directions"])
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root", type=Path)
    return parser.parse_args()


def prompt_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row["sample_id"]), str(row["caption"])


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: row is not an object")
            rows.append(row)
    return rows


def validate_file(
    data_root: Path,
    manifest: dict[str, Any],
    name: str,
) -> list[dict[str, Any]]:
    metadata = manifest["files"][name]
    path = Path(metadata["path"])
    try:
        path.relative_to(data_root.resolve())
    except ValueError as error:
        raise ValueError(f"{name} is outside data root: {path}") from error
    rows = read_jsonl(path)
    content = path.read_bytes()
    if len(rows) != metadata["rows"]:
        raise ValueError(f"{name} row count mismatch")
    if sum(len(row["qa_pairs"]) for row in rows) != metadata["questions"]:
        raise ValueError(f"{name} question count mismatch")
    if sha256_bytes(content) != metadata["sha256"]:
        raise ValueError(f"{name} sha256 mismatch")
    return rows


def main() -> int:
    args = parse_args()
    data_root = args.data_root.resolve()
    manifest_path = data_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    split_rows: dict[str, list[dict[str, Any]]] = {}
    split_keys: dict[str, set[tuple[str, str]]] = {}
    for split in SPLITS:
        rows = validate_file(data_root, manifest, f"full_cycle_{split}")
        keys = {prompt_key(row) for row in rows}
        if len(keys) != len(rows):
            raise ValueError(f"duplicate prompts in full_cycle_{split}")
        split_rows[split] = rows
        split_keys[split] = keys

    overlaps = {
        f"{left}__{right}": len(split_keys[left] & split_keys[right])
        for left_index, left in enumerate(SPLITS)
        for right in SPLITS[left_index + 1 :]
    }
    if any(overlaps.values()):
        raise ValueError(f"TIFA split leakage detected: {overlaps}")

    all_rows = [row for split in SPLITS for row in split_rows[split]]
    source_prompts = Counter(row["source_family"] for row in all_rows)
    source_questions = Counter(
        row["source_family"] for row in all_rows for _ in row["qa_pairs"]
    )
    if dict(source_prompts) != EXPECTED_SOURCE_PROMPTS:
        raise ValueError(
            f"source prompt counts are incomplete: {dict(source_prompts)}"
        )
    if dict(source_questions) != EXPECTED_SOURCE_QUESTIONS:
        raise ValueError(
            f"source question counts are incomplete: {dict(source_questions)}"
        )

    answer_errors = []
    for row in all_rows:
        for question in row["qa_pairs"]:
            target = int(question["target_index"])
            choices = question["choices"]
            if (
                not 0 <= target < len(choices)
                or normalized(choices[target]) != normalized(question["answer"])
            ):
                answer_errors.append((row["sample_id"], question["source_index"]))
    if answer_errors:
        raise ValueError(f"invalid answer targets: {answer_errors[:5]}")

    paired = [row for row in all_rows if row["has_reference_image"]]
    unpaired = [row for row in all_rows if not row["has_reference_image"]]
    missing_references = [
        row["sample_id"]
        for row in paired
        if not Path(row.get("image_path", "")).is_file()
    ]
    if missing_references:
        raise FileNotFoundError(
            f"missing paired references: {missing_references[:5]}"
        )
    if any(row["source_family"] != "coco" for row in paired):
        raise ValueError("non-COCO row was marked as paired")
    if any(row["source_family"] == "coco" for row in unpaired):
        raise ValueError("COCO row was marked as unpaired")
    if len(paired) != EXPECTED_SOURCE_PROMPTS["coco"]:
        raise ValueError(f"expected 2000 paired rows, found {len(paired)}")

    subset_counts: dict[str, dict[str, int]] = {}
    for subset, predicate in SUBSET_PREDICATES.items():
        subset_counts[subset] = {}
        for split in SPLITS:
            rows = validate_file(data_root, manifest, f"{subset}_{split}")
            keys = [prompt_key(row) for row in rows]
            if len(set(keys)) != len(keys):
                raise ValueError(f"duplicate prompts in {subset}_{split}")
            if not set(keys).issubset(split_keys[split]):
                raise ValueError(f"{subset}_{split} is not a full split subset")
            if any(not predicate(row) for row in rows):
                raise ValueError(f"{subset}_{split} violates its predicate")
            expected = sum(predicate(row) for row in split_rows[split])
            if len(rows) != expected:
                raise ValueError(
                    f"{subset}_{split} has {len(rows)} rows, expected {expected}"
                )
            subset_counts[subset][split] = len(rows)

    direction_prompt_counts = Counter(
        direction for row in all_rows for direction in row["axis_directions"]
    )
    direction_question_counts = Counter(
        direction
        for row in all_rows
        for question in row["qa_pairs"]
        for direction in question_axis_directions(question)
    )
    source_spatial_prompts = Counter(
        row["source_family"]
        for row in all_rows
        if row["spatial_question_count"] > 0
    )
    source_directional_prompts = Counter(
        row["source_family"] for row in all_rows if row["axis_directions"]
    )

    report = {
        "status": "data_complete",
        "official_source_complete": True,
        "prompts": len(all_rows),
        "questions": sum(len(row["qa_pairs"]) for row in all_rows),
        "source_prompt_counts": dict(sorted(source_prompts.items())),
        "source_question_counts": dict(sorted(source_questions.items())),
        "paired_reference_prompts": len(paired),
        "unpaired_prompts": len(unpaired),
        "missing_reference_images": len(missing_references),
        "split_counts": {
            split: {
                "prompts": len(split_rows[split]),
                "questions": sum(
                    len(row["qa_pairs"]) for row in split_rows[split]
                ),
            }
            for split in SPLITS
        },
        "split_overlap_counts": overlaps,
        "answer_target_errors": len(answer_errors),
        "subset_counts": subset_counts,
        "axis_direction_prompt_counts": dict(sorted(direction_prompt_counts.items())),
        "axis_direction_question_counts": dict(
            sorted(direction_question_counts.items())
        ),
        "source_spatial_prompt_counts": dict(sorted(source_spatial_prompts.items())),
        "source_directional_prompt_counts": dict(
            sorted(source_directional_prompts.items())
        ),
        "gate_readiness": {
            "paired_coco": "ready_real_image_strict",
            "unpaired": "data_ready_teacher_acceptance_not_built",
        },
        "formal_training_started": False,
    }
    report_path = data_root / "full_cycle_audit.json"
    report_content = (
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    atomic_write(report_path, report_content)

    manifest["full_cycle"]["audit"] = {
        "path": str(report_path),
        "sha256": sha256_bytes(report_content),
        "status": report["status"],
    }
    atomic_write(
        manifest_path,
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
