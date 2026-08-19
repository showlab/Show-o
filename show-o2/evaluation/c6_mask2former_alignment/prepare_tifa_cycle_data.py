#!/usr/bin/env python3
"""Download, validate, group, and split TIFA v1.0 for self-cycle training."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
import urllib.request
from collections import Counter, OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


UPSTREAM_REPOSITORY = "https://github.com/Yushi-Hu/tifa"
UPSTREAM_COMMIT = "8e37050f44ab4ee681e4251f3c8a9d4cc86e901e"
SOURCE_RELATIVE_PATH = "tifa_v1.0/tifa_v1.0_question_answers.json"
SOURCE_BYTES = 9_630_428
SOURCE_GIT_BLOB_SHA1 = "6a563f9c157e5b2a24b9f4936edd3934a8a70add"
SPLIT_THRESHOLDS = (("train", 9000), ("validation", 9500), ("test", 10000))
SOURCE_FAMILIES = ("coco", "drawbench", "paintskill", "partiprompt")
AXIS_DIRECTION_PATTERNS = {
    "left": re.compile(r"\bleft\b"),
    "right": re.compile(r"\bright\b"),
    "above": re.compile(r"\b(?:above|over|top|up)\b"),
    "below": re.compile(r"\b(?:below|under|underneath|beneath|bottom|down)\b"),
}
EXPECTED_SOURCE_PROMPTS = {
    "coco": 2000,
    "drawbench": 161,
    "paintskill": 500,
    "partiprompt": 1412,
}
EXPECTED_SOURCE_QUESTIONS = {
    "coco": 13735,
    "drawbench": 1011,
    "paintskill": 2920,
    "partiprompt": 8163,
}


def parse_args() -> argparse.Namespace:
    workspace = Path(__file__).resolve().parents[4]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=workspace / "data/showo2_self_cycle/tifa_v1",
    )
    parser.add_argument(
        "--coco-root",
        type=Path,
        default=workspace / "datasets/COCO2017/val2017",
    )
    parser.add_argument(
        "--source-json",
        type=Path,
        help="Use an existing official JSON after applying the same pinned validation.",
    )
    parser.add_argument("--force-download", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def git_blob_sha1(content: bytes) -> str:
    header = f"blob {len(content)}\0".encode("ascii")
    return hashlib.sha1(header + content).hexdigest()


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def validate_source_bytes(content: bytes) -> list[dict[str, Any]]:
    if len(content) != SOURCE_BYTES:
        raise ValueError(f"TIFA source has {len(content)} bytes, expected {SOURCE_BYTES}")
    actual_blob = git_blob_sha1(content)
    if actual_blob != SOURCE_GIT_BLOB_SHA1:
        raise ValueError(
            f"TIFA Git blob mismatch: {actual_blob} != {SOURCE_GIT_BLOB_SHA1}"
        )
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as error:
        raise ValueError(f"TIFA source is invalid JSON: {error}") from error
    if not isinstance(payload, list) or not payload:
        raise ValueError("TIFA source must be a non-empty JSON list")
    return payload


def download_source() -> bytes:
    url = (
        "https://raw.githubusercontent.com/Yushi-Hu/tifa/"
        f"{UPSTREAM_COMMIT}/{SOURCE_RELATIVE_PATH}"
    )
    last_error: Exception | None = None
    for attempt in range(1, 6):
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "showo2-tifa-self-cycle", "Accept-Encoding": "identity"},
        )
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                content = response.read()
            validate_source_bytes(content)
            return content
        except Exception as error:  # Network and integrity failures both retry.
            last_error = error
            if attempt < 5:
                time.sleep(min(2 ** attempt, 16))
    raise RuntimeError(f"could not download the pinned TIFA source: {last_error}")


def atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(content)
    os.replace(temporary, path)


def normalized(value: Any) -> str:
    return " ".join(str(value).strip().casefold().split())


def source_family(sample_id: str) -> str:
    family = str(sample_id).strip().casefold().split("_", 1)[0]
    if family not in SOURCE_FAMILIES:
        raise ValueError(f"unsupported TIFA source family: {sample_id!r}")
    return family


def question_axis_directions(question: dict[str, Any]) -> tuple[str, ...]:
    if str(question.get("element_type", "")).strip().casefold() != "spatial":
        return ()
    element = normalized(question.get("element", ""))
    return tuple(
        direction
        for direction, pattern in AXIS_DIRECTION_PATTERNS.items()
        if pattern.search(element)
    )


def record_axis_directions(record: dict[str, Any]) -> tuple[str, ...]:
    directions = {
        direction
        for question in record["qa_pairs"]
        for direction in question_axis_directions(question)
    }
    return tuple(
        direction for direction in AXIS_DIRECTION_PATTERNS if direction in directions
    )


def validate_question(raw: Any, index: int) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError(f"TIFA question {index} is not an object")
    required = ("id", "caption", "question", "choices", "answer", "element_type")
    missing = [name for name in required if raw.get(name) in (None, "")]
    if missing:
        raise ValueError(f"TIFA question {index} is missing {missing}")
    choices = raw["choices"]
    if not isinstance(choices, list) or not 2 <= len(choices) <= 4:
        raise ValueError(f"TIFA question {index} must have 2-4 choices")
    choice_keys = [normalized(choice) for choice in choices]
    if any(not value for value in choice_keys) or len(set(choice_keys)) != len(choice_keys):
        raise ValueError(f"TIFA question {index} has empty or duplicate choices")
    answer_key = normalized(raw["answer"])
    matches = [offset for offset, value in enumerate(choice_keys) if value == answer_key]
    if len(matches) != 1:
        raise ValueError(
            f"TIFA question {index} answer {raw['answer']!r} does not match one choice"
        )
    return {
        "question": str(raw["question"]).strip(),
        "choices": [str(choice).strip() for choice in choices],
        "answer": str(raw["answer"]).strip(),
        "target_index": matches[0],
        "element_type": str(raw["element_type"]).strip().casefold(),
        "element": str(raw.get("element", "")).strip(),
        "source_index": index,
    }


def grouped_records(payload: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: OrderedDict[tuple[str, str], dict[str, Any]] = OrderedDict()
    for index, raw in enumerate(payload):
        question = validate_question(raw, index)
        sample_id = str(raw["id"]).strip()
        caption = str(raw["caption"]).strip()
        key = sample_id, caption
        record = groups.setdefault(
            key,
            {
                "sample_id": sample_id,
                "caption": caption,
                "source_family": source_family(sample_id),
                "coco_val_id": None,
                "qa_pairs": [],
            },
        )
        coco_val_id = raw.get("coco_val_id")
        if coco_val_id not in (None, ""):
            coco_value = str(coco_val_id).strip()
            existing = record["coco_val_id"]
            if existing not in (None, coco_value):
                raise ValueError(f"TIFA prompt {sample_id} has inconsistent COCO ids")
            record["coco_val_id"] = coco_value
        record["qa_pairs"].append(question)
    records = []
    for record in groups.values():
        dimensions = sorted({qa["element_type"] for qa in record["qa_pairs"]})
        record["dimensions"] = dimensions
        record["spatial_question_count"] = sum(
            qa["element_type"] == "spatial" for qa in record["qa_pairs"]
        )
        record["axis_directions"] = list(record_axis_directions(record))
        record["question_count"] = len(record["qa_pairs"])
        records.append(record)
    return records


def split_name(sample_id: str, caption: str) -> str:
    digest = hashlib.sha256(f"{sample_id}\0{caption}".encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") % 10_000
    for name, threshold in SPLIT_THRESHOLDS:
        if bucket < threshold:
            return name
    raise AssertionError(bucket)


def attach_coco_image(record: dict[str, Any], coco_root: Path) -> bool:
    raw_id = record.get("coco_val_id")
    if raw_id in (None, ""):
        return False
    try:
        filename = f"{int(str(raw_id)):012d}.jpg"
    except ValueError:
        return False
    image_path = coco_root / filename
    if not image_path.is_file():
        return False
    record["image_path"] = str(image_path.resolve())
    return True


def annotate_cycle_record(
    record: dict[str, Any], coco_root: Path
) -> dict[str, Any]:
    copied = json.loads(json.dumps(record))
    has_reference = attach_coco_image(copied, coco_root)
    family = copied["source_family"]
    if family == "coco" and not has_reference:
        raise FileNotFoundError(
            f"paired TIFA COCO reference is missing for {copied['sample_id']}"
        )
    if family != "coco" and has_reference:
        raise ValueError(
            f"unpaired TIFA source unexpectedly has an image: {copied['sample_id']}"
        )
    copied["has_reference_image"] = has_reference
    copied["gate_policy"] = (
        "real_image_strict" if has_reference else "unpaired_answer_target"
    )
    copied["teacher_acceptance_status"] = (
        "not_required" if has_reference else "not_built"
    )
    copied["generation_start_policy"] = (
        "reference_flow_path"
        if has_reference
        else "noise_or_detached_prefix_required"
    )
    return copied


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(records)
    content = b"".join(
        (json.dumps(row, sort_keys=True) + "\n").encode("utf-8") for row in rows
    )
    atomic_write(path, content)
    dimensions = Counter(
        qa["element_type"] for row in rows for qa in row["qa_pairs"]
    )
    sources = Counter(row.get("source_family", "unknown") for row in rows)
    gates = Counter(
        row["gate_policy"] for row in rows if row.get("gate_policy")
    )
    direction_prompts = Counter(
        direction for row in rows for direction in row.get("axis_directions", [])
    )
    direction_questions = Counter(
        direction
        for row in rows
        for qa in row["qa_pairs"]
        for direction in question_axis_directions(qa)
    )
    return {
        "path": str(path.resolve()),
        "rows": len(rows),
        "questions": sum(len(row["qa_pairs"]) for row in rows),
        "bytes": len(content),
        "sha256": sha256_bytes(content),
        "dimension_counts": dict(sorted(dimensions.items())),
        "source_counts": dict(sorted(sources.items())),
        "gate_policy_counts": dict(sorted(gates.items())),
        "axis_direction_prompt_counts": dict(sorted(direction_prompts.items())),
        "axis_direction_question_counts": dict(sorted(direction_questions.items())),
    }


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_path = args.output_dir / "source/tifa_v1.0_question_answers.json"
    if args.source_json is not None:
        content = args.source_json.expanduser().read_bytes()
        source_kind = "provided"
    elif source_path.is_file() and not args.force_download:
        content = source_path.read_bytes()
        source_kind = "existing"
    else:
        content = download_source()
        source_kind = "downloaded"
    payload = validate_source_bytes(content)
    atomic_write(source_path, content)

    records = grouped_records(payload)
    split_rows: dict[str, list[dict[str, Any]]] = {
        name: [] for name, _ in SPLIT_THRESHOLDS
    }
    for record in records:
        split_rows[split_name(record["sample_id"], record["caption"])].append(record)

    files: dict[str, Any] = {}
    coco_available = 0
    multidim_available = 0
    for split, rows in split_rows.items():
        files[f"all_{split}"] = write_jsonl(
            args.output_dir / f"all_{split}.jsonl", rows
        )
        cycle_rows = [
            annotate_cycle_record(row, args.coco_root) for row in rows
        ]
        multidim_cycle_rows = [
            row for row in cycle_rows if len(row["dimensions"]) >= 2
        ]
        spatial_cycle_rows = [
            row for row in cycle_rows if row["spatial_question_count"] > 0
        ]
        directional_cycle_rows = [
            row for row in cycle_rows if row["axis_directions"]
        ]
        unpaired_directional_rows = [
            row for row in cycle_rows
            if not row["has_reference_image"] and row["axis_directions"]
        ]
        files[f"full_cycle_{split}"] = write_jsonl(
            args.output_dir / f"full_cycle_{split}.jsonl", cycle_rows
        )
        files[f"full_cycle_multidim_{split}"] = write_jsonl(
            args.output_dir / f"full_cycle_multidim_{split}.jsonl", multidim_cycle_rows
        )
        files[f"full_cycle_spatial_{split}"] = write_jsonl(
            args.output_dir / f"full_cycle_spatial_{split}.jsonl", spatial_cycle_rows
        )
        files[f"full_cycle_directional_{split}"] = write_jsonl(
            args.output_dir / f"full_cycle_directional_{split}.jsonl", directional_cycle_rows
        )
        files[f"full_cycle_unpaired_directional_{split}"] = write_jsonl(
            args.output_dir / f"full_cycle_unpaired_directional_{split}.jsonl", unpaired_directional_rows
        )
        coco_rows = []
        for row in rows:
            copied = json.loads(json.dumps(row))
            if attach_coco_image(copied, args.coco_root):
                coco_rows.append(copied)
        multidim_rows = [row for row in coco_rows if len(row["dimensions"]) >= 2]
        coco_available += len(coco_rows)
        multidim_available += len(multidim_rows)
        files[f"coco_{split}"] = write_jsonl(
            args.output_dir / f"coco_{split}.jsonl", coco_rows
        )
        files[f"coco_multidim_{split}"] = write_jsonl(
            args.output_dir / f"coco_multidim_{split}.jsonl", multidim_rows
        )

    source_dimensions = Counter(qa["element_type"] for row in records for qa in row["qa_pairs"])
    manifest = {
        "created_at": utc_now(),
        "status": "ready",
        "source": {
            "repository": UPSTREAM_REPOSITORY,
            "commit": UPSTREAM_COMMIT,
            "relative_path": SOURCE_RELATIVE_PATH,
            "path": str(source_path.resolve()),
            "source_kind": source_kind,
            "bytes": len(content),
            "sha256": sha256_bytes(content),
            "git_blob_sha1": git_blob_sha1(content),
            "license": "Apache-2.0",
            "questions": len(payload),
            "prompts": len(records),
            "dimension_counts": dict(sorted(source_dimensions.items())),
        },
        "split_policy": {
            "unit": "sample_id+caption",
            "method": "sha256 bucket",
            "thresholds": {name: threshold for name, threshold in SPLIT_THRESHOLDS},
        },
        "coco": {
            "root": str(args.coco_root.resolve()),
            "available_prompts": coco_available,
            "multidimensional_prompts": multidim_available,
        },
        "full_cycle": {
            "status": "data_complete",
            "prompts": len(records),
            "questions": len(payload),
            "paired_reference_prompts": coco_available,
            "unpaired_prompts": len(records) - coco_available,
            "gate_policies": {
                "paired": "real_image_strict",
                "unpaired": "unpaired_answer_target",
            },
            "unpaired_teacher_acceptance": {
                "status": "not_built",
                "required_for_strict_gate": True,
            },
            "subsets": [
                "full_cycle",
                "full_cycle_multidim",
                "full_cycle_spatial",
                "full_cycle_directional",
                "full_cycle_unpaired_directional",
            ],
        },
        "files": files,
        "training_note": (
            "Training on these official TIFA prompts precludes reporting the official "
            "TIFA v1.0 benchmark as an unseen test; use the held-out split or GQA."
        ),
    }
    manifest_path = args.output_dir / "manifest.json"
    atomic_write(
        manifest_path,
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    print(json.dumps({
        "status": "ready",
        "manifest": str(manifest_path.resolve()),
        "questions": len(payload),
        "prompts": len(records),
        "coco_prompts": coco_available,
        "coco_multidim_prompts": multidim_available,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
