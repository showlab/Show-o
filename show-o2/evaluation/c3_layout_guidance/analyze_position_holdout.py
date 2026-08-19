#!/usr/bin/env python3
"""Decompose paired GenEval Position failures using saved detector boxes."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

RELATIONS = ("left of", "right of", "above", "below")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--position-threshold", type=float, default=0.1)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def result_key(row: dict[str, Any]) -> tuple[int, int]:
    path = Path(row["filename"])
    return int(path.parents[1].name), int(path.stem)


def load_results(run_dir: Path, variant: str) -> dict[tuple[int, int], dict]:
    path = run_dir / variant / "results/results.jsonl"
    return {result_key(row): row for row in read_jsonl(path)}


def position_requirements(metadata: dict[str, Any]) -> tuple[dict, dict, str]:
    includes = metadata["include"]
    positioned = [
        (index, item) for index, item in enumerate(includes) if "position" in item
    ]
    if len(positioned) != 1:
        raise ValueError(f"Expected one positioned object, got {positioned}")
    _, positioned_req = positioned[0]
    relation, target_index = positioned_req["position"]
    return positioned_req, includes[int(target_index)], str(relation)


def top_box(details: dict[str, list], class_name: str) -> list[float] | None:
    boxes = details.get(class_name, [])
    return boxes[0] if boxes else None


def official_geometry(
    positioned_box: list[float],
    reference_box: list[float],
    position_threshold: float,
) -> dict[str, float]:
    boxes = (positioned_box, reference_box)
    centers = [((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0) for box in boxes]
    dimensions = [(abs(box[2] - box[0]), abs(box[3] - box[1])) for box in boxes]
    raw_dx = centers[0][0] - centers[1][0]
    raw_dy = centers[0][1] - centers[1][1]
    raw = (raw_dx, raw_dy)
    revised = []
    for axis in range(2):
        value = raw[axis]
        tolerance = position_threshold * (dimensions[0][axis] + dimensions[1][axis])
        magnitude = max(abs(value) - tolerance, 0.0)
        revised.append(math.copysign(magnitude, value) if value else 0.0)
    norm = math.hypot(raw_dx, raw_dy)
    normalized = (revised[0] / norm, revised[1] / norm) if norm else (0.0, 0.0)
    return {
        "raw_dx": raw_dx,
        "raw_dy": raw_dy,
        "revised_dx": revised[0],
        "revised_dy": revised[1],
        "normalized_dx": normalized[0],
        "normalized_dy": normalized[1],
        "positioned_confidence": float(positioned_box[4]),
        "reference_confidence": float(reference_box[4]),
    }


def expected_values(relation: str, geometry: dict[str, float]) -> dict[str, float]:
    axis = 0 if relation in ("left of", "right of") else 1
    sign = -1.0 if relation in ("left of", "above") else 1.0
    raw = geometry["raw_dx"] if axis == 0 else geometry["raw_dy"]
    revised = geometry["revised_dx"] if axis == 0 else geometry["revised_dy"]
    normalized = geometry["normalized_dx"] if axis == 0 else geometry["normalized_dy"]
    other = geometry["normalized_dy"] if axis == 0 else geometry["normalized_dx"]
    return {
        "expected_raw_gap": sign * raw,
        "expected_revised_gap": sign * revised,
        "expected_component": sign * normalized,
        "other_component_abs": abs(other),
    }


def classify(
    row: dict[str, Any],
    metadata: dict[str, Any],
    position_threshold: float,
) -> tuple[str, dict[str, float] | None]:
    positioned_req, reference_req, relation = position_requirements(metadata)
    details = json.loads(row["details"])
    positioned_box = top_box(details, positioned_req["class"])
    reference_box = top_box(details, reference_req["class"])
    if positioned_box is None and reference_box is None:
        return "both_missing", None
    if positioned_box is None:
        return "positioned_missing", None
    if reference_box is None:
        return "reference_missing", None

    geometry = official_geometry(positioned_box, reference_box, position_threshold)
    geometry.update(expected_values(relation, geometry))
    if row["correct"]:
        return "correct", geometry
    if abs(geometry["revised_dx"]) < 1e-3 and abs(geometry["revised_dy"]) < 1e-3:
        return "overlap_or_too_close", geometry
    if geometry["expected_raw_gap"] <= 0.0:
        return "reversed_expected_axis", geometry
    if geometry["expected_revised_gap"] <= 1e-3:
        return "expected_gap_removed_by_size_tolerance", geometry
    if geometry["expected_component"] <= 0.5 and geometry["other_component_abs"] > 0.5:
        return "wrong_axis_dominates", geometry
    if geometry["expected_component"] <= 0.5:
        return "expected_axis_not_dominant", geometry
    return "other_geometry", geometry


def summarize(
    manifest: list[dict[str, Any]],
    results: dict[tuple[int, int], dict],
    position_threshold: float,
) -> dict[str, Any]:
    overall: Counter[str] = Counter()
    by_relation: dict[str, Counter[str]] = defaultdict(Counter)
    by_prompt: dict[int, Counter[str]] = defaultdict(Counter)
    rows = []
    for record in manifest:
        local_index = int(record["local_index"])
        for sample_index in range(4):
            row = results[(local_index, sample_index)]
            category, geometry = classify(row, record["metadata"], position_threshold)
            overall[category] += 1
            by_relation[record["relation"]][category] += 1
            by_prompt[local_index][category] += 1
            rows.append(
                {
                    "local_index": local_index,
                    "source_index": int(record["source_index"]),
                    "relation": record["relation"],
                    "prompt": record["prompt"],
                    "sample_index": sample_index,
                    "correct": bool(row["correct"]),
                    "category": category,
                    "reason": row["reason"],
                    "geometry": geometry,
                }
            )
    return {
        "overall": dict(overall),
        "by_relation": {
            relation: dict(by_relation[relation]) for relation in RELATIONS
        },
        "by_prompt": [
            {
                "local_index": int(record["local_index"]),
                "source_index": int(record["source_index"]),
                "relation": record["relation"],
                "prompt": record["prompt"],
                "categories": dict(by_prompt[int(record["local_index"])]),
            }
            for record in manifest
        ],
        "rows": rows,
    }


def load_guidance_summary(run_dir: Path, manifest: list[dict]) -> list[dict]:
    summaries = []
    for record in manifest:
        local_index = int(record["local_index"])
        paths = sorted(
            (run_dir / "guided/raw" / f"{local_index:05d}").glob("diagnostics-*.json")
        )
        if len(paths) != 1:
            raise ValueError(
                f"Expected one diagnostics file for prompt {local_index}, got {paths}"
            )
        diagnostics = json.loads(paths[0].read_text(encoding="utf-8"))
        guided_steps = [step for step in diagnostics if step["guidance"]]
        records = [item for step in guided_steps for item in step["guidance"]]
        first = guided_steps[0]["guidance"][0]
        last = guided_steps[-1]["guidance"][-1]
        first_by_sample = {
            int(item["batch_index"]): item
            for item in first["objects"]
            if int(item["object_index"]) == 0
        }
        last_by_sample = {
            int(item["batch_index"]): item
            for item in last["objects"]
            if int(item["object_index"]) == 0
        }
        summaries.append(
            {
                "local_index": local_index,
                "source_index": int(record["source_index"]),
                "relation": record["relation"],
                "energy_first": records[0]["energy"],
                "energy_last": records[-1]["energy"],
                "mean_update_rms": sum(item["update_rms"] for item in records)
                / len(records),
                "samples": [
                    {
                        "sample_index": sample_index,
                        "delta_first": first_by_sample[sample_index][
                            "relation_signed_delta"
                        ],
                        "delta_last": last_by_sample[sample_index][
                            "relation_signed_delta"
                        ],
                        "violation_first": first_by_sample[sample_index][
                            "relation_violation"
                        ],
                        "violation_last": last_by_sample[sample_index][
                            "relation_violation"
                        ],
                    }
                    for sample_index in range(4)
                ],
            }
        )
    return summaries


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    manifest = read_jsonl(args.manifest.resolve())
    baseline_results = load_results(run_dir, "baseline")
    guided_results = load_results(run_dir, "guided")
    baseline = summarize(manifest, baseline_results, args.position_threshold)
    guided = summarize(manifest, guided_results, args.position_threshold)

    transitions: Counter[str] = Counter()
    changed = []
    for baseline_row, guided_row in zip(baseline["rows"], guided["rows"]):
        if baseline_row["correct"] == guided_row["correct"]:
            transition = "same"
        elif guided_row["correct"]:
            transition = "gain"
        else:
            transition = "harm"
        transitions[transition] += 1
        if transition != "same":
            changed.append(
                {
                    "local_index": baseline_row["local_index"],
                    "source_index": baseline_row["source_index"],
                    "relation": baseline_row["relation"],
                    "sample_index": baseline_row["sample_index"],
                    "baseline_category": baseline_row["category"],
                    "guided_category": guided_row["category"],
                    "baseline_reason": baseline_row["reason"],
                    "guided_reason": guided_row["reason"],
                }
            )

    payload = {
        "run_dir": str(run_dir),
        "manifest": str(args.manifest.resolve()),
        "position_threshold": args.position_threshold,
        "baseline": baseline,
        "guided": guided,
        "paired_transitions": dict(transitions),
        "changed": changed,
        "guidance": load_guidance_summary(run_dir, manifest),
    }
    output = args.output or (run_dir / "failure_analysis.json")
    output.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output.resolve()),
                "baseline": baseline["overall"],
                "guided": guided["overall"],
                "paired_transitions": dict(transitions),
                "changed": changed,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
