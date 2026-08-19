#!/usr/bin/env python3
"""Compare detector results against a checkpoint control and apply a strict gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def rows(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def key(row):
    path = Path(row["filename"])
    return int(path.parents[1].name), int(path.stem)


def both_present(row):
    metadata = json.loads(row["metadata"])
    detected = json.loads(row["details"])
    return all(
        len(detected.get(item["class"], [])) >= int(item["count"])
        for item in metadata.get("include", [])
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--control", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--checkpoint-map", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--apply-gate", action="store_true")
    args = parser.parse_args()
    variants = [args.control, *[x for x in args.candidates.split(",") if x]]
    result = {
        variant: {key(row): row for row in rows(args.run_dir / variant / "results/results.jsonl")}
        for variant in variants
    }
    keys = set(result[args.control])
    if any(set(values) != keys for values in result.values()):
        raise ValueError("variant detector result keys do not match")
    control = result[args.control]
    summary = {}
    for variant, values in result.items():
        gain = loss = 0
        for item_key in keys:
            current, base = bool(values[item_key]["correct"]), bool(control[item_key]["correct"])
            gain += int(current and not base)
            loss += int(base and not current)
        summary[variant] = {
            "correct": sum(bool(row["correct"]) for row in values.values()),
            "total": len(values),
            "both_objects_detected": sum(both_present(row) for row in values.values()),
            "gain_vs_control": gain,
            "loss_vs_control": loss,
            "net_gain_vs_control": gain - loss,
        }
    control_metrics = summary[args.control]
    candidates = variants[1:]
    passing = [
        variant for variant in candidates
        if summary[variant]["correct"] > control_metrics["correct"]
        and summary[variant]["gain_vs_control"] > summary[variant]["loss_vs_control"]
        and summary[variant]["both_objects_detected"] >= control_metrics["both_objects_detected"]
    ]
    checkpoints = json.loads(args.checkpoint_map.read_text())

    def step(variant):
        return int(variant.rsplit("_", 1)[-1]) if variant.rsplit("_", 1)[-1].isdigit() else 10**9

    selected = max(
        passing,
        key=lambda variant: (
            summary[variant]["correct"],
            summary[variant]["both_objects_detected"],
            -summary[variant]["loss_vs_control"],
            -step(variant),
        ),
        default=None,
    )
    payload = {
        "status": "passed" if selected else ("rejected" if args.apply_gate else "completed"),
        "control": args.control,
        "gate_rule": (
            "correct > control AND paired gain > loss AND both-object detections >= control"
        ),
        "variants": summary,
        "passing_variants": passing,
        "selected_variant": selected,
        "selected_checkpoint": checkpoints.get(selected) if selected else None,
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
