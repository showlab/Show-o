#!/usr/bin/env python3
"""Build paired CLIPScore comparisons from C7 per-image result files."""

from __future__ import annotations

import argparse
import itertools
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


FIELDS = ("clipscore", "raw_cosine", "relation_margin_raw")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--variants", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260809)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def read_records(path: Path) -> dict[tuple[int, int], dict[str, Any]]:
    records: dict[tuple[int, int], dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            key = (int(record["prompt_id"]), int(record["sample_index"]))
            if key in records:
                raise ValueError(f"duplicate image key in {path}: {key}")
            records[key] = record
    if not records:
        raise ValueError(f"no records found in {path}")
    return records


def cluster_ci(
    deltas: dict[tuple[int, int], float], samples: int, seed: int
) -> list[float]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for (prompt_id, _), delta in deltas.items():
        grouped[prompt_id].append(delta)
    means = np.asarray(
        [np.mean(grouped[key]) for key in sorted(grouped)], dtype=np.float64
    )
    rng = np.random.default_rng(seed)
    draws = rng.choice(means, size=(samples, len(means)), replace=True)
    return [float(value) for value in np.quantile(draws.mean(axis=1), [0.025, 0.975])]


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def main() -> int:
    args = parse_args()
    variants = tuple(item.strip() for item in args.variants.split(",") if item.strip())
    if len(variants) < 2 or len(set(variants)) != len(variants):
        raise ValueError("variants must contain at least two unique names")
    if args.bootstrap_samples < 1:
        raise ValueError("bootstrap-samples must be positive")
    run_root = args.run_root.resolve()
    records = {
        variant: read_records(run_root / variant / "per_image_scores.jsonl")
        for variant in variants
    }
    reference_keys = set(records[variants[0]])
    for variant in variants[1:]:
        if set(records[variant]) != reference_keys:
            raise ValueError(f"image keys do not match for {variant}")
    for key in reference_keys:
        reference = records[variants[0]][key]
        identity = (
            reference["prompt"],
            reference["counterfactual_prompt"],
            reference["relation"],
        )
        for variant in variants[1:]:
            candidate = records[variant][key]
            if (
                candidate["prompt"],
                candidate["counterfactual_prompt"],
                candidate["relation"],
            ) != identity:
                raise ValueError(f"prompt metadata mismatch for {variant}: {key}")

    result: dict[str, Any] = {
        "status": "completed",
        "protocol": "c7-paired-clipscore-position72-v1",
        "images": len(reference_keys),
        "prompts": len({key[0] for key in reference_keys}),
        "variants": {},
        "pairwise": {},
    }
    for variant in variants:
        result["variants"][variant] = {
            field: float(np.mean([records[variant][key][field] for key in reference_keys]))
            for field in FIELDS
        }

    for left, right in itertools.combinations(variants, 2):
        comparison: dict[str, Any] = {}
        for offset, field in enumerate(FIELDS):
            deltas = {
                key: float(records[right][key][field] - records[left][key][field])
                for key in reference_keys
            }
            values = np.asarray(list(deltas.values()), dtype=np.float64)
            comparison[field] = {
                "delta_right_minus_left": float(values.mean()),
                "prompt_cluster_bootstrap_95_ci": cluster_ci(
                    deltas,
                    args.bootstrap_samples,
                    args.bootstrap_seed + offset,
                ),
                "right_wins": int(np.sum(values > 1e-12)),
                "left_wins": int(np.sum(values < -1e-12)),
                "ties": int(np.sum(np.abs(values) <= 1e-12)),
            }
        result["pairwise"][f"{left}__vs__{right}"] = comparison

    output = args.output or run_root / "comparison_summary.json"
    atomic_json(output.resolve(), result)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
