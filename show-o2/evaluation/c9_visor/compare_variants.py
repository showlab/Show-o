#!/usr/bin/env python3
"""Paired comparison of two VISOR image-result files."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

SHOWO2_ROOT = Path(__file__).resolve().parents[2]
if str(SHOWO2_ROOT) not in sys.path:
    sys.path.insert(0, str(SHOWO2_ROOT))

from evaluation.c9_visor.visor_metrics import RELATIONS, aggregate_visor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260811)
    return parser.parse_args()


def read_rows(path: Path) -> dict[tuple[int, int], dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: dict[tuple[int, int], dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        key = (int(row["unique_id"]), int(row["sample_index"]))
        if key in rows:
            raise ValueError(f"duplicate result at {path}:{line_number}: {key}")
        rows[key] = row
    if not rows:
        raise ValueError(f"VISOR result file is empty: {path}")
    return rows


def _paired_counts(
    keys: list[tuple[int, int]],
    control: dict[tuple[int, int], dict[str, Any]],
    candidate: dict[tuple[int, int], dict[str, Any]],
    field: str,
) -> dict[str, int]:
    gain = sum(not bool(control[key][field]) and bool(candidate[key][field]) for key in keys)
    loss = sum(bool(control[key][field]) and not bool(candidate[key][field]) for key in keys)
    return {"gain": gain, "loss": loss, "net": gain - loss}


def _metric_delta(
    control_rows: list[dict[str, Any]], candidate_rows: list[dict[str, Any]]
) -> dict[str, float]:
    control_metrics = aggregate_visor(control_rows)
    candidate_metrics = aggregate_visor(candidate_rows)
    names = ("object_accuracy_both", "visor_cond", "visor_uncond")
    return {
        name: float(candidate_metrics[name]) - float(control_metrics[name])
        for name in names
    }


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def cluster_bootstrap(
    control: dict[tuple[int, int], dict[str, Any]],
    candidate: dict[tuple[int, int], dict[str, Any]],
    samples: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    if samples < 1:
        raise ValueError("bootstrap-samples must be positive")
    by_prompt: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for key in sorted(control):
        by_prompt[key[0]].append(key)
    prompt_ids = sorted(by_prompt)
    random_state = random.Random(seed)
    distributions = {
        "object_accuracy_both": [],
        "visor_cond": [],
        "visor_uncond": [],
    }
    for _ in range(samples):
        sampled_prompts = [random_state.choice(prompt_ids) for _ in prompt_ids]
        control_rows = []
        candidate_rows = []
        for draw_index, prompt in enumerate(sampled_prompts):
            for key in by_prompt[prompt]:
                control_row = dict(control[key])
                candidate_row = dict(candidate[key])
                # A cluster may be sampled repeatedly; temporary draw ids keep
                # the bootstrap sample keys unique without changing metrics.
                control_row["unique_id"] = draw_index
                candidate_row["unique_id"] = draw_index
                control_rows.append(control_row)
                candidate_rows.append(candidate_row)
        delta = _metric_delta(control_rows, candidate_rows)
        for name, value in delta.items():
            distributions[name].append(value)
    return {
        name: {
            "lower_95": _percentile(values, 0.025),
            "median": _percentile(values, 0.5),
            "upper_95": _percentile(values, 0.975),
        }
        for name, values in distributions.items()
    }


def compare(
    control: dict[tuple[int, int], dict[str, Any]],
    candidate: dict[tuple[int, int], dict[str, Any]],
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    if set(control) != set(candidate):
        raise ValueError("control and candidate VISOR result keys differ")
    keys = sorted(control)
    for key in keys:
        fields = ("prompt", "object_a", "object_b", "relation")
        if any(control[key][field] != candidate[key][field] for field in fields):
            raise ValueError(f"paired VISOR metadata differs at {key}")
    control_rows = [control[key] for key in keys]
    candidate_rows = [candidate[key] for key in keys]
    relation_pairs = {
        relation: [key for key in keys if control[key]["relation"] == relation]
        for relation in RELATIONS
    }
    return {
        "images": len(keys),
        "prompts": len({key[0] for key in keys}),
        "control": aggregate_visor(control_rows),
        "candidate": aggregate_visor(candidate_rows),
        "delta_percentage_points": _metric_delta(control_rows, candidate_rows),
        "paired": {
            "joint_success": _paired_counts(keys, control, candidate, "joint_success"),
            "object_both": _paired_counts(keys, control, candidate, "detected_both"),
        },
        "paired_joint_by_relation": {
            relation: _paired_counts(relation_pairs[relation], control, candidate, "joint_success")
            for relation in RELATIONS
        },
        "prompt_cluster_bootstrap_95": cluster_bootstrap(
            control, candidate, bootstrap_samples, bootstrap_seed
        ),
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": bootstrap_seed,
    }


def main() -> None:
    args = parse_args()
    summary = compare(
        read_rows(args.control),
        read_rows(args.candidate),
        args.bootstrap_samples,
        args.bootstrap_seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

