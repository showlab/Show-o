#!/usr/bin/env python3
"""Aggregate Show-o2 spatial layer probe records and rank guidance layers."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def clean_mean(values: list[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(value)]
    return mean(finite) if finite else None


def aggregate_layer(records: list[dict[str, Any]], source: str, index: int) -> dict[str, Any]:
    rows = [record["layer_metrics"][source][index] for record in records]
    variants = [variant for row in rows for variant in row["variants"]]
    relations: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row, record in zip(rows, records):
        relations[record["relations"]["matched"]].append(row)
    return {
        "source": source,
        "layer_index": index,
        "layer_number": index + 1,
        "records": len(rows),
        "pair_direction_accuracy": mean(float(row["pair_direction_correct"]) for row in rows),
        "relation_flip_rate": mean(float(row["relation_flip"]) for row in rows),
        "pair_margin_accuracy": mean(float(row["pair_margin_correct"]) for row in rows),
        "variant_direction_accuracy": mean(float(row["direction_correct"]) for row in variants),
        "mean_signed_delta": clean_mean([row["relation_signed_delta"] for row in variants]),
        "mean_abs_signed_delta": clean_mean([abs(row["relation_signed_delta"]) for row in variants]),
        "mean_text_mass": clean_mean([row["text_mass"] for row in variants]),
        "mean_boxdiff_inner_loss": clean_mean([row["boxdiff_inner_loss"] for row in variants]),
        "mean_boxdiff_outer_loss": clean_mean([row["boxdiff_outer_loss"] for row in variants]),
        "mean_boxdiff_corner_loss": clean_mean([row["boxdiff_corner_loss"] for row in variants]),
        "by_relation": {
            relation: {
                "records": len(relation_rows),
                "pair_direction_accuracy": mean(
                    float(row["pair_direction_correct"]) for row in relation_rows
                ),
                "relation_flip_rate": mean(float(row["relation_flip"]) for row in relation_rows),
            }
            for relation, relation_rows in sorted(relations.items())
        },
    }


def aggregate_gradients(records: list[dict[str, Any]], source: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        for index, values in record["gradient_attribution"][source].items():
            grouped[index].append(values)
    output = {}
    for index, rows in sorted(grouped.items(), key=lambda item: int(item[0])):
        und_rms = clean_mean([row["und_latent_gradient_rms"] for row in rows])
        gen_rms = clean_mean([row["gen_latent_gradient_rms"] for row in rows])
        output[index] = {
            "layer_index": int(index),
            "layer_number": int(index) + 1,
            "records": len(rows),
            "mean_energy": clean_mean([row["energy"] for row in rows]),
            "mean_latent_gradient_rms": clean_mean([row["latent_gradient_rms"] for row in rows]),
            "mean_und_latent_gradient_rms": und_rms,
            "mean_gen_latent_gradient_rms": gen_rms,
            "und_to_gen_rms_ratio": und_rms / gen_rms if und_rms is not None and gen_rms else None,
            "mean_und_gen_cosine": clean_mean([row["und_gen_cosine"] for row in rows]),
            "mean_und_total_cosine": clean_mean([row["und_total_cosine"] for row in rows]),
            "mean_gen_total_cosine": clean_mean([row["gen_total_cosine"] for row in rows]),
            "mean_branch_reconstruction_relative_rms": clean_mean(
                [row["branch_reconstruction_relative_rms"] for row in rows]
            ),
            "mean_activation_gradient_rms": {
                name: clean_mean(
                    [row["activations"][name]["gradient_rms"] for row in rows]
                )
                for name in rows[0]["activations"]
            },
            "mean_gradient_times_activation_rms": {
                name: clean_mean(
                    [
                        row["activations"][name]["gradient_times_activation_rms"]
                        for row in rows
                    ]
                )
                for name in rows[0]["activations"]
            },
        }
    return output


def rank_key(row: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        row["pair_direction_accuracy"],
        row["relation_flip_rate"],
        row["variant_direction_accuracy"],
        row["mean_abs_signed_delta"] or 0.0,
    )


def main() -> None:
    args = parse_args()
    samples_path = args.run_dir / "samples.jsonl"
    records = [
        json.loads(line)
        for line in samples_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError(f"No records in {samples_path}")
    layer_counts = {
        source: len(records[0]["layer_metrics"][source])
        for source in ("qwen", "diffusion")
    }
    layers = {
        source: [aggregate_layer(records, source, index) for index in range(count)]
        for source, count in layer_counts.items()
    }
    recommendations = {
        source: sorted(values, key=rank_key, reverse=True)[: args.top_k]
        for source, values in layers.items()
    }
    summary = {
        "status": "completed",
        "run_dir": str(args.run_dir.resolve()),
        "records": len(records),
        "source_indices": sorted({record["source_index"] for record in records}),
        "timesteps": sorted({record["timestep"] for record in records}),
        "layers": layers,
        "gradient_attribution": {
            source: aggregate_gradients(records, source)
            for source in ("qwen", "diffusion")
        },
        "recommended_layers": recommendations,
        "ranking": "pair direction accuracy, then inverse-prompt spatial flip rate, then per-variant direction accuracy, then signed-delta magnitude",
        "caveat": "This localizes attention geometry and gradient transport. Final layer selection still requires same-seed detector-scored generation A/B.",
    }
    output = args.output or args.run_dir / "summary.json"
    output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "output": str(output.resolve()),
                "records": len(records),
                "recommended": {
                    source: [
                        {
                            "layer_number": row["layer_number"],
                            "pair_direction_accuracy": row["pair_direction_accuracy"],
                            "relation_flip_rate": row["relation_flip_rate"],
                        }
                        for row in values
                    ]
                    for source, values in recommendations.items()
                },
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
