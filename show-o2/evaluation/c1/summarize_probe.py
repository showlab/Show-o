#!/usr/bin/env python3
"""Aggregate Show-o2 gradient-probe samples with bootstrap confidence intervals."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path


CORE_SCOPES = (
    "fusion",
    "vision.all",
    "qwen.all",
    "qwen.early",
    "qwen.middle",
    "qwen.late",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def mean_ci(values: list[float], bootstrap: int, rng: random.Random) -> dict:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not clean:
        return {"n": 0, "mean": None, "ci95_low": None, "ci95_high": None}
    mean = statistics.fmean(clean)
    if len(clean) == 1 or bootstrap < 1:
        return {"n": len(clean), "mean": mean, "ci95_low": None, "ci95_high": None}
    means = [statistics.fmean(rng.choices(clean, k=len(clean))) for _ in range(bootstrap)]
    return {
        "n": len(clean),
        "mean": mean,
        "ci95_low": percentile(means, 0.025),
        "ci95_high": percentile(means, 0.975),
    }


def describe(rows: list[dict], bootstrap: int, rng: random.Random) -> dict:
    cosines = [row["cosine"] for row in rows if row["cosine"] is not None]
    conflicts = [float(row["conflict"]) for row in rows if row["conflict"] is not None]
    cancellations = [row["cancellation_ratio"] for row in rows if row["cancellation_ratio"] is not None]
    dominance = [row["dominance_a_over_b"] for row in rows if row["dominance_a_over_b"] is not None]
    return {
        "cosine": mean_ci(cosines, bootstrap, rng),
        "conflict_rate": mean_ci(conflicts, bootstrap, rng),
        "cancellation_ratio": mean_ci(cancellations, bootstrap, rng),
        "dominance_a_over_b": mean_ci(dominance, bootstrap, rng),
    }


def render_number(value, digits: int = 4) -> str:
    return "N/A" if value is None else f"{value:.{digits}f}"


def render_ci(metric: dict) -> str:
    if metric["mean"] is None:
        return "N/A"
    if metric["ci95_low"] is None:
        return render_number(metric["mean"])
    return f"{metric['mean']:.4f} [{metric['ci95_low']:.4f}, {metric['ci95_high']:.4f}]"


def main() -> None:
    args = parse_args()
    if not args.samples.is_file():
        raise FileNotFoundError(args.samples)
    samples = [json.loads(line) for line in args.samples.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not samples:
        raise RuntimeError("No completed probe samples")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    records = []
    for sample in samples:
        for pair, scopes in sample["pairs"].items():
            for scope, values in scopes.items():
                records.append({
                    "sample_id": sample["sample_id"],
                    "tag": sample["tag"],
                    "correct": bool(sample["correct"]),
                    "pair": pair,
                    "scope": scope,
                    **values,
                })

    def aggregate(subset: list[dict]) -> dict:
        grouped = defaultdict(list)
        for record in subset:
            grouped[(record["pair"], record["scope"])].append(record)
        return {
            pair: {
                scope: describe(rows, args.bootstrap, rng)
                for (group_pair, scope), rows in grouped.items()
                if group_pair == pair
            }
            for pair in sorted({record["pair"] for record in subset})
        }

    overall = aggregate(records)
    by_tag = {}
    for tag in sorted({sample["tag"] for sample in samples}):
        by_tag[tag] = aggregate([record for record in records if record["tag"] == tag])
    by_correct = {
        "correct": aggregate([record for record in records if record["correct"]]),
        "incorrect": aggregate([record for record in records if not record["correct"]]),
    }
    losses = {
        loss: mean_ci([sample["losses"][loss] for sample in samples], args.bootstrap, rng)
        for loss in ("lm", "mmu", "t2i")
    }
    runtime = {
        "sample_seconds": mean_ci([sample["runtime_seconds"] for sample in samples], args.bootstrap, rng),
        "peak_cuda_memory_bytes": max(sample["peak_cuda_memory_bytes"] for sample in samples),
    }
    summary = {
        "samples": len(samples),
        "sample_ids": [sample["sample_id"] for sample in samples],
        "losses": losses,
        "runtime": runtime,
        "overall": overall,
        "by_tag": by_tag,
        "by_correct": by_correct,
        "interpretation": {
            "strict_conflict": "dot_product < 0, equivalently cosine < 0 when both gradients are non-zero",
            "mmu_t2i": "image-conditioned NTP versus text-to-image flow; valid for Fusion and Qwen",
            "lm_mmu": "text-only NTP versus image-conditioned NTP; valid for Qwen, not Fusion",
            "lm_t2i": "text-only NTP versus text-to-image flow; valid for Qwen, not Fusion",
        },
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")

    lines = [
        "# Show-o2 C1 gradient-conflict probe",
        "",
        f"Completed paired samples: {len(samples)}",
        "",
        "A strict conflict means the two task gradients have a negative dot product. Values are mean [95% bootstrap CI].",
        "",
    ]
    for stratum, label in (("correct", "Clean pairs (GenEval correct)"), ("incorrect", "Generation failures")):
        stratum_summary = by_correct[stratum].get("mmu_t2i", {})
        if not stratum_summary:
            continue
        lines += [
            f"## {label}: MMU-T2I",
            "",
            "| Scope | Cosine | Conflict rate | Cancellation | MMU/flow norm ratio |",
            "|---|---:|---:|---:|---:|",
        ]
        for scope in CORE_SCOPES:
            if scope not in stratum_summary:
                continue
            stats = stratum_summary[scope]
            lines.append(
                f"| {scope} | {render_ci(stats['cosine'])} | {render_ci(stats['conflict_rate'])} | "
                f"{render_ci(stats['cancellation_ratio'])} | {render_ci(stats['dominance_a_over_b'])} |"
            )
        lines.append("")

    for pair in ("mmu_t2i", "lm_mmu", "lm_t2i"):
        if pair not in overall:
            continue
        lines += [
            f"## {pair}",
            "",
            "| Scope | Cosine | Conflict rate | Cancellation | A/B norm ratio |",
            "|---|---:|---:|---:|---:|",
        ]
        for scope in CORE_SCOPES:
            if scope not in overall[pair]:
                continue
            stats = overall[pair][scope]
            lines.append(
                f"| {scope} | {render_ci(stats['cosine'])} | {render_ci(stats['conflict_rate'])} | "
                f"{render_ci(stats['cancellation_ratio'])} | {render_ci(stats['dominance_a_over_b'])} |"
            )
        lines.append("")

    lines += ["## MMU-T2I by GenEval category", "", "| Category | Scope | Cosine | Conflict rate |", "|---|---|---:|---:|"]
    for tag, tag_summary in by_tag.items():
        pair = tag_summary.get("mmu_t2i", {})
        for scope in ("fusion", "qwen.all"):
            if scope not in pair:
                continue
            lines.append(
                f"| {tag} | {scope} | {render_ci(pair[scope]['cosine'])} | "
                f"{render_ci(pair[scope]['conflict_rate'])} |"
            )
    lines.append("")
    (args.output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": "ok", "samples": len(samples), "summary": str(args.output_dir / 'summary.json')}))


if __name__ == "__main__":
    main()
