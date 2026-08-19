#!/usr/bin/env python3
"""Aggregate C2 condition-localization samples with bootstrap intervals."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
import numpy as np


CORE_STAGES = (
    "fusion",
    "qwen_layer_01",
    "qwen_layer_15",
    "qwen_layer_28",
    "qwen_final",
    "diff_proj",
    "diffusion_head_01",
    "diffusion_head_05",
    "diffusion_head_10",
    "velocity_tokens",
    "velocity",
    "euler_prediction",
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
    position = (len(ordered) - 1) * q
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def mean_ci(values: list[float | None], bootstrap: int, rng: random.Random) -> dict:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not clean:
        return {"n": 0, "mean": None, "ci95_low": None, "ci95_high": None}
    mean = statistics.fmean(clean)
    if len(clean) == 1 or bootstrap < 1:
        return {"n": len(clean), "mean": mean, "ci95_low": None, "ci95_high": None}
    values_array = np.asarray(clean, dtype=np.float64)
    generator = np.random.default_rng(rng.randrange(2**63))
    indices = generator.integers(0, len(clean), size=(bootstrap, len(clean)))
    estimates = values_array[indices].mean(axis=1).tolist()
    return {
        "n": len(clean),
        "mean": mean,
        "ci95_low": percentile(estimates, 0.025),
        "ci95_high": percentile(estimates, 0.975),
    }

def prompt_key(row: dict) -> str:
    return row.get("prompt_id") or row["sample_id"].split("-", 1)[0]


def clustered_ci(rows: list[dict], getter, bootstrap: int, rng: random.Random) -> dict:
    grouped = defaultdict(list)
    for row in rows:
        value = getter(row)
        if value is not None and math.isfinite(float(value)):
            grouped[prompt_key(row)].append(float(value))
    cluster_means = [statistics.fmean(values) for values in grouped.values() if values]
    result = mean_ci(cluster_means, bootstrap, rng)
    result["n_clusters"] = len(cluster_means)
    result["n_records"] = sum(len(values) for values in grouped.values())
    return result


def paired_outcome_ci(rows: list[dict], getter, bootstrap: int, rng: random.Random) -> dict:
    grouped = defaultdict(lambda: {True: [], False: []})
    for row in rows:
        value = getter(row)
        if value is not None and math.isfinite(float(value)):
            grouped[prompt_key(row)][bool(row["correct"])].append(float(value))
    differences = [
        statistics.fmean(values[True]) - statistics.fmean(values[False])
        for values in grouped.values()
        if values[True] and values[False]
    ]
    result = mean_ci(differences, bootstrap, rng)
    result["n_paired_prompts"] = len(differences)
    result["direction"] = "correct_minus_incorrect"
    return result


def paired_outcome_summary(rows: list[dict], bootstrap: int, rng: random.Random) -> dict:
    return {
        stage: paired_outcome_ci(
            rows,
            lambda row, s=stage: row["stages"][s]["counterfactual"]["condition_gain"],
            bootstrap,
            rng,
        )
        for stage in ("qwen_final", "diffusion_head_10", "velocity", "euler_prediction")
    }


def aggregate(rows: list[dict], bootstrap: int, rng: random.Random) -> dict:
    stages = sorted({stage for row in rows for stage in row["stages"]})
    output = {}
    for stage in stages:
        output[stage] = {}
        for variant in ("same", "counterfactual", "null"):
            output[stage][variant] = {
                metric: clustered_ci(
                    [row for row in rows if stage in row["stages"]],
                    lambda row, s=stage, v=variant, m=metric: row["stages"][s][v].get(m),
                    bootstrap, rng,
                )
                for metric in ("relative_l2", "cosine", "rms_delta", "reference_rms", "condition_gain")
            }
    return output


def aggregate_errors(rows: list[dict], bootstrap: int, rng: random.Random) -> dict:
    output = {}
    for variant in ("matched", "same", "counterfactual", "null"):
        output[variant] = {}
        for metric in ("velocity_target_mse", "euler_clean_mse"):
            output[variant][metric] = clustered_ci(
                rows, lambda row, v=variant, m=metric: row["prediction_errors"][v][m], bootstrap, rng
            )
            margin = f"{metric}_margin_vs_matched"
            if variant != "matched":
                output[variant][margin] = clustered_ci(
                    rows, lambda row, v=variant, m=margin: row["prediction_errors"][v][m], bootstrap, rng
                )
    return output


def render(metric: dict, digits: int = 5) -> str:
    if metric["mean"] is None:
        return "N/A"
    if metric["ci95_low"] is None:
        return f"{metric['mean']:.{digits}f}"
    return f"{metric['mean']:.{digits}f} [{metric['ci95_low']:.{digits}f}, {metric['ci95_high']:.{digits}f}]"


def retention(aggregated: dict, stage: str, variant: str = "counterfactual") -> float | None:
    if stage not in aggregated or "qwen_final" not in aggregated:
        return None
    numerator = aggregated[stage][variant]["condition_gain"]["mean"]
    denominator = aggregated["qwen_final"][variant]["condition_gain"]["mean"]
    return numerator / denominator if numerator is not None and denominator not in (None, 0.0) else None


def main() -> None:
    args = parse_args()
    if not args.samples.is_file():
        raise FileNotFoundError(args.samples)
    rows = [json.loads(line) for line in args.samples.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise RuntimeError("No completed C2 samples")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    overall = aggregate(rows, args.bootstrap, rng)
    by_tag = {
        tag: aggregate([row for row in rows if row["tag"] == tag], args.bootstrap, rng)
        for tag in sorted({row["tag"] for row in rows})
    }
    by_timestep = {
        f"{timestep:.6f}": aggregate(
            [row for row in rows if row["timestep"] == timestep], args.bootstrap, rng
        )
        for timestep in sorted({row["timestep"] for row in rows})
    }
    by_outcome = {
        label: aggregate([row for row in rows if bool(row["correct"]) == correct], args.bootstrap, rng)
        for label, correct in (("correct", True), ("incorrect", False))
        if any(bool(row["correct"]) == correct for row in rows)
    }
    paired_outcome = paired_outcome_summary(rows, args.bootstrap, rng)
    paired_outcome_by_tag = {
        tag: paired_outcome_summary([row for row in rows if row["tag"] == tag], args.bootstrap, rng)
        for tag in sorted({row["tag"] for row in rows})
    }
    by_tag_outcome = {
        tag: {
            label: aggregate([row for row in rows if row["tag"] == tag and bool(row["correct"]) == correct], args.bootstrap, rng)
            for label, correct in (("correct", True), ("incorrect", False))
            if any(row["tag"] == tag and bool(row["correct"]) == correct for row in rows)
        }
        for tag in sorted({row["tag"] for row in rows})
    }
    token_matched_rows = [
        row for row in rows
        if row["prompt_token_counts"]["matched"] == row["prompt_token_counts"]["counterfactual"]
    ]
    token_length_control = {
        "records": len(token_matched_rows),
        "fraction": len(token_matched_rows) / len(rows),
        "aggregate": aggregate(token_matched_rows, args.bootstrap, rng) if token_matched_rows else {},
    }
    prediction_errors = aggregate_errors(rows, args.bootstrap, rng)
    stage_retention = {stage: retention(overall, stage) for stage in overall}
    runtime = {
        "record_seconds": mean_ci([row["runtime_seconds"] for row in rows], args.bootstrap, rng),
        "peak_cuda_memory_bytes": max(row["peak_cuda_memory_bytes"] for row in rows),
    }

    same_floor = overall["qwen_final"]["same"]["relative_l2"]["mean"]
    qwen_gain = overall["qwen_final"]["counterfactual"]["condition_gain"]["mean"]
    final_head = next((stage for stage in reversed(CORE_STAGES) if stage.startswith("diffusion_head_") and stage in overall), None)
    diagnosis = {
        "fusion_is_prompt_independent_control": overall["fusion"]["counterfactual"]["condition_gain"]["mean"],
        "same_prompt_numerical_floor_at_qwen": same_floor,
        "qwen_counterfactual_gain": qwen_gain,
        "final_head_retention_vs_qwen": retention(overall, final_head) if final_head else None,
        "velocity_retention_vs_qwen": retention(overall, "velocity"),
        "paired_outcome": paired_outcome,
        "paired_outcome_by_tag": paired_outcome_by_tag,
        "interpretation": [
            "Near-zero Fusion gain is expected: Fusion sees x_t but no prompt; it is an internal negative control.",
            "Low Qwen gain above the same-prompt floor indicates weak text-to-image-token propagation in the shared backbone.",
            "High Qwen gain followed by low diffusion-head retention localizes attenuation to the diffusion head.",
            "High velocity sensitivity with poor matched-vs-counterfactual target-error margin points beyond one-step conditioning, such as global sampling or VAE decoding.",
            "Token-length-matched results reduce absolute-position and RoPE confounding from counterfactual wording.",
        ],
    }
    summary = {
        "records": len(rows),
        "unique_samples": len({row["sample_id"] for row in rows}),
        "unique_prompts": len({prompt_key(row) for row in rows}),
        "correct_images": len({row["sample_id"] for row in rows if row["correct"]}),
        "incorrect_images": len({row["sample_id"] for row in rows if not row["correct"]}),
        "bootstrap_unit": "prompt_id",
        "tags": sorted({row["tag"] for row in rows}),
        "timesteps": sorted({row["timestep"] for row in rows}),
        "overall": overall,
        "by_tag": by_tag,
        "by_outcome": by_outcome,
        "by_tag_outcome": by_tag_outcome,
        "by_timestep": by_timestep,
        "token_length_control": token_length_control,
        "prediction_errors": prediction_errors,
        "counterfactual_retention_vs_qwen_final": stage_retention,
        "runtime": runtime,
        "diagnosis": diagnosis,
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")

    lines = [
        "# Show-o2 C2 condition-localization probe",
        "",
        f"Completed records: {len(rows)} ({summary['unique_samples']} images x {len(summary['timesteps'])} timesteps)",
        f"Independent prompt clusters: {summary['unique_prompts']}; correct images: {summary['correct_images']}; incorrect images: {summary['incorrect_images']}",
        "Confidence intervals use prompt-cluster bootstrap; images and timesteps from one prompt are not treated as independent units.",
        f"Counterfactual token-length matched: {len(token_matched_rows)}/{len(rows)} ({100 * token_length_control['fraction']:.1f}%)",
        "",
        "Values are mean [95% bootstrap CI]. Condition gain is counterfactual relative-L2 minus the duplicated same-prompt floor.",
        "",
        "## Overall path",
        "",
        "| Stage | Same relative L2 | Counterfactual relative L2 | Condition gain | Retention vs Qwen final |",
        "|---|---:|---:|---:|---:|",
    ]
    for stage in CORE_STAGES:
        if stage not in overall:
            continue
        ratio = stage_retention[stage]
        lines.append(
            f"| {stage} | {render(overall[stage]['same']['relative_l2'])} | "
            f"{render(overall[stage]['counterfactual']['relative_l2'])} | "
            f"{render(overall[stage]['counterfactual']['condition_gain'])} | "
            f"{'N/A' if ratio is None else f'{ratio:.4f}'} |"
        )

    lines += [
        "",
        "## By GenEval category",
        "",
        "| Category | Qwen gain | Final head gain | Velocity gain | Velocity/Qwen |",
        "|---|---:|---:|---:|---:|",
    ]
    for tag, values in by_tag.items():
        final_stage = final_head if final_head in values else next(
            stage for stage in reversed(CORE_STAGES) if stage.startswith("diffusion_head_") and stage in values
        )
        ratio = retention(values, "velocity")
        lines.append(
            f"| {tag} | {render(values['qwen_final']['counterfactual']['condition_gain'])} | "
            f"{render(values[final_stage]['counterfactual']['condition_gain'])} | "
            f"{render(values['velocity']['counterfactual']['condition_gain'])} | "
            f"{'N/A' if ratio is None else f'{ratio:.4f}'} |"
        )

    lines += [
        "",
        "## By C0 generation outcome",
        "",
        "| Outcome | Qwen gain | Final head gain | Velocity gain | Velocity/Qwen |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, values in by_outcome.items():
        ratio = retention(values, "velocity")
        lines.append(
            f"| {label} | {render(values['qwen_final']['counterfactual']['condition_gain'])} | "
            f"{render(values[final_head]['counterfactual']['condition_gain'])} | "
            f"{render(values['velocity']['counterfactual']['condition_gain'])} | "
            f"{'N/A' if ratio is None else f'{ratio:.4f}'} |"
        )

    lines += [
        "",
        "## Category by outcome",
        "",
        "| Category | Outcome | Qwen gain | Final head gain | Velocity gain |",
        "|---|---|---:|---:|---:|",
    ]
    for tag, outcomes in by_tag_outcome.items():
        for label, values in outcomes.items():
            lines.append(
                f"| {tag} | {label} | {render(values['qwen_final']['counterfactual']['condition_gain'])} | "
                f"{render(values[final_head]['counterfactual']['condition_gain'])} | "
                f"{render(values['velocity']['counterfactual']['condition_gain'])} |"
            )
    lines += [
        "",
        "## Paired outcome difference",
        "",
        "Only prompts with both correct and incorrect C0 images are included. Positive means stronger condition gain on correct images.",
        "",
        "| Stage | Correct - incorrect gain | Paired prompts |",
        "|---|---:|---:|",
    ]
    for stage, metric in paired_outcome.items():
        lines.append(f"| {stage} | {render(metric)} | {metric['n_paired_prompts']} |")

    lines += [
        "",
        "## Paired outcome difference by category",
        "",
        "| Category | Stage | Correct - incorrect gain | Paired prompts |",
        "|---|---|---:|---:|",
    ]
    for tag, stages in paired_outcome_by_tag.items():
        for stage, metric in stages.items():
            if metric["n_paired_prompts"] == 0:
                continue
            lines.append(
                f"| {tag} | {stage} | {render(metric)} | {metric['n_paired_prompts']} |"
            )


    lines += [
        "",
        "## By timestep",
        "",
        "| t | Qwen gain | Final head gain | Velocity gain | Velocity/Qwen |",
        "|---:|---:|---:|---:|---:|",
    ]
    for timestep, values in by_timestep.items():
        ratio = retention(values, "velocity")
        lines.append(
            f"| {float(timestep):.2f} | {render(values['qwen_final']['counterfactual']['condition_gain'])} | "
            f"{render(values[final_head]['counterfactual']['condition_gain'])} | "
            f"{render(values['velocity']['counterfactual']['condition_gain'])} | "
            f"{'N/A' if ratio is None else f'{ratio:.4f}'} |"
        )

    lines += [
        "",
        "## Target consistency",
        "",
        "Positive margins mean the alternative prompt predicts the matched image/velocity less accurately than the matched prompt.",
        "",
        "| Variant | Velocity target MSE | Margin | Euler clean MSE | Margin |",
        "|---|---:|---:|---:|---:|",
    ]
    for variant in ("matched", "same", "counterfactual", "null"):
        values = prediction_errors[variant]
        velocity_margin = values.get("velocity_target_mse_margin_vs_matched")
        euler_margin = values.get("euler_clean_mse_margin_vs_matched")
        lines.append(
            f"| {variant} | {render(values['velocity_target_mse'])} | "
            f"{'0' if velocity_margin is None else render(velocity_margin)} | "
            f"{render(values['euler_clean_mse'])} | {'0' if euler_margin is None else render(euler_margin)} |"
        )

    lines += ["", "## Interpretation", ""]
    lines.extend(f"- {item}" for item in diagnosis["interpretation"])
    lines.append("")
    (args.output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": "ok", "records": len(rows), "summary": str(args.output_dir / "summary.json")}, sort_keys=True))


if __name__ == "__main__":
    main()
