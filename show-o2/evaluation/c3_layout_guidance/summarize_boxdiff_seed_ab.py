#!/usr/bin/env python3
"""Summarize detector scores and pairing integrity for the BoxDiff seed A/B."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def result_key(row: dict[str, Any]) -> tuple[int, int]:
    path = Path(row["filename"])
    return int(path.parents[1].name), int(path.stem)


def score_rows(run_dir: Path, variant: str) -> dict[tuple[int, int], dict[str, Any]]:
    path = run_dir / variant / "results/results.jsonl"
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = read_jsonl(path)
    keyed = {result_key(row): row for row in rows}
    if len(keyed) != len(rows):
        raise ValueError(f"Duplicate detector result key in {path}")
    return keyed


def pairing_summary(
    hashes: dict[str, dict[str, str]],
    variants: tuple[str, ...],
    expected_keys: set[str],
) -> dict[str, Any]:
    expected = set(variants)
    missing = {}
    mismatched = {}
    absent_keys = sorted(expected_keys - set(hashes))
    extra_keys = sorted(set(hashes) - expected_keys)
    for key, values in hashes.items():
        absent = sorted(expected - set(values))
        if absent:
            missing[key] = absent
        unique = sorted(set(values.values()))
        if len(unique) != 1:
            mismatched[key] = values
    return {
        "pairs": len(hashes),
        "expected_pairs": len(expected_keys),
        "expected_variants": list(variants),
        "complete_pairs": len(hashes) - len(missing),
        "matching_pairs": len(hashes) - len(mismatched),
        "all_complete": not missing and not absent_keys and not extra_keys,
        "all_match": not mismatched,
        "missing": missing,
        "absent_keys": absent_keys,
        "extra_keys": extra_keys,
        "mismatched": mismatched,
    }


def diagnostic_summary(run_dir: Path, variant: str) -> dict[str, Any]:
    paths = sorted((run_dir / variant / "raw").glob("*/diagnostics-*.json"))
    guided_records = []
    for path in paths:
        diagnostics = read_json(path)
        guided_records.extend(
            inner
            for step in diagnostics
            for inner in step.get("guidance", ())
        )
    if not guided_records:
        return {
            "guided_inner_records": 0,
            "mean_energy": None,
            "mean_grad_rms": None,
            "mean_update_rms": None,
            "mean_relation_violation": None,
        }
    violations = [
        item["relation_violation"]
        for record in guided_records
        for item in record["objects"]
        if item.get("object_index") == 0
        and item.get("relation_violation") is not None
    ]
    accepted = [
        bool(value)
        for record in guided_records
        if record.get("backtracking_enabled", False)
        for value in record.get("accepted", ())
    ]
    score_drops = [
        before - after
        for record in guided_records
        if record.get("backtracking_enabled", False)
        for before, after in zip(
            record.get("acceptance_score_before", ()),
            record.get("acceptance_score_after", ()),
        )
    ]
    attempts = [
        int(record.get("backtracking_attempts", 0))
        for record in guided_records
        if record.get("backtracking_enabled", False)
    ]
    return {
        "guided_inner_records": len(guided_records),
        "mean_energy": mean(record["energy"] for record in guided_records),
        "mean_grad_rms": mean(record["grad_rms"] for record in guided_records),
        "mean_update_rms": mean(record["update_rms"] for record in guided_records),
        "mean_relation_violation": mean(violations) if violations else None,
        "accepted_candidates": sum(accepted) if accepted else None,
        "candidate_decisions": len(accepted) if accepted else 0,
        "acceptance_rate": mean(accepted) if accepted else None,
        "mean_acceptance_score_drop": mean(score_drops) if score_drops else None,
        "mean_backtracking_attempts": mean(attempts) if attempts else None,
    }


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    protocol = read_json(run_dir / "protocol.json")
    manifest = read_jsonl(Path(protocol["manifest"]))
    # Older protocol files did not persist a prompt limit. In that case the
    # manifest itself is the complete expected set, allowing the pairing check
    # to report invalid data instead of failing before it can produce a summary.
    prompt_limit = protocol.get("prompts")
    if prompt_limit is not None:
        manifest = manifest[: int(prompt_limit)]
    raw_variants = protocol["variants"]
    variants = (
        tuple(raw_variants.keys())
        if isinstance(raw_variants, dict)
        else tuple(raw_variants)
    )
    if "baseline" not in variants:
        raise ValueError("Protocol variants must include baseline")
    # Protocol JSON is persisted with sorted keys, so a variant such as b_100
    # can sort before baseline even when baseline was evaluated first.
    variants = ("baseline", *(name for name in variants if name != "baseline"))
    relation_by_local = {
        int(row["local_index"]): str(row["relation"]) for row in manifest
    }
    expected_keys = {
        (int(row["local_index"]), sample_index)
        for row in manifest
        for sample_index in range(int(protocol["samples_per_prompt"]))
    }
    results = {variant: score_rows(run_dir, variant) for variant in variants}
    for variant, rows in results.items():
        if set(rows) != expected_keys:
            raise ValueError(
                f"Detector keys for {variant} differ from protocol: "
                f"missing={sorted(expected_keys - set(rows))}, "
                f"extra={sorted(set(rows) - expected_keys)}"
            )

    baseline = results["baseline"]
    variant_summaries = {}
    transitions = {}
    for variant in variants:
        rows = results[variant]
        correct = sum(bool(row["correct"]) for row in rows.values())
        by_relation: dict[str, Counter[str]] = defaultdict(Counter)
        transition_counts: Counter[str] = Counter()
        changed = []
        for key in sorted(expected_keys):
            relation = relation_by_local[key[0]]
            current_correct = bool(rows[key]["correct"])
            baseline_correct = bool(baseline[key]["correct"])
            by_relation[relation]["correct" if current_correct else "wrong"] += 1
            if current_correct == baseline_correct:
                transition = "same_correct" if current_correct else "same_wrong"
            elif current_correct:
                transition = "gain"
            else:
                transition = "loss"
            transition_counts[transition] += 1
            if transition in {"gain", "loss"}:
                changed.append(
                    {
                        "local_index": key[0],
                        "sample_index": key[1],
                        "relation": relation,
                        "transition": transition,
                    }
                )
        variant_summaries[variant] = {
            "correct": correct,
            "total": len(rows),
            "accuracy": correct / len(rows),
            "delta_correct_vs_baseline": correct
            - sum(bool(row["correct"]) for row in baseline.values()),
            "by_relation": {
                relation: {
                    **dict(counts),
                    "total": sum(counts.values()),
                    "accuracy": counts["correct"] / sum(counts.values()),
                }
                for relation, counts in sorted(by_relation.items())
            },
            "diagnostics": diagnostic_summary(run_dir, variant),
        }
        transitions[variant] = {
            "gain": transition_counts["gain"],
            "loss": transition_counts["loss"],
            "same_correct": transition_counts["same_correct"],
            "same_wrong": transition_counts["same_wrong"],
            "net_gain": transition_counts["gain"] - transition_counts["loss"],
            "changed": changed,
        }

    expected_hash_keys = {
        f"{int(row['source_index'])}:{sample_index}"
        for row in manifest
        for sample_index in range(int(protocol["samples_per_prompt"]))
    }
    pairing = pairing_summary(
        read_json(run_dir / "paired_initial_latents.json"),
        variants,
        expected_hash_keys,
    )
    summary = {
        "status": "completed",
        "run_dir": str(run_dir),
        "protocol_version": protocol["protocol_version"],
        "variants": variant_summaries,
        "paired_transitions_vs_baseline": transitions,
        "initial_latent_pairing": pairing,
        "best_variant": max(
            variants,
            key=lambda variant: (
                variant_summaries[variant]["accuracy"],
                -transitions[variant]["loss"],
            ),
        ),
        "selection_rule": "highest detector accuracy, then fewer losses versus same-seed baseline",
        "caveat": (
            "This Position A/B is a development experiment. Confirm a selected "
            "setting on an external spatial benchmark that was not used for layer "
            "or hyperparameter selection."
        ),
    }
    if not pairing["all_complete"] or not pairing["all_match"]:
        summary["status"] = "invalid_pairing"
    output = args.output or run_dir / "summary.json"
    output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "output": str(output),
                "best_variant": summary["best_variant"],
                "scores": {
                    variant: {
                        "correct": values["correct"],
                        "total": values["total"],
                        "accuracy": values["accuracy"],
                    }
                    for variant, values in variant_summaries.items()
                },
                "pairing": {
                    "all_complete": pairing["all_complete"],
                    "all_match": pairing["all_match"],
                },
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
