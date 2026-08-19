#!/usr/bin/env python3
"""Validate and summarize a zero-init generation-adaptation ablation."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch


EXPECTED_GENEVAL_IMAGES = 288
EXPECTED_GENEVAL_PROMPTS = 72
EXPECTED_SELF_CYCLE_IMAGES = 689
EXPECTED_SELF_CYCLE_QUESTIONS = 1087


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def count_jsonl(path: Path) -> int:
    if not path.is_file():
        raise FileNotFoundError(path)
    return sum(bool(line.strip()) for line in path.open(encoding="utf-8"))


def geneval_metrics(results_path: Path) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in results_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    require(len(rows) == EXPECTED_GENEVAL_IMAGES, "GenEval result count mismatch")
    by_prompt: dict[str, list[bool]] = defaultdict(list)
    by_relation: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        correct = bool(row["correct"])
        by_prompt[str(row["prompt"])].append(correct)
        metadata = json.loads(row["metadata"])
        relation = str(metadata["include"][1]["position"][0])
        by_relation[relation].append(correct)
    require(len(by_prompt) == EXPECTED_GENEVAL_PROMPTS, "GenEval prompt count mismatch")
    return {
        "image_accuracy": sum(bool(row["correct"]) for row in rows) / len(rows),
        "prompt_pass": sum(all(values) for values in by_prompt.values()) / len(by_prompt),
        "by_relation": {
            relation: {
                "correct": sum(values),
                "total": len(values),
                "accuracy": sum(values) / len(values),
            }
            for relation, values in sorted(by_relation.items())
        },
    }


def compact_self_cycle(summary: dict[str, Any]) -> dict[str, Any]:
    questions = summary["metrics"]["questions"]
    prompts = summary["metrics"]["prompts"]
    return {
        "correct": int(questions["correct"]),
        "questions": int(questions["questions"]),
        "accuracy": float(questions["accuracy"]),
        "mean_token_f1": float(questions["mean_token_f1"]),
        "prompt_hard_score": float(prompts["hard_score"]),
        "prompt_soft_score": float(prompts["soft_score"]),
    }


def validate_training(
    train_root: Path,
    step: int,
    *,
    expected_route: str,
    expected_u_cycle: bool,
    expected_rollout_steps: int,
    expected_flow_weight: float,
    expected_retention_weight: float,
    expected_manifest: Path,
    expected_manifest_rows: int,
    expected_save_steps: tuple[int, ...],
) -> dict[str, Any]:
    status = load_json(train_root / "status.json")
    metadata = load_json(train_root / "metadata.json")
    require(status.get("status") == "completed", "direct-cycle training is incomplete")
    require(int(status.get("step", -1)) == step, "training step mismatch")
    require(int(status.get("total_steps", -1)) == step, "training total-step mismatch")
    require(
        Path(str(status.get("manifest", ""))).resolve() == expected_manifest.resolve(),
        "training manifest mismatch",
    )
    require(
        int(status.get("manifest_rows", -1)) == expected_manifest_rows,
        "training manifest-row count mismatch",
    )
    schedule = status.get("checkpoint_schedule", {})
    require(schedule.get("mode") == "explicit", "checkpoint schedule is not explicit")
    require(
        tuple(int(value) for value in schedule.get("save_steps", []))
        == expected_save_steps,
        "checkpoint schedule mismatch",
    )
    require(
        status.get("initialization") == "original_showo2_zero_output_adaptation",
        "training was not initialized from the zero-output adapter",
    )
    resume_checkpoint = status.get("resume_checkpoint")
    if resume_checkpoint is not None:
        resume_path = Path(str(resume_checkpoint)).resolve()
        require(
            resume_path.parent == (train_root / "checkpoints").resolve(),
            "training resumed from outside the zero-init direct-cycle run",
        )
    adaptation = status.get("adaptation", {})
    require(
        adaptation.get("route") == expected_route,
        f"training route mismatch: expected={expected_route} "
        f"actual={adaptation.get('route')}",
    )
    actual_u_cycle = bool(status.get("u_cycle", {}).get("enabled"))
    require(actual_u_cycle == expected_u_cycle, "frozen-U cycle mode mismatch")
    if expected_u_cycle:
        require(
            status.get("u_cycle", {}).get("gate_policy") == "strict",
            "frozen-U cycle is not using the strict gate",
        )
    require(
        status.get("rollout", {}).get("model_evaluations") == expected_rollout_steps,
        "training rollout-step mismatch",
    )
    weights = status.get("loss_weights", {})
    require(float(weights.get("u_task", 0.0)) == float(expected_u_cycle), "U-task weight mismatch")
    require(
        float(weights.get("flow", -1.0)) == expected_flow_weight,
        "flow-loss weight mismatch",
    )
    require(
        float(weights.get("teacher_retention", -1.0)) == expected_retention_weight,
        "teacher-retention weight mismatch",
    )
    require(float(weights.get("mask_bce", 0.0)) > 0.0, "Mask-logits BCE is disabled")
    require(float(weights.get("mask_dice", 0.0)) > 0.0, "Mask-logits Dice is disabled")
    require(float(weights.get("relation_pass", 0.0)) > 0.0, "relation loss is disabled")
    require(metadata.get("base_model_frozen") is True, "base model was not frozen")
    require(metadata.get("u_frozen") is True, "U route was not frozen")
    require(metadata.get("understanding_adapter") is None, "unexpected understanding adapter")
    checkpoint = train_root / "checkpoints" / f"step-{step:06d}.pt"
    require(checkpoint.is_file(), f"missing checkpoint: {checkpoint}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    require(int(payload.get("step", -1)) == step, "checkpoint step mismatch")
    require(
        payload.get("adaptation_config", {}).get("route") == expected_route,
        "checkpoint route mismatch",
    )
    require(bool(payload.get("adaptation_state")), "checkpoint has no adaptation state")
    observed_checkpoint_steps = tuple(
        sorted(
            int(path.stem.removeprefix("step-"))
            for path in (train_root / "checkpoints").glob("step-*.pt")
        )
    )
    require(
        observed_checkpoint_steps == expected_save_steps,
        "checkpoint files do not exactly match the requested save steps",
    )
    return {
        "status": status,
        "metadata": metadata,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "internal_resume_checkpoint": resume_checkpoint,
    }


def validate_evaluation(eval_root: Path, variant: str, step: int, seed: int) -> dict[str, Any]:
    geneval_root = eval_root / "geneval_position72" / variant
    geneval_status = load_json(eval_root / "geneval_position72/status.json")
    require(geneval_status.get("status") == "completed", "GenEval status is incomplete")
    images = sorted((geneval_root / "geneval").glob("*/samples/*.png"))
    require(len(images) == EXPECTED_GENEVAL_IMAGES, "GenEval image count mismatch")
    results_path = geneval_root / "results/results.jsonl"
    require(count_jsonl(results_path) == EXPECTED_GENEVAL_IMAGES, "GenEval row count mismatch")

    self_root = eval_root / "self_cycle" / f"seed-{seed}" / variant
    self_status = load_json(self_root / "status.json")
    self_summary = load_json(self_root / "summary.json")
    require(self_status.get("status") == "completed", "Self-cycle status is incomplete")
    require(self_summary.get("status") == "completed", "Self-cycle summary is incomplete")
    require(
        int(self_summary.get("adapter", {}).get("step", -1)) == step,
        "Self-cycle checkpoint step mismatch",
    )
    require(
        int(self_summary.get("protocol", {}).get("base_seed", -1)) == seed,
        "Self-cycle seed mismatch",
    )
    require(
        int(self_summary["metrics"]["prompts"]["count"]) == EXPECTED_SELF_CYCLE_IMAGES,
        "Self-cycle image count mismatch",
    )
    require(
        int(self_summary["metrics"]["questions"]["questions"])
        == EXPECTED_SELF_CYCLE_QUESTIONS,
        "Self-cycle question count mismatch",
    )
    require(
        self_summary.get("adapter", {}).get("disabled_for_structured_qa") is True,
        "G-adapter was not disabled during structured QA",
    )
    return {
        "geneval": geneval_metrics(results_path),
        "self_cycle": compact_self_cycle(self_summary),
        "geneval_results": str(results_path.resolve()),
        "self_cycle_summary": str((self_root / "summary.json").resolve()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--warmup-comparison", type=Path, required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument(
        "--expected-route", choices=("g_adapter", "lora"), default="g_adapter"
    )
    parser.add_argument("--expected-u-cycle", choices=("on", "off"), required=True)
    parser.add_argument("--expected-rollout-steps", type=int, required=True)
    parser.add_argument("--expected-flow-weight", type=float, required=True)
    parser.add_argument("--expected-retention-weight", type=float, required=True)
    parser.add_argument("--expected-manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-rows", type=int, required=True)
    parser.add_argument("--expected-save-steps", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    expected_save_steps = tuple(
        int(value) for value in args.expected_save_steps.split(",") if value
    )
    require(bool(expected_save_steps), "expected-save-steps cannot be empty")
    training = validate_training(
        args.train_root.resolve(),
        args.step,
        expected_route=args.expected_route,
        expected_u_cycle=args.expected_u_cycle == "on",
        expected_rollout_steps=args.expected_rollout_steps,
        expected_flow_weight=args.expected_flow_weight,
        expected_retention_weight=args.expected_retention_weight,
        expected_manifest=args.expected_manifest,
        expected_manifest_rows=args.expected_manifest_rows,
        expected_save_steps=expected_save_steps,
    )
    evaluation = validate_evaluation(
        args.eval_root.resolve(), args.variant, args.step, args.seed
    )
    warmup = load_json(args.warmup_comparison.resolve())
    require(warmup.get("status") == "completed", "warmup comparison is incomplete")
    comparisons = {
        "baseline": {
            "geneval": warmup["baseline"]["geneval_position72"],
            "self_cycle": compact_self_cycle(warmup["baseline"]["self_cycle"]),
        },
        "warmup_step_100": {
            "geneval": warmup["checkpoints"]["100"]["geneval_position72"],
            "self_cycle": compact_self_cycle(
                warmup["checkpoints"]["100"]["self_cycle"]
            ),
        },
        "warmup_step_200": {
            "geneval": warmup["checkpoints"]["200"]["geneval_position72"],
            "self_cycle": compact_self_cycle(
                warmup["checkpoints"]["200"]["self_cycle"]
            ),
        },
        args.experiment_name: {
            "geneval": evaluation["geneval"],
            "self_cycle": evaluation["self_cycle"],
        },
    }
    baseline = comparisons["baseline"]
    direct = comparisons[args.experiment_name]
    deltas = {
        "geneval_image_accuracy": (
            direct["geneval"]["image_accuracy"]
            - baseline["geneval"]["image_accuracy"]
        ),
        "geneval_prompt_pass": (
            direct["geneval"]["prompt_pass"]
            - baseline["geneval"]["prompt_pass"]
        ),
        "self_cycle_accuracy": (
            direct["self_cycle"]["accuracy"]
            - baseline["self_cycle"]["accuracy"]
        ),
        "self_cycle_token_f1": (
            direct["self_cycle"]["mean_token_f1"]
            - baseline["self_cycle"]["mean_token_f1"]
        ),
    }
    output = {
        "status": "completed",
        "protocol": {
            "ablation": args.experiment_name,
            "warmup_checkpoint": None,
            "training_step": args.step,
            "training_route": args.expected_route,
            "training_u_cycle": args.expected_u_cycle == "on",
            "training_rollout_steps": args.expected_rollout_steps,
            "training_flow_weight": args.expected_flow_weight,
            "training_retention_weight": args.expected_retention_weight,
            "training_manifest": str(args.expected_manifest.resolve()),
            "training_manifest_rows": args.expected_manifest_rows,
            "training_save_steps": list(expected_save_steps),
            "self_cycle_seed": args.seed,
            "geneval_images": EXPECTED_GENEVAL_IMAGES,
            "self_cycle_images": EXPECTED_SELF_CYCLE_IMAGES,
            "self_cycle_questions": EXPECTED_SELF_CYCLE_QUESTIONS,
        },
        "training": training,
        "evaluation": evaluation,
        "comparisons": comparisons,
        "delta_vs_baseline": deltas,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
