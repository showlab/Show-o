#!/usr/bin/env python3
"""Validate K=4 post-training checkpoints and benchmark artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


EXPECTED_IMAGES = 288
EXPECTED_SELF_CYCLE_IMAGES = 689
EXPECTED_SELF_CYCLE_QUESTIONS = 1087
EXPECTED_SEEDS = (20260810, 20260811, 20260812, 20260813)


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def count_jsonl(path: Path) -> int:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as handle:
        return sum(bool(line.strip()) for line in handle)


def validate_checkpoint(path: Path, expected_step: int) -> dict[str, Any]:
    import torch

    require(path.is_file(), f"missing checkpoint: {path}")
    require(path.stat().st_size >= 90_000_000, f"checkpoint too small: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    require(isinstance(checkpoint, dict), f"checkpoint is not a mapping: {path}")
    require(checkpoint.get("step") == expected_step, f"checkpoint step mismatch: {path}")
    adapter = checkpoint.get("drsa_adapter")
    require(isinstance(adapter, dict) and adapter, f"missing drsa_adapter: {path}")
    return {
        "path": str(path.resolve()),
        "step": expected_step,
        "bytes": path.stat().st_size,
        "adapter_tensors": len(adapter),
    }


def validate_training(train_root: Path, steps: tuple[int, ...]) -> dict[str, Any]:
    status = load_json(train_root / "status.json")
    require(status.get("status") == "completed", "training status is not completed")
    require(status.get("step") == 5000, "training did not finish step 5000")
    require(status.get("total_steps") == 5000, "unexpected training total_steps")
    rollout = status.get("rollout", {})
    require(rollout.get("model_evaluations") == 4, "training is not K=4")
    require(
        rollout.get("recurrent_euler_transitions") == 3,
        "training does not report three recurrent transitions",
    )
    checkpoints = [
        validate_checkpoint(
            train_root / "checkpoints" / f"step-{step:06d}.pt", step
        )
        for step in steps
    ]
    return {"status": "completed", "training": status, "checkpoints": checkpoints}


def validate_geneval(step_root: Path, variant: str) -> dict[str, Any]:
    run_dir = step_root / "geneval_position72"
    status = load_json(run_dir / "status.json")
    require(status.get("status") == "completed", f"GenEval status incomplete: {run_dir}")
    require(status.get("completed_images") == EXPECTED_IMAGES, "GenEval image count mismatch")
    require(status.get("total_images") == EXPECTED_IMAGES, "GenEval total mismatch")
    require((run_dir / "BENCHMARK_COMPLETED").is_file(), "missing GenEval completion marker")
    variant_root = run_dir / variant
    geneval_root = variant_root / "geneval"
    images = sorted(geneval_root.glob("*/samples/*.png"))
    require(len(images) == EXPECTED_IMAGES, f"expected 288 GenEval PNGs, found {len(images)}")
    prompt_dirs = {image.parent.parent.name for image in images}
    require(len(prompt_dirs) == 72, f"expected 72 GenEval prompts, found {len(prompt_dirs)}")
    results_path = variant_root / "results" / "results.jsonl"
    result_rows = count_jsonl(results_path)
    require(result_rows == EXPECTED_IMAGES, f"expected 288 GenEval rows, found {result_rows}")
    return {
        "status": "completed",
        "run_dir": str(run_dir.resolve()),
        "geneval_root": str(geneval_root.resolve()),
        "images": len(images),
        "prompts": len(prompt_dirs),
        "result_rows": result_rows,
    }


def validate_self_cycle(
    step_root: Path, variant: str, step: int, seeds: tuple[int, ...]
) -> list[dict[str, Any]]:
    validated = []
    for seed in seeds:
        run_dir = step_root / "self_cycle" / f"seed-{seed}" / variant
        status = load_json(run_dir / "status.json")
        summary = load_json(run_dir / "summary.json")
        require(status.get("status") == "completed", f"Self-cycle status incomplete: {run_dir}")
        require(status.get("completed_images") == EXPECTED_SELF_CYCLE_IMAGES, "Self-cycle image count mismatch")
        require(status.get("completed_questions") == EXPECTED_SELF_CYCLE_QUESTIONS, "Self-cycle question count mismatch")
        require(summary.get("status") == "completed", f"Self-cycle summary incomplete: {run_dir}")
        require(summary.get("adapter", {}).get("step") == step, f"Self-cycle adapter step mismatch: {run_dir}")
        protocol = summary.get("protocol", {})
        require(protocol.get("base_seed") == seed, f"Self-cycle seed mismatch: {run_dir}")
        require(protocol.get("num_inference_steps") == 50, f"Self-cycle inference-step mismatch: {run_dir}")
        require(protocol.get("guidance_scale") == 5.0, f"Self-cycle CFG mismatch: {run_dir}")
        metrics = summary.get("metrics", {})
        require(metrics.get("prompts", {}).get("count") == EXPECTED_SELF_CYCLE_IMAGES, "Self-cycle prompt metric count mismatch")
        require(metrics.get("questions", {}).get("questions") == EXPECTED_SELF_CYCLE_QUESTIONS, "Self-cycle question metric count mismatch")
        require((run_dir / "BENCHMARK_COMPLETED").is_file(), f"missing Self-cycle completion marker: {run_dir}")
        validated.append(
            {
                "status": "completed",
                "seed": seed,
                "run_dir": str(run_dir.resolve()),
                "images": EXPECTED_SELF_CYCLE_IMAGES,
                "questions": EXPECTED_SELF_CYCLE_QUESTIONS,
                "metrics": metrics,
            }
        )
    return validated


def validate_step(
    step_root: Path, variant: str, step: int, seeds: tuple[int, ...]
) -> dict[str, Any]:
    return {
        "status": "completed",
        "step": step,
        "variant": variant,
        "geneval": validate_geneval(step_root, variant),
        "self_cycle": validate_self_cycle(step_root, variant, step, seeds),
    }


def validate_vqa(output_dir: Path, variants: tuple[str, ...]) -> dict[str, Any]:
    status = load_json(output_dir / "status.json")
    summary = load_json(output_dir / "summary.json")
    require(status.get("status") == "completed", "VQAScore status is not completed")
    require(summary.get("status") == "completed", "VQAScore summary is not completed")
    require(summary.get("images_per_variant") == EXPECTED_IMAGES, "VQAScore per-variant image count mismatch")
    require(summary.get("total_images") == EXPECTED_IMAGES * len(variants), "VQAScore total image count mismatch")
    require(summary.get("total_vqa_queries") == 2 * EXPECTED_IMAGES * len(variants), "VQAScore query count mismatch")
    require(set(summary.get("variants", {})) == set(variants), "VQAScore variant set mismatch")
    rows = count_jsonl(output_dir / "per_image_scores.jsonl")
    require(rows == EXPECTED_IMAGES * len(variants), f"VQAScore row count mismatch: {rows}")
    return {
        "status": "completed",
        "output_dir": str(output_dir.resolve()),
        "images_per_variant": EXPECTED_IMAGES,
        "total_images": rows,
        "variants": summary["variants"],
        "pairwise": summary.get("pairwise", {}),
        "runtime": summary.get("runtime", {}),
    }


def parse_csv_ints(value: str) -> tuple[int, ...]:
    parsed = tuple(int(item) for item in value.split(",") if item)
    require(bool(parsed), "expected at least one integer")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    training = subparsers.add_parser("training")
    training.add_argument("train_root", type=Path)
    training.add_argument("--steps", default="500,1000,2500,5000")

    step = subparsers.add_parser("step")
    step.add_argument("step_root", type=Path)
    step.add_argument("--variant", required=True)
    step.add_argument("--step", type=int, required=True)
    step.add_argument("--seeds", default=",".join(map(str, EXPECTED_SEEDS)))

    vqa = subparsers.add_parser("vqa")
    vqa.add_argument("output_dir", type=Path)
    vqa.add_argument("--variants", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "training":
        result = validate_training(args.train_root.resolve(), parse_csv_ints(args.steps))
    elif args.command == "step":
        result = validate_step(
            args.step_root.resolve(),
            args.variant,
            args.step,
            parse_csv_ints(args.seeds),
        )
    else:
        variants = tuple(item for item in args.variants.split(",") if item)
        require(len(variants) >= 2, "expected at least two VQAScore variants")
        result = validate_vqa(args.output_dir.resolve(), variants)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
