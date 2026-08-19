#!/usr/bin/env python3
"""Validate and summarize the Flow x teacher-replay checkpoint benchmark matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from validate_zero_init_direct_cycle import (
    compact_self_cycle,
    geneval_metrics,
    load_json,
    require,
)


VARIANTS = {
    "mask_logits": (0.0, 0.0),
    "mask_flow": (0.25, 0.0),
    "mask_teacher_replay": (0.0, 0.20),
    "mask_flow_teacher_replay": (0.25, 0.20),
}
ROUTES = ("g_adapter", "lora")
EXPECTED_GENEVAL_IMAGES = 288
EXPECTED_GENEVAL_PROMPTS = 72


def parse_steps(value: str) -> tuple[int, ...]:
    steps = tuple(int(item) for item in value.split(",") if item)
    require(bool(steps), "step list cannot be empty")
    require(tuple(sorted(set(steps))) == steps, "steps must be unique and sorted")
    return steps


def count_jsonl(path: Path) -> int:
    if not path.is_file():
        raise FileNotFoundError(path)
    return sum(bool(line.strip()) for line in path.open(encoding="utf-8"))


def variant_roots(pure_root: Path, factorial_root: Path) -> dict[str, Path]:
    return {
        "mask_logits": pure_root,
        **{
            name: factorial_root / name
            for name in VARIANTS
            if name != "mask_logits"
        },
    }


def validate_training_matrix(
    roots: dict[str, Path],
    manifest: Path,
    max_steps: int,
    save_steps: tuple[int, ...],
    benchmark_steps: tuple[int, ...],
) -> dict[str, Any]:
    require(
        set(benchmark_steps).issubset(save_steps),
        "benchmark steps are not a subset of saved checkpoints",
    )
    matrix: dict[str, Any] = {}
    for name, root in roots.items():
        expected_flow, expected_replay = VARIANTS[name]
        config = load_json(root / "queue_config.json")
        require(int(config.get("max_steps", -1)) == max_steps, f"{name}: max_steps")
        require(
            tuple(config.get("save_steps", [])) == save_steps,
            f"{name}: queue save-step mismatch",
        )
        require(
            Path(str(config.get("manifest", ""))).resolve() == manifest.resolve(),
            f"{name}: queue manifest mismatch",
        )
        weights = config.get("weights", {})
        require(float(weights.get("u_task", -1.0)) == 0.0, f"{name}: U is enabled")
        require(
            float(weights.get("flow", -1.0)) == expected_flow,
            f"{name}: flow weight mismatch",
        )
        require(
            float(weights.get("teacher_retention", -1.0)) == expected_replay,
            f"{name}: teacher-replay weight mismatch",
        )
        route_rows: dict[str, Any] = {}
        for route in ROUTES:
            train_root = root / route
            status = load_json(train_root / "status.json")
            require(status.get("status") == "completed", f"{name}/{route}: incomplete")
            require(int(status.get("step", -1)) == max_steps, f"{name}/{route}: step")
            require(
                int(status.get("manifest_rows", -1)) == max_steps,
                f"{name}/{route}: manifest rows",
            )
            require(
                status.get("initialization")
                == "original_showo2_zero_output_adaptation",
                f"{name}/{route}: initialization",
            )
            require(
                status.get("adaptation", {}).get("route") == route,
                f"{name}/{route}: route mismatch",
            )
            require(
                tuple(
                    status.get("checkpoint_schedule", {}).get("save_steps", [])
                )
                == save_steps,
                f"{name}/{route}: status save-step mismatch",
            )
            loss_weights = status.get("loss_weights", {})
            require(
                float(loss_weights.get("flow", -1.0)) == expected_flow,
                f"{name}/{route}: trained flow weight",
            )
            require(
                float(loss_weights.get("teacher_retention", -1.0))
                == expected_replay,
                f"{name}/{route}: trained teacher-replay weight",
            )
            observed = tuple(
                sorted(
                    int(path.stem.removeprefix("step-"))
                    for path in (train_root / "checkpoints").glob("step-*.pt")
                )
            )
            require(observed == save_steps, f"{name}/{route}: checkpoint files")
            checkpoints: dict[str, Any] = {}
            for step in benchmark_steps:
                checkpoint = train_root / "checkpoints" / f"step-{step:06d}.pt"
                payload = torch.load(
                    checkpoint, map_location="cpu", weights_only=True, mmap=True
                )
                require(int(payload.get("step", -1)) == step, f"{checkpoint}: step")
                require(
                    payload.get("adaptation_config", {}).get("route") == route,
                    f"{checkpoint}: route",
                )
                require(bool(payload.get("adaptation_state")), f"{checkpoint}: state")
                checkpoints[str(step)] = {
                    "path": str(checkpoint.resolve()),
                    "bytes": checkpoint.stat().st_size,
                }
            route_rows[route] = {
                "trainable_parameters": int(
                    status["adaptation"]["trainable_parameters"]
                ),
                "checkpoints": checkpoints,
            }
        difference = abs(
            route_rows["g_adapter"]["trainable_parameters"]
            - route_rows["lora"]["trainable_parameters"]
        )
        reference = route_rows["g_adapter"]["trainable_parameters"]
        require(
            difference <= 12 and difference / reference <= 0.00001,
            f"{name}: route parameter mismatch",
        )
        matrix[name] = {
            "root": str(root.resolve()),
            "flow_weight": expected_flow,
            "teacher_replay_weight": expected_replay,
            "routes": route_rows,
            "absolute_parameter_difference": difference,
        }
    return matrix


def validate_checkpoint_evaluation(
    eval_root: Path,
    variant: str,
    checkpoint: Path,
    route: str,
    step: int,
    seed: int,
    compact: dict[str, Any],
    self_cycle_dataset: Path,
) -> dict[str, Any]:
    require((eval_root / "QUEUE_COMPLETED").is_file(), f"incomplete eval: {eval_root}")
    variant_config = load_json(eval_root / "variants.json")
    configured = Path(variant_config[variant]["drsa_checkpoint"]).resolve()
    require(configured == checkpoint.resolve(), f"{eval_root}: checkpoint path")
    geneval_root = eval_root / "geneval_position72" / variant
    images = sorted((geneval_root / "geneval").glob("*/samples/*.png"))
    require(len(images) == EXPECTED_GENEVAL_IMAGES, f"{eval_root}: GenEval images")
    results = geneval_root / "results/results.jsonl"
    require(count_jsonl(results) == EXPECTED_GENEVAL_IMAGES, f"{eval_root}: rows")
    geneval = geneval_metrics(results)

    self_root = eval_root / "self_cycle" / f"seed-{seed}" / variant
    summary = load_json(self_root / "summary.json")
    protocol = load_json(self_root / "protocol.json")
    require(summary.get("status") == "completed", f"{self_root}: summary")
    require(
        int(summary.get("adapter", {}).get("step", -1)) == step,
        f"{self_root}: adapter step",
    )
    require(
        summary.get("adapter", {}).get("route", {}).get("route") == route,
        f"{self_root}: adapter route",
    )
    require(
        Path(protocol["dataset"]).resolve() == self_cycle_dataset.resolve(),
        f"{self_root}: dataset",
    )
    require(
        protocol.get("record_indices") == compact["record_indices"],
        f"{self_root}: compact indices",
    )
    require(int(protocol.get("num_inference_steps", -1)) == 50, f"{self_root}: steps")
    require(int(protocol.get("base_seed", -1)) == seed, f"{self_root}: seed")
    require(
        protocol.get("generation_prompt_hidden_from_mmu") is True,
        f"{self_root}: prompt leakage",
    )
    require(
        int(summary["metrics"]["prompts"]["count"])
        == int(compact["selected_records"]),
        f"{self_root}: prompt count",
    )
    require(
        int(summary["metrics"]["questions"]["questions"])
        == int(compact["selected_questions"]),
        f"{self_root}: question count",
    )
    require(
        summary.get("adapter", {}).get("disabled_for_structured_qa") is True,
        f"{self_root}: adaptation active during QA",
    )
    return {
        "checkpoint": str(checkpoint.resolve()),
        "evaluation_root": str(eval_root.resolve()),
        "geneval": geneval,
        "self_cycle": compact_self_cycle(summary),
    }


def metric_view(evaluation: dict[str, Any]) -> dict[str, float]:
    return {
        "geneval_image_accuracy": evaluation["geneval"]["image_accuracy"],
        "geneval_prompt_pass": evaluation["geneval"]["prompt_pass"],
        "self_cycle_accuracy": evaluation["self_cycle"]["accuracy"],
        "self_cycle_mean_token_f1": evaluation["self_cycle"]["mean_token_f1"],
        "self_cycle_prompt_hard_score": evaluation["self_cycle"][
            "prompt_hard_score"
        ],
        "self_cycle_prompt_soft_score": evaluation["self_cycle"][
            "prompt_soft_score"
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pure-root", type=Path, required=True)
    parser.add_argument("--factorial-root", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--self-cycle-dataset", type=Path, required=True)
    parser.add_argument("--compact-report", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=17100)
    parser.add_argument("--save-steps", required=True)
    parser.add_argument("--benchmark-steps", required=True)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    save_steps = parse_steps(args.save_steps)
    benchmark_steps = parse_steps(args.benchmark_steps)
    compact = load_json(args.compact_report)
    require(
        compact.get("protocol") == "self-cycle-compact96-stratified-v1",
        "compact protocol mismatch",
    )
    roots = variant_roots(args.pure_root.resolve(), args.factorial_root.resolve())
    training = validate_training_matrix(
        roots,
        args.train_manifest.resolve(),
        args.max_steps,
        save_steps,
        benchmark_steps,
    )
    output: dict[str, Any] = {
        "status": "preflight_completed" if args.preflight_only else "completed",
        "design": "Flow x frozen-teacher replay 2x2",
        "protocol": {
            "save_steps": list(save_steps),
            "benchmark_steps": list(benchmark_steps),
            "routes": list(ROUTES),
            "geneval_prompts": EXPECTED_GENEVAL_PROMPTS,
            "geneval_images": EXPECTED_GENEVAL_IMAGES,
            "self_cycle": compact,
            "self_cycle_inference_steps": 50,
            "self_cycle_seed": args.seed,
        },
        "training": training,
    }
    if not args.preflight_only:
        output["results"] = {}
        for name, root in roots.items():
            output["results"][name] = {}
            for route in ROUTES:
                output["results"][name][route] = {}
                for step in benchmark_steps:
                    variant = f"{name}_{route}_step_{step:06d}"
                    eval_root = (
                        args.benchmark_root
                        / name
                        / route
                        / f"step-{step:06d}"
                    )
                    checkpoint = (
                        root / route / "checkpoints" / f"step-{step:06d}.pt"
                    )
                    evaluation = validate_checkpoint_evaluation(
                        eval_root,
                        variant,
                        checkpoint,
                        route,
                        step,
                        args.seed,
                        compact,
                        args.self_cycle_dataset.resolve(),
                    )
                    evaluation["metrics"] = metric_view(evaluation)
                    output["results"][name][route][str(step)] = evaluation
        output["effects"] = {}
        for route in ROUTES:
            output["effects"][route] = {}
            for step in benchmark_steps:
                key = str(step)
                values = {
                    name: output["results"][name][route][key]["metrics"]
                    for name in VARIANTS
                }
                output["effects"][route][key] = {}
                for metric in values["mask_logits"]:
                    y00 = values["mask_logits"][metric]
                    y10 = values["mask_flow"][metric]
                    y01 = values["mask_teacher_replay"][metric]
                    y11 = values["mask_flow_teacher_replay"][metric]
                    output["effects"][route][key][metric] = {
                        "flow_main_effect": ((y10 - y00) + (y11 - y01)) / 2.0,
                        "teacher_replay_main_effect": (
                            (y01 - y00) + (y11 - y10)
                        )
                        / 2.0,
                        "flow_x_teacher_replay_interaction": y11 - y10 - y01 + y00,
                    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
