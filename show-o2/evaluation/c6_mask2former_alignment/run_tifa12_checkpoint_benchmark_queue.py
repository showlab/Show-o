#!/usr/bin/env python3
"""Benchmark six checkpoints from the four-route TIFA-12 self-cycle run."""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

import torch


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import run_multiaxis_ab_checkpoint_queue as base


WORKSPACE = Path("/root/autodl-tmp")
TIFA_TRAIN_MANIFEST = WORKSPACE / "data/showo2_self_cycle/tifa12_external_v1/train.jsonl"
TIFA_DATA_MANIFEST = WORKSPACE / "data/showo2_self_cycle/tifa12_external_v1/manifest.json"
TIFA_LOCK = WORKSPACE / "data/showo2_self_cycle/tifa12_external_v1/checkpoints.lock.json"
TIFA_SAVE_STEPS = (100, 500, 1000, 2000, 4000, 6000, 8000, 10000, 17000, 25000, 34012)
DEFAULT_BENCHMARK_STEPS = "100,500,1000,6000,17000,34012"
EXPECTED_ROUTE_LABELS = {
    "tifa_s1_m": {
        "g_adapter": "s1_m_g_adapter",
        "lora": "s1_m_lora",
    },
    "tifa_s1_mf": {
        "g_adapter": "s1_mf_g_adapter",
        "lora": "s1_mf_lora",
    },
}

# The shared runner uses module-level schedules for path and protocol validation.
base.SAVE_STEPS = TIFA_SAVE_STEPS


def source_views(run_root: Path, training_root: Path) -> dict[str, Path]:
    roots: dict[str, Path] = {}
    for variant, routes in EXPECTED_ROUTE_LABELS.items():
        root = run_root / "checkpoint_sources" / variant
        root.mkdir(parents=True, exist_ok=True)
        roots[variant] = root
        for route, label in routes.items():
            source = (training_root / label).resolve()
            if not source.is_dir():
                raise FileNotFoundError(source)
            link = root / route
            if link.is_symlink():
                if link.resolve() != source:
                    raise RuntimeError(f"source view mismatch: {link} -> {link.resolve()} != {source}")
            elif link.exists():
                raise RuntimeError(f"source view path is not a symlink: {link}")
            else:
                link.symlink_to(source, target_is_directory=True)
    return roots


def locked_starts() -> dict[str, dict[str, Any]]:
    lock = base.load_json(TIFA_LOCK)
    starts = lock.get("starts", [])
    if len(starts) != 4:
        raise RuntimeError(f"expected four locked starts, found {len(starts)}")
    result: dict[str, dict[str, Any]] = {}
    for start in starts:
        label = f"{str(start['branch']).lower().replace('-', '_')}_{start['adaptation_route']}"
        if label in result:
            raise RuntimeError(f"duplicate locked start: {label}")
        result[label] = start
    return result


class TifaBenchmarkQueue(base.Queue):
    def write_status(self, status: str, phase: str, detail: str = "") -> None:
        super().write_status(status, phase, detail)
        payload = base.load_json(self.status_path)
        payload["source_training_root"] = str(self.args.training_root.resolve())
        payload["matrix"] = {
            "variants": [variant.name for variant in self.variants],
            "routes": list(base.ROUTES),
            "model_checkpoint_points": len(self.variants) * len(base.ROUTES) * len(self.steps),
        }
        payload["protocol"].update(
            {
                "saved_steps": list(TIFA_SAVE_STEPS),
                "vqascore": (
                    f"{len(self.variants)} variants paired per route/checkpoint on "
                    "288 Position-72 images"
                ),
                "training": (
                    "external TIFA-12 frozen-U strict-gate self-cycle; "
                    "generation prompt hidden from understanding"
                ),
            }
        )
        base.atomic_json(self.status_path, payload)

    def wait_for_training(self) -> None:
        root = self.args.training_root.resolve()
        while not (root / "QUEUE_COMPLETED").is_file():
            if (root / "QUEUE_FAILED").is_file():
                raise RuntimeError(f"upstream TIFA-12 training failed: {root}")
            queue = base.load_json(root / "queue_status.json")
            self.current_task = None
            self.active_status_paths = [
                str(root / label / "status.json")
                for routes in EXPECTED_ROUTE_LABELS.values()
                for label in routes.values()
            ]
            tasks = queue.get("tasks", [])
            completed = sum(task.get("status") == "completed" for task in tasks)
            self.write_status(
                "waiting",
                "waiting_for_training",
                (
                    f"TIFA-12 training is still running; current={queue.get('current', 'unknown')} "
                    f"completed_routes={completed}/4"
                ),
            )
            time.sleep(self.args.wait_seconds)
        queue = base.load_json(root / "queue_status.json")
        if queue.get("status") != "completed":
            raise RuntimeError(f"QUEUE_COMPLETED exists but queue status is {queue.get('status')!r}")

    def preflight(self) -> None:
        output: dict[str, Any] = {
            "status": "running",
            "source_training_root": str(self.args.training_root.resolve()),
            "benchmark_steps": list(self.steps),
            "saved_steps": list(TIFA_SAVE_STEPS),
            "variants": {},
        }
        base.atomic_json(self.run_root / "preflight.json", output)
        for required in (
            base.GENEVAL_PYTHON,
            base.SHOWO_PYTHON,
            base.VQA_PYTHON,
            base.POSITION_MANIFEST,
            base.SELF_CYCLE_DATASET,
            base.VQA_MODEL / "model.safetensors.index.json",
            base.ROOT / "Wan2.1_VAE.pth",
            TIFA_TRAIN_MANIFEST,
            TIFA_DATA_MANIFEST,
            TIFA_LOCK,
        ):
            if not required.exists():
                raise FileNotFoundError(required)

        training_queue = base.load_json(self.args.training_root / "queue_status.json")
        if training_queue.get("status") != "completed":
            raise RuntimeError("TIFA-12 training queue is not completed")
        if int(training_queue.get("max_steps_per_route", -1)) != TIFA_SAVE_STEPS[-1]:
            raise RuntimeError("TIFA-12 max step mismatch")
        if Path(training_queue.get("data_manifest", "")).resolve() != TIFA_DATA_MANIFEST.resolve():
            raise RuntimeError("TIFA-12 data manifest mismatch")

        starts = locked_starts()
        queue_starts = {
            task["label"]: task.get("start", {}) for task in training_queue.get("tasks", [])
        }
        if set(queue_starts) != set(starts):
            raise RuntimeError("training queue starts do not match the locked four routes")

        for variant in self.variants:
            routes: dict[str, Any] = {}
            for route in base.ROUTES:
                label = EXPECTED_ROUTE_LABELS[variant.name][route]
                route_root = variant.root / route
                expected_root = (self.args.training_root / label).resolve()
                if route_root.resolve() != expected_root:
                    raise RuntimeError(f"{variant.name}/{route}: source route mismatch")
                status = base.load_json(route_root / "status.json")
                if status.get("status") != "completed" or int(status.get("step", -1)) != TIFA_SAVE_STEPS[-1]:
                    raise RuntimeError(f"{label}: incomplete training")
                if status.get("label") != label:
                    raise RuntimeError(f"{label}: status label mismatch")
                if status.get("adaptation", {}).get("route") != route:
                    raise RuntimeError(f"{label}: adaptation route mismatch")
                if Path(status.get("manifest", "")).resolve() != TIFA_TRAIN_MANIFEST.resolve():
                    raise RuntimeError(f"{label}: train manifest mismatch")
                if tuple(status.get("save_steps", [])) != TIFA_SAVE_STEPS:
                    raise RuntimeError(f"{label}: save schedule mismatch")
                if int(status.get("dataset_rows", -1)) != TIFA_SAVE_STEPS[-1]:
                    raise RuntimeError(f"{label}: dataset size mismatch")
                if status.get("gate_policy") != "strict_per_question":
                    raise RuntimeError(f"{label}: gate policy mismatch")
                if status.get("generation_caption_in_u") is not False:
                    raise RuntimeError(f"{label}: generation prompt leakage policy mismatch")
                if status.get("understanding_input") != "predicted_image+question+choices_only":
                    raise RuntimeError(f"{label}: understanding input mismatch")
                if int(status.get("rollout_steps", -1)) != 4:
                    raise RuntimeError(f"{label}: rollout step mismatch")

                locked = starts[label]
                queued = queue_starts[label]
                if locked != queued:
                    raise RuntimeError(f"{label}: queued start differs from lock")
                if status.get("init_checkpoint_sha256") != locked.get("sha256"):
                    raise RuntimeError(f"{label}: initial checkpoint SHA256 mismatch")

                observed = tuple(
                    sorted(
                        int(path.stem.removeprefix("step-"))
                        for path in (route_root / "checkpoints").glob("step-*.pt")
                    )
                )
                if observed != TIFA_SAVE_STEPS:
                    raise RuntimeError(f"{label}: checkpoint files mismatch: {observed}")

                checkpoints: dict[str, Any] = {}
                for step in self.steps:
                    checkpoint = base.checkpoint_path(variant, route, step)
                    payload = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
                    if int(payload.get("step", -1)) != step:
                        raise RuntimeError(f"{checkpoint}: internal step mismatch")
                    config = payload.get("adaptation_config", {})
                    if config.get("route") != route:
                        raise RuntimeError(f"{checkpoint}: internal route mismatch")
                    if not payload.get("adaptation_state"):
                        raise RuntimeError(f"{checkpoint}: empty adaptation state")
                    training = payload.get("training_config", {})
                    if int(training.get("max_steps", -1)) != TIFA_SAVE_STEPS[-1]:
                        raise RuntimeError(f"{checkpoint}: training max step mismatch")
                    if int(training.get("rollout_steps", -1)) != 4:
                        raise RuntimeError(f"{checkpoint}: rollout step mismatch")
                    if Path(training.get("manifest", "")).resolve() != TIFA_TRAIN_MANIFEST.resolve():
                        raise RuntimeError(f"{checkpoint}: payload manifest mismatch")
                    if payload.get("init_checkpoint_sha256") != locked.get("sha256"):
                        raise RuntimeError(f"{checkpoint}: payload initial SHA256 mismatch")
                    checkpoints[str(step)] = {
                        "path": str(checkpoint.resolve()),
                        "bytes": checkpoint.stat().st_size,
                        "optimizer_updates": int(payload.get("optimizer_updates", 0)),
                    }
                    del payload

                routes[route] = {
                    "label": label,
                    "source_root": str(expected_root),
                    "initial_checkpoint": locked,
                    "trainable_parameters": int(status["adaptation"]["trainable_parameters"]),
                    "checkpoints": checkpoints,
                }

            difference = abs(
                routes["g_adapter"]["trainable_parameters"]
                - routes["lora"]["trainable_parameters"]
            )
            if difference > 12:
                raise RuntimeError(f"{variant.name}: route parameter mismatch is {difference}")
            output["variants"][variant.name] = {
                "label": variant.label,
                "protocol": variant.protocol,
                "inherited_flow_weight": variant.flow_weight,
                "routes": routes,
                "route_parameter_difference": difference,
            }

        output["status"] = "completed"
        output["completed_at"] = base.now()
        output["model_checkpoint_points"] = len(self.variants) * len(base.ROUTES) * len(self.steps)
        base.atomic_json(self.run_root / "preflight.json", output)

    def annotate_summary(self) -> None:
        path = self.run_root / "comparison_summary.json"
        summary = base.load_json(path)
        if summary.get("status") != "completed":
            raise RuntimeError("cannot annotate incomplete comparison summary")
        summary.update(
            {
                "design": "TIFA-12 frozen-U self-cycle continuation of S1 Mask and Mask+Flow",
                "source_training_root": str(self.args.training_root.resolve()),
                "benchmark_steps": list(self.steps),
                "model_checkpoint_points": len(self.variants) * len(base.ROUTES) * len(self.steps),
            }
        )
        base.atomic_json(path, summary)

    def run(self) -> None:
        super().run()
        self.annotate_summary()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--benchmark-steps", default=DEFAULT_BENCHMARK_STEPS)
    parser.add_argument("--eval-seed", type=int, default=20260810)
    parser.add_argument("--wait-seconds", type=int, default=30)
    parser.add_argument("--retry-seconds", type=int, default=30)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--vqa-batch-size", type=int, default=8)
    parser.add_argument("--plan-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.training_root = args.training_root.resolve()
    args.wait_root = args.training_root
    args.wait_marker = "QUEUE_COMPLETED"
    args.wait_label = "formal four-route external TIFA-12 self-cycle training"
    steps = base.parse_steps(args.benchmark_steps)
    views = source_views(args.run_root.resolve(), args.training_root)
    variants = (
        base.Variant(
            "tifa_s1_m",
            "S1 Mask-logits after TIFA-12 self-cycle",
            views["tifa_s1_m"],
            TIFA_TRAIN_MANIFEST,
            "orthogonal-axis Mask initialization plus frozen-U TIFA-12 self-cycle",
            0.0,
        ),
        base.Variant(
            "tifa_s1_mf",
            "S1 Mask-logits + Flow after TIFA-12 self-cycle",
            views["tifa_s1_mf"],
            TIFA_TRAIN_MANIFEST,
            "orthogonal-axis Mask+Flow initialization plus frozen-U TIFA-12 self-cycle",
            0.25,
        ),
    )
    queue = TifaBenchmarkQueue(args, variants, steps)
    expected_tasks = 3 + 10 * len(steps)
    if len(queue.planned) != expected_tasks:
        raise RuntimeError(f"unexpected task count: {len(queue.planned)} != {expected_tasks}")
    if args.plan_only:
        counts = Counter(
            "eval" if task.startswith("eval__") else
            "vqa" if task.startswith("vqa__") else
            "vsr" if task.startswith("vsr__") or task == "vsr_baseline_dev" else task
            for task in queue.planned
        )
        print(json.dumps({
            "status": "planned",
            "steps": list(steps),
            "model_checkpoint_points": len(variants) * len(base.ROUTES) * len(steps),
            "total_tasks": len(queue.planned),
            "task_counts": dict(counts),
        }, indent=2, sort_keys=True))
        return 0
    try:
        queue.run()
        return 0
    except BaseException as error:
        queue.error = f"{type(error).__name__}: {error}"
        (queue.run_root / "BENCHMARK_QUEUE_COMPLETED").unlink(missing_ok=True)
        queue.write_status("failed", "failed", traceback.format_exc())
        (queue.run_root / "BENCHMARK_QUEUE_FAILED").touch()
        raise


if __name__ == "__main__":
    raise SystemExit(main())
