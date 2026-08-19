#!/usr/bin/env python3
"""Run the single-axis vs multi-axis Mask-logits checkpoint benchmark matrix."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    omp_threads_valid = int(os.environ.get("OMP_NUM_THREADS", "0")) > 0
except ValueError:
    omp_threads_valid = False
if not omp_threads_valid:
    os.environ["OMP_NUM_THREADS"] = "4"

import torch


ROOT = Path("/root/autodl-tmp/Show-o/show-o2")
WORKSPACE = Path("/root/autodl-tmp")
GENEVAL_PYTHON = WORKSPACE / "venvs/geneval-score/bin/python"
SHOWO_PYTHON = WORKSPACE / "venvs/showo2-eval/bin/python"
VQA_PYTHON = WORKSPACE / "venvs/showo2-score-vqa/bin/python"
POSITION_MANIFEST = WORKSPACE / "data/showo2_layout/position_unseen72_v1.jsonl"
SELF_CYCLE_DATASET = WORKSPACE / "data/showo2_self_cycle/unicycle/self_built.jsonl"
VQA_MODEL = WORKSPACE / "models/c7/Qwen2.5-VL-7B-Instruct"
SAVE_STEPS = (100, 200, 500, 1000, 2000, 4000, 5000, 6000, 8000, 10000, 13000, 17100)
ROUTES = ("g_adapter", "lora")


@dataclass(frozen=True)
class Variant:
    name: str
    label: str
    root: Path
    manifest: Path
    protocol: str
    flow_weight: float


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def parse_steps(value: str) -> tuple[int, ...]:
    steps = tuple(int(item) for item in value.split(",") if item)
    if not steps or tuple(sorted(set(steps))) != steps:
        raise ValueError(f"steps must be non-empty, unique, and sorted: {value}")
    if not set(steps).issubset(SAVE_STEPS):
        raise ValueError(f"benchmark steps are not a subset of {SAVE_STEPS}")
    return steps


def checkpoint_path(variant: Variant, route: str, step: int) -> Path:
    return variant.root / route / "checkpoints" / f"step-{step:06d}.pt"


def eval_variant_name(variant: Variant, route: str, step: int) -> str:
    return f"{variant.name}_{route}_step_{step:06d}"


def eval_root(run_root: Path, variant: Variant, route: str, step: int) -> Path:
    return run_root / "checkpoint_eval" / variant.name / route / f"step-{step:06d}"


def vqa_root(run_root: Path, route: str, step: int) -> Path:
    return run_root / "vqascore" / route / f"step-{step:06d}"


def vsr_root(run_root: Path, variant: Variant, route: str, step: int) -> Path:
    return run_root / "vsr_dev" / variant.name / route / f"step-{step:06d}"


class Queue:
    def __init__(
        self,
        args: argparse.Namespace,
        variants: tuple[Variant, ...],
        steps: tuple[int, ...],
    ) -> None:
        self.args = args
        self.variants = variants
        self.steps = steps
        self.run_root = args.run_root.resolve()
        self.status_path = self.run_root / "queue_status.json"
        self.started_at = load_json(self.status_path).get("started_at", now())
        self.current_task: str | None = None
        self.active_status_paths: list[str] = []
        self.error: str | None = None
        self.planned = self.build_plan()
        self.run_root.mkdir(parents=True, exist_ok=True)
        (self.run_root / "logs").mkdir(exist_ok=True)

    def build_plan(self) -> list[str]:
        tasks = ["preflight"]
        for step in self.steps:
            for variant in self.variants:
                for route in ROUTES:
                    tasks.append(f"eval__{variant.name}__{route}__{step}")
        for step in self.steps:
            for route in ROUTES:
                tasks.append(f"vqa__{route}__{step}")
        tasks.append("vsr_baseline_dev")
        for step in self.steps:
            for variant in self.variants:
                for route in ROUTES:
                    tasks.append(f"vsr__{variant.name}__{route}__{step}")
        tasks.append("aggregate")
        return tasks

    def completed_tasks(self) -> list[str]:
        return [task for task in self.planned if self.task_complete(task)]

    def task_complete(self, task: str) -> bool:
        if task == "preflight":
            return load_json(self.run_root / "preflight.json").get("status") == "completed"
        if task == "aggregate":
            return (
                (self.run_root / "BENCHMARK_QUEUE_COMPLETED").is_file()
                and load_json(self.run_root / "comparison_summary.json").get("status")
                == "completed"
            )
        if task == "vsr_baseline_dev":
            summary = load_json(self.run_root / "vsr_dev/baseline/summary.json")
            return summary.get("status") == "completed" and int(summary.get("total", 0)) == 340
        kind, *parts = task.split("__")
        if kind == "eval":
            variant = self.variant(parts[0])
            route, step = parts[1], int(parts[2])
            run = eval_root(self.run_root, variant, route, step)
            if not (run / "QUEUE_COMPLETED").is_file():
                return False
            name = eval_variant_name(variant, route, step)
            geneval_variant = run / "geneval_position72" / name
            images = sorted((geneval_variant / "geneval").glob("*/samples/*.png"))
            rows = geneval_variant / "results/results.jsonl"
            self_root = run / f"self_cycle/seed-{self.args.eval_seed}/{name}"
            summary = load_json(self_root / "summary.json")
            try:
                result_rows = self.count_jsonl(rows)
            except OSError:
                return False
            adapter = summary.get("adapter", {})
            metrics = summary.get("metrics", {})
            prompts = metrics.get("prompts", {})
            questions = metrics.get("questions", {})
            return (
                len(images) == 288
                and result_rows == 288
                and summary.get("status") == "completed"
                and int(adapter.get("step", -1)) == step
                and adapter.get("route", {}).get("route") == route
                and int(prompts.get("count", 0)) == 96
                and int(questions.get("questions", 0)) == 166
            )
        if kind == "vqa":
            route, step = parts[0], int(parts[1])
            summary = load_json(vqa_root(self.run_root, route, step) / "summary.json")
            return (
                summary.get("status") == "completed"
                and int(summary.get("images_per_variant", 0)) == 288
                and set(summary.get("variants", [])) == {variant.name for variant in self.variants}
            )
        if kind == "vsr":
            variant = self.variant(parts[0])
            route, step = parts[1], int(parts[2])
            summary = load_json(vsr_root(self.run_root, variant, route, step) / "summary.json")
            return (
                summary.get("status") == "completed"
                and int(summary.get("total", 0)) == 340
                and int(summary.get("checkpoint_step", -1)) == step
            )
        return False

    def variant(self, name: str) -> Variant:
        return next(variant for variant in self.variants if variant.name == name)

    def write_status(self, status: str, phase: str, detail: str = "") -> None:
        completed = self.completed_tasks()
        payload = {
            "status": status,
            "phase": phase,
            "detail": detail,
            "started_at": self.started_at,
            "updated_at": now(),
            "total_tasks": len(self.planned),
            "completed_tasks": len(completed),
            "planned_task_ids": self.planned,
            "completed_task_ids": completed,
            "current_task": self.current_task,
            "active_status_paths": self.active_status_paths,
            "training_wait_root": str(self.args.wait_root.resolve()),
            "wait_marker": self.args.wait_marker,
            "wait_label": self.args.wait_label,
            "registry_path": str((self.run_root / "registry.json").resolve()),
            "protocol": {
                "benchmark_steps": list(self.steps),
                "saved_steps": list(SAVE_STEPS),
                "routes": list(ROUTES),
                "geneval": "Position-72, 72 prompts, 4 samples, 12 inference steps",
                "self_cycle": "compact96-stratified-v1, seed 20260810, 50 inference steps",
                "vqascore": "four variants paired per route/checkpoint on 288 Position-72 images",
                "vsr": "dev split, 340 examples; baseline bridge-off once, checkpoint bridge-on",
            },
            "error": self.error,
        }
        atomic_json(self.status_path, payload)

    def wait_for_training(self) -> None:
        wait_root = self.args.wait_root.resolve()
        marker = self.args.wait_marker
        failure_marker = (
            "BENCHMARK_QUEUE_FAILED"
            if marker == "BENCHMARK_QUEUE_COMPLETED"
            else "QUEUE_FAILED"
        )
        while not (wait_root / marker).is_file():
            if (wait_root / failure_marker).is_file():
                upstream_status = load_json(wait_root / "queue_status.json")
                if not (
                    marker == "BENCHMARK_QUEUE_COMPLETED"
                    and upstream_status.get("status") == "running"
                ):
                    raise RuntimeError(f"upstream stage failed: {wait_root}")
            self.current_task = None
            if marker == "QUEUE_COMPLETED":
                self.active_status_paths = [
                    str(wait_root / "g_adapter/status.json"),
                    str(wait_root / "lora/status.json"),
                ]
                detail = load_json(wait_root / "queue_config.json")
                detail_text = (
                    f"new multi-axis Mask-logits+Flow is still running; "
                    f"max_steps={detail.get('max_steps', '?')}"
                )
            else:
                self.active_status_paths = [str(wait_root / "queue_status.json")]
                upstream = load_json(wait_root / "queue_status.json")
                detail_text = (
                    f"{self.args.wait_label} is still running; "
                    f"phase={upstream.get('phase', upstream.get('status', 'unknown'))} "
                    f"completed={upstream.get('completed_tasks', '?')}/{upstream.get('total_tasks', '?')}"
                )
            self.write_status(
                "waiting",
                "waiting_for_training",
                detail_text,
            )
            time.sleep(self.args.wait_seconds)

    def write_registry(self) -> None:
        payload = {
            "status": "configured",
            "created_at": now(),
            "variants": {
                variant.name: {
                    **asdict(variant),
                    "root": str(variant.root.resolve()),
                    "manifest": str(variant.manifest.resolve()),
                }
                for variant in self.variants
            },
            "routes": list(ROUTES),
            "saved_steps": list(SAVE_STEPS),
            "benchmark_steps": list(self.steps),
        }
        existing = load_json(self.run_root / "registry.json")
        if existing and {k: v for k, v in existing.items() if k != "created_at"} != {
            k: v for k, v in payload.items() if k != "created_at"
        }:
            raise RuntimeError("existing registry does not match requested experiment")
        if not existing:
            atomic_json(self.run_root / "registry.json", payload)

    def preflight(self) -> None:
        output: dict[str, Any] = {"status": "running", "variants": {}}
        atomic_json(self.run_root / "preflight.json", output)
        for required in (
            GENEVAL_PYTHON,
            SHOWO_PYTHON,
            VQA_PYTHON,
            POSITION_MANIFEST,
            SELF_CYCLE_DATASET,
            VQA_MODEL / "model.safetensors.index.json",
            ROOT / "Wan2.1_VAE.pth",
        ):
            if not required.exists():
                raise FileNotFoundError(required)
        for variant in self.variants:
            config = load_json(variant.root / "queue_config.json")
            if int(config.get("max_steps", -1)) != 17100:
                raise RuntimeError(f"{variant.name}: max_steps mismatch")
            if tuple(config.get("save_steps", [])) != SAVE_STEPS:
                raise RuntimeError(f"{variant.name}: save schedule mismatch")
            if Path(config.get("manifest", "")).resolve() != variant.manifest.resolve():
                raise RuntimeError(f"{variant.name}: manifest mismatch")
            weights = config.get("weights", {})
            if float(weights.get("flow", -1)) != variant.flow_weight:
                raise RuntimeError(f"{variant.name}: Flow weight mismatch")
            if float(weights.get("u_task", -1)) != 0.0 or float(weights.get("teacher_retention", -1)) != 0.0:
                raise RuntimeError(f"{variant.name}: unexpected U/teacher objective")
            routes: dict[str, Any] = {}
            for route in ROUTES:
                route_root = variant.root / route
                status = load_json(route_root / "status.json")
                if status.get("status") != "completed" or int(status.get("step", -1)) != 17100:
                    raise RuntimeError(f"{variant.name}/{route}: incomplete training")
                if status.get("initialization") != "original_showo2_zero_output_adaptation":
                    raise RuntimeError(f"{variant.name}/{route}: initialization mismatch")
                if status.get("adaptation", {}).get("route") != route:
                    raise RuntimeError(f"{variant.name}/{route}: route mismatch")
                if Path(status.get("manifest", "")).resolve() != variant.manifest.resolve():
                    raise RuntimeError(f"{variant.name}/{route}: trained manifest mismatch")
                if tuple(status.get("checkpoint_schedule", {}).get("save_steps", [])) != SAVE_STEPS:
                    raise RuntimeError(f"{variant.name}/{route}: status save schedule mismatch")
                observed = tuple(
                    sorted(int(path.stem.removeprefix("step-")) for path in (route_root / "checkpoints").glob("step-*.pt"))
                )
                if observed != SAVE_STEPS:
                    raise RuntimeError(f"{variant.name}/{route}: checkpoint files mismatch: {observed}")
                checkpoints: dict[str, Any] = {}
                for step in self.steps:
                    checkpoint = checkpoint_path(variant, route, step)
                    payload = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
                    if int(payload.get("step", -1)) != step:
                        raise RuntimeError(f"{checkpoint}: internal step mismatch")
                    if payload.get("adaptation_config", {}).get("route") != route:
                        raise RuntimeError(f"{checkpoint}: internal route mismatch")
                    if not payload.get("adaptation_state"):
                        raise RuntimeError(f"{checkpoint}: empty adaptation state")
                    checkpoints[str(step)] = {"path": str(checkpoint.resolve()), "bytes": checkpoint.stat().st_size}
                routes[route] = {
                    "trainable_parameters": int(status["adaptation"]["trainable_parameters"]),
                    "checkpoints": checkpoints,
                }
            difference = abs(routes["g_adapter"]["trainable_parameters"] - routes["lora"]["trainable_parameters"])
            if difference > 12:
                raise RuntimeError(f"{variant.name}: parameter mismatch is {difference}")
            output["variants"][variant.name] = {
                "label": variant.label,
                "root": str(variant.root.resolve()),
                "manifest": str(variant.manifest.resolve()),
                "protocol": variant.protocol,
                "flow_weight": variant.flow_weight,
                "routes": routes,
                "route_parameter_difference": difference,
            }
        output["status"] = "completed"
        output["completed_at"] = now()
        atomic_json(self.run_root / "preflight.json", output)

    def run_command(
        self,
        task: str,
        command: list[str],
        log_path: Path,
        active_status_paths: list[Path],
        env: dict[str, str] | None = None,
    ) -> None:
        if self.task_complete(task):
            return
        self.current_task = task
        self.active_status_paths = [str(path.resolve()) for path in active_status_paths]
        log_path.parent.mkdir(parents=True, exist_ok=True)
        child_env = os.environ.copy()
        child_env.update(
            {
                "HF_HOME": str(WORKSPACE / "cache/huggingface"),
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "TOKENIZERS_PARALLELISM": "false",
                "OMP_NUM_THREADS": "4",
                "MKL_NUM_THREADS": "4",
                "CUDA_VISIBLE_DEVICES": "0",
                "PYTHONPATH": f"{ROOT}:{child_env.get('PYTHONPATH', '')}",
            }
        )
        if env:
            child_env.update(env)
        for attempt in range(1, self.args.max_attempts + 1):
            self.write_status("running", "benchmarking", f"attempt={attempt}/{self.args.max_attempts}")
            with log_path.open("a", encoding="utf-8") as log:
                log.write(f"\n[{now()}] task={task} attempt={attempt} command={command!r}\n")
                log.flush()
                result = subprocess.run(command, cwd=ROOT, env=child_env, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode == 0 and self.task_complete(task):
                return
            with (self.run_root / "logs/retry_history.log").open("a", encoding="utf-8") as retry_log:
                retry_log.write(f"{now()} task={task} attempt={attempt} exit={result.returncode}\n")
            if attempt < self.args.max_attempts:
                time.sleep(self.args.retry_seconds)
        raise RuntimeError(f"task failed validation after {self.args.max_attempts} attempts: {task}")

    def compact_indices(self) -> str:
        report = self.run_root / "compact96_stratified_v1.json"
        command = [
            str(GENEVAL_PYTHON),
            str(ROOT / "evaluation/c6_mask2former_alignment/select_compact_self_cycle.py"),
            "--dataset",
            str(SELF_CYCLE_DATASET),
            "--size",
            "96",
            "--seed",
            str(self.args.eval_seed),
            "--report",
            str(report),
            "--print-csv",
        ]
        return subprocess.check_output(command, cwd=ROOT, text=True).strip()

    def run_checkpoint_evals(self, indices: str) -> None:
        runner = ROOT / "evaluation/c6_mask2former_alignment/run_flow_retention_eval_queue.sh"
        for step in self.steps:
            for variant in self.variants:
                for route in ROUTES:
                    task = f"eval__{variant.name}__{route}__{step}"
                    run = eval_root(self.run_root, variant, route, step)
                    name = eval_variant_name(variant, route, step)
                    status_paths = [
                        run / "geneval_position72/status.json",
                        run / f"self_cycle/seed-{self.args.eval_seed}/{name}/status.json",
                    ]
                    self.run_command(
                        task,
                        ["bash", str(runner)],
                        run / "queue-worker.log",
                        status_paths,
                        env={
                            "RUN_ROOT": str(run),
                            "CHECKPOINT": str(checkpoint_path(variant, route, step)),
                            "VARIANT": name,
                            "PROTOCOL_VERSION": f"c6-axis-ab-{variant.name}-{route}-step{step}-v1",
                            "SEEDS": str(self.args.eval_seed),
                            "SELF_CYCLE_PROTOCOL": "self-cycle-compact96-stratified-v1",
                            "SELF_CYCLE_RECORD_INDICES": indices,
                            "SELF_CYCLE_NUM_INFERENCE_STEPS": "50",
                        },
                    )
                    (run / "QUEUE_FAILED").unlink(missing_ok=True)

    def run_vqascore(self) -> None:
        scorer = ROOT / "evaluation/c7_generation_benchmark/score_vqa.py"
        for step in self.steps:
            for route in ROUTES:
                task = f"vqa__{route}__{step}"
                output = vqa_root(self.run_root, route, step)
                command = [str(VQA_PYTHON), "-u", str(scorer)]
                for variant in self.variants:
                    name = eval_variant_name(variant, route, step)
                    source = eval_root(self.run_root, variant, route, step) / "geneval_position72" / name / "geneval"
                    command.extend(["--variant", f"{variant.name}={source}"])
                command.extend(
                    [
                        "--output-dir",
                        str(output),
                        "--model-path",
                        str(VQA_MODEL),
                        "--batch-size",
                        str(self.args.vqa_batch_size),
                        "--attn-implementation",
                        "sdpa",
                    ]
                )
                self.run_command(task, command, output / "run.log", [output / "status.json"])

    def run_vsr(self) -> None:
        evaluator = ROOT / "evaluation/c6_mask2former_alignment/eval_g_adapter_vsr.py"
        baseline = self.run_root / "vsr_dev/baseline"
        self.run_command(
            "vsr_baseline_dev",
            [
                str(GENEVAL_PYTHON),
                "-u",
                str(evaluator),
                "--run-dir",
                str(baseline),
                "--split",
                "dev",
                "--bridge",
                "off",
            ],
            baseline / "run.log",
            [baseline / "status.json"],
        )
        for step in self.steps:
            for variant in self.variants:
                for route in ROUTES:
                    task = f"vsr__{variant.name}__{route}__{step}"
                    output = vsr_root(self.run_root, variant, route, step)
                    self.run_command(
                        task,
                        [
                            str(GENEVAL_PYTHON),
                            "-u",
                            str(evaluator),
                            "--run-dir",
                            str(output),
                            "--checkpoint",
                            str(checkpoint_path(variant, route, step)),
                            "--split",
                            "dev",
                            "--bridge",
                            "on",
                        ],
                        output / "run.log",
                        [output / "status.json"],
                    )

    @staticmethod
    def count_jsonl(path: Path) -> int:
        return sum(bool(line.strip()) for line in path.open(encoding="utf-8"))

    def aggregate(self) -> None:
        from validate_zero_init_direct_cycle import compact_self_cycle, geneval_metrics

        compact = load_json(self.run_root / "compact96_stratified_v1.json")
        baseline_vsr = load_json(self.run_root / "vsr_dev/baseline/summary.json")
        results: dict[str, Any] = {}
        for variant in self.variants:
            results[variant.name] = {}
            for route in ROUTES:
                results[variant.name][route] = {}
                for step in self.steps:
                    name = eval_variant_name(variant, route, step)
                    root = eval_root(self.run_root, variant, route, step)
                    geneval_variant = root / "geneval_position72" / name
                    images = sorted((geneval_variant / "geneval").glob("*/samples/*.png"))
                    rows = geneval_variant / "results/results.jsonl"
                    if len(images) != 288 or self.count_jsonl(rows) != 288:
                        raise RuntimeError(f"{variant.name}/{route}/{step}: incomplete GenEval")
                    self_summary = load_json(root / f"self_cycle/seed-{self.args.eval_seed}/{name}/summary.json")
                    self_protocol = load_json(root / f"self_cycle/seed-{self.args.eval_seed}/{name}/protocol.json")
                    if self_summary.get("status") != "completed":
                        raise RuntimeError(f"{variant.name}/{route}/{step}: incomplete Self-cycle")
                    if self_protocol.get("record_indices") != compact.get("record_indices"):
                        raise RuntimeError(f"{variant.name}/{route}/{step}: compact selection mismatch")
                    if self_protocol.get("generation_prompt_hidden_from_mmu") is not True:
                        raise RuntimeError(f"{variant.name}/{route}/{step}: Self-cycle prompt leakage")
                    vsr = load_json(vsr_root(self.run_root, variant, route, step) / "summary.json")
                    if vsr.get("status") != "completed" or int(vsr.get("total", 0)) != 340:
                        raise RuntimeError(f"{variant.name}/{route}/{step}: incomplete VSR")
                    vqa = load_json(vqa_root(self.run_root, route, step) / "summary.json")
                    variant_vqa = vqa.get("variants", {}).get(variant.name, {})
                    if not variant_vqa:
                        raise RuntimeError(f"{variant.name}/{route}/{step}: missing VQAScore")
                    results[variant.name][route][str(step)] = {
                        "checkpoint": str(checkpoint_path(variant, route, step).resolve()),
                        "geneval": geneval_metrics(rows),
                        "self_cycle": compact_self_cycle(self_summary),
                        "vqascore": variant_vqa,
                        "vsr_dev": vsr,
                    }
        summary = {
            "status": "completed",
            "completed_at": now(),
            "design": "single-axis vs orthogonal-axis Mask-logits, crossed with Flow and adaptation route",
            "registry": load_json(self.run_root / "registry.json"),
            "protocol": load_json(self.status_path).get("protocol", {}),
            "vsr_baseline_dev": baseline_vsr,
            "results": results,
        }
        atomic_json(self.run_root / "comparison_summary.json", summary)
        (self.run_root / "BENCHMARK_QUEUE_COMPLETED").touch()
        (self.run_root / "BENCHMARK_QUEUE_FAILED").unlink(missing_ok=True)

    def run(self) -> None:
        self.write_registry()
        self.wait_for_training()
        self.current_task = "preflight"
        self.active_status_paths = [str(self.run_root / "preflight.json")]
        if not self.task_complete("preflight"):
            self.write_status("running", "preflight", "validating 4 variants x 2 routes x 12 saved checkpoints")
            self.preflight()
        indices = self.compact_indices()
        self.run_checkpoint_evals(indices)
        self.run_vqascore()
        self.run_vsr()
        self.current_task = "aggregate"
        self.active_status_paths = []
        self.write_status("running", "aggregate", "validating and collecting all outputs")
        if not self.task_complete("aggregate"):
            self.aggregate()
        self.current_task = None
        self.active_status_paths = []
        self.write_status("completed", "completed", "all artifacts validated")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--s0-mask-root", type=Path, required=True)
    parser.add_argument("--s0-mask-flow-root", type=Path, required=True)
    parser.add_argument("--s1-mask-root", type=Path, required=True)
    parser.add_argument("--s1-mask-flow-root", type=Path, required=True)
    parser.add_argument("--wait-root", type=Path, required=True)
    parser.add_argument("--wait-marker", default="QUEUE_COMPLETED")
    parser.add_argument("--wait-label", default="upstream training")
    parser.add_argument("--benchmark-steps", default="100,500,1000,5000,17100")
    parser.add_argument("--eval-seed", type=int, default=20260810)
    parser.add_argument("--wait-seconds", type=int, default=30)
    parser.add_argument("--retry-seconds", type=int, default=30)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--vqa-batch-size", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    old_manifest = WORKSPACE / "data/drsa/bprime_coco_geneval_v2/train.jsonl"
    new_manifest = WORKSPACE / "data/drsa/mask_logits_coco_multiaxis_v1/train.jsonl"
    variants = (
        Variant("s0_mask_logits", "old single-axis Mask-logits", args.s0_mask_root, old_manifest, "single-axis four-way historical ablation", 0.0),
        Variant("s0_mask_flow", "old single-axis Mask-logits + Flow", args.s0_mask_flow_root, old_manifest, "single-axis four-way historical ablation", 0.25),
        Variant("s1_mask_logits", "new orthogonal-axis Mask-logits", args.s1_mask_root, new_manifest, "horizontal/vertical axes isolated; dual-axis positives allowed", 0.0),
        Variant("s1_mask_flow", "new orthogonal-axis Mask-logits + Flow", args.s1_mask_flow_root, new_manifest, "horizontal/vertical axes isolated; dual-axis positives allowed", 0.25),
    )
    queue = Queue(args, variants, parse_steps(args.benchmark_steps))
    try:
        queue.run()
        return 0
    except BaseException as error:
        queue.error = f"{type(error).__name__}: {error}"
        queue.write_status("failed", "failed", traceback.format_exc())
        (queue.run_root / "BENCHMARK_QUEUE_FAILED").touch()
        raise


if __name__ == "__main__":
    raise SystemExit(main())
