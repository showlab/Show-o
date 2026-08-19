#!/usr/bin/env python3
"""Run the resumable post-warmup benchmark matrix in a fixed order."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import torch


ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT.parents[1]
STEPS = (100, 200, 500, 1000, 2000, 4000, 5000, 6000, 8000, 10000, 43993)
EXCLUDED_HISTORICAL_STEPS = (20000, 30000)
SELF_CYCLE_SEED = 20260810
COCO_COUNT = 500
VISOR_PROMPTS = 200
VISOR_SAMPLES_PER_PROMPT = 1
VISOR_INFERENCE_STEPS = 12
VISOR_IMAGES = VISOR_PROMPTS * VISOR_SAMPLES_PER_PROMPT
VISOR_PROTOCOL = "showo2-sr2d-visor-200x1-12step-pretest-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--min-free-gib", type=float, default=8.0)
    return parser.parse_args()


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def json_completed(path: Path) -> bool:
    try:
        return read_json(path).get("status") == "completed"
    except (FileNotFoundError, json.JSONDecodeError):
        return False


def line_count(path: Path) -> int:
    if not path.is_file():
        return 0
    return sum(1 for line in path.open() if line.strip())


def variant(step: int) -> str:
    return f"warmup_step_{step}"


def checkpoint_path(args: argparse.Namespace, step: int) -> Path:
    if step == 5000:
        return args.run_root / "replay-step5000/checkpoints/step-005000.pt"
    return args.train_root / "checkpoints" / f"step-{step:06d}.pt"


def completed_checkpoint(path: Path, step: int) -> bool:
    if not path.is_file():
        return False
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        return (
            int(payload.get("step", -1)) == step
            and payload.get("adaptation_config", {}).get("route") == "g_adapter"
        )
    except Exception:
        return False


class Queue:
    def __init__(self, args: argparse.Namespace, task_ids: list[str]):
        self.args = args
        self.task_ids = task_ids
        self.status_path = args.run_root / "queue_status.json"
        self.logs = args.run_root / "logs"
        self.logs.mkdir(parents=True, exist_ok=True)
        self.completed: list[str] = []
        self.started = time.time()
        self.update("initializing", None, [])

    def update(self, state: str, current: str | None,
               active_status_paths: list[Path], error: str | None = None) -> None:
        payload = {
            "status": state,
            "current_task": current,
            "completed_tasks": len(self.completed),
            "total_tasks": len(self.task_ids),
            "completed_task_ids": self.completed,
            "planned_task_ids": self.task_ids,
            "active_status_paths": [str(path) for path in active_status_paths],
            "elapsed_seconds": time.time() - self.started,
            "updated_at": now(),
            "protocol": {
                "steps": list(STEPS),
                "excluded_historical_steps": list(EXCLUDED_HISTORICAL_STEPS),
                "self_cycle_seeds": [SELF_CYCLE_SEED],
                "coco_images_per_checkpoint": COCO_COUNT,
                "coco_scope": "pretest",
                "visor_protocol": VISOR_PROTOCOL,
                "visor_prompts": VISOR_PROMPTS,
                "visor_samples_per_prompt": VISOR_SAMPLES_PER_PROMPT,
                "visor_images_per_variant": VISOR_IMAGES,
                "visor_inference_steps": VISOR_INFERENCE_STEPS,
                "visor_scope": "pretest",
            },
        }
        if error:
            payload["error"] = error
        write_json(self.status_path, payload)

    def run(self, task_id: str, command: list[str], validator: Callable[[], bool],
            *, env: dict[str, str] | None = None,
            active_status_paths: list[Path] | None = None) -> None:
        if validator():
            self.completed.append(task_id)
            self.update("running", None, [])
            return
        paths = active_status_paths or []
        log_path = self.logs / f"{task_id}.log"
        for attempt in range(1, self.args.max_attempts + 1):
            self.update("running", task_id, paths)
            with log_path.open("a", encoding="utf-8") as log:
                log.write(f"\n[{now()}] attempt={attempt} command={command!r}\n")
                log.flush()
                result = subprocess.run(
                    command, cwd=ROOT, env=env, stdout=log,
                    stderr=subprocess.STDOUT, check=False,
                )
            if result.returncode == 0 and validator():
                self.completed.append(task_id)
                self.update("running", None, [])
                return
            with log_path.open("a", encoding="utf-8") as log:
                log.write(
                    f"[{now()}] validation_failed returncode={result.returncode}\n"
                )
            if attempt < self.args.max_attempts:
                time.sleep(15)
        raise RuntimeError(f"task failed validation after retries: {task_id}")


def base_env() -> dict[str, str]:
    env = dict(os.environ)
    env.update({
        "PYTHONPATH": f"{ROOT}:{env.get('PYTHONPATH', '')}",
        "HF_HOME": str(WORKSPACE / "cache/huggingface"),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "OMP_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4",
        "CUDA_VISIBLE_DEVICES": "0",
    })
    return env


def require_free_space(args: argparse.Namespace) -> None:
    free = shutil.disk_usage(WORKSPACE).free
    if free < args.min_free_gib * 1024 ** 3:
        raise RuntimeError(
            f"only {free / 1024 ** 3:.2f} GiB free; require {args.min_free_gib:.2f} GiB"
        )


def prepare_baseline_coco_view(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    expected = {f"{index:05d}.png" for index in range(COCO_COUNT)}
    for name in sorted(expected):
        src, dst = source / name, destination / name
        if not src.is_file():
            raise FileNotFoundError(src)
        if not dst.exists():
            os.link(src, dst)
    extras = {path.name for path in destination.glob("*.png")} - expected
    if extras:
        raise RuntimeError(f"baseline COCO view has extras: {sorted(extras)[:5]}")


def vsr_valid(path: Path, total: int) -> bool:
    try:
        data = read_json(path / "summary.json")
        return data.get("status") == "completed" and int(data.get("total", -1)) == total
    except (FileNotFoundError, json.JSONDecodeError):
        return False


def geneval_valid(path: Path, name: str) -> bool:
    root = path / name
    results = root / "results/results.jsonl"
    images = list((root / "geneval").glob("[0-9][0-9][0-9][0-9][0-9]/samples/*.png"))
    return line_count(results) == 288 and len(images) == 288


def self_cycle_valid(path: Path, step: int) -> bool:
    try:
        data = read_json(path / "summary.json")
        metrics = data["metrics"]
        return (
            data.get("status") == "completed"
            and int(data.get("adapter", {}).get("step", -1)) == step
            and int(metrics["prompts"]["count"]) == 689
            and int(metrics["questions"]["questions"]) == 1087
        )
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        return False


def visor_valid(path: Path) -> bool:
    try:
        summary = read_json(path / "owlvit/summary.json")
        status = read_json(path / "generation/status.json")
        generation_protocol = read_json(path / "generation/protocol.json")
        scorer_protocol = summary.get("protocol", {})
        return (
            summary.get("status") == "completed"
            and status.get("status") == "completed"
            and int(status.get("prompts", -1)) == VISOR_PROMPTS
            and int(status.get("completed_images", -1)) == VISOR_IMAGES
            and int(generation_protocol.get("samples_per_prompt", -1)) == VISOR_SAMPLES_PER_PROMPT
            and int(generation_protocol.get("num_inference_steps", -1)) == VISOR_INFERENCE_STEPS
            and int(scorer_protocol.get("samples_per_prompt", -1)) == VISOR_SAMPLES_PER_PROMPT
        )
    except (FileNotFoundError, json.JSONDecodeError):
        return False


def coco_valid(path: Path) -> bool:
    try:
        data = read_json(path / "generation_summary.json")
        return data.get("status") == "completed" and int(data.get("images", -1)) == COCO_COUNT
    except (FileNotFoundError, json.JSONDecodeError):
        return False


def create_task_ids() -> list[str]:
    tasks = ["prepare_coco500", "replay_step5000", "vsr_baseline_dev", "vsr_baseline_test"]
    tasks += [f"vsr_{split}_step_{step}" for step in STEPS for split in ("dev", "test")]
    tasks += [f"geneval_step_{step}" for step in STEPS]
    tasks += [f"self_cycle_step_{step}" for step in STEPS]
    tasks += ["visor_baseline"] + [f"visor_step_{step}" for step in STEPS]
    tasks += ["vqascore_all"]
    tasks += [f"coco500_step_{step}" for step in STEPS]
    tasks += ["fid_kid_all", "cmmd_all", "aggregate"]
    return tasks


def aggregate(args: argparse.Namespace) -> None:
    output: dict[str, Any] = {
        "status": "completed",
        "protocol": {
            "steps": list(STEPS), "self_cycle_seed": SELF_CYCLE_SEED,
            "excluded_historical_steps": list(EXCLUDED_HISTORICAL_STEPS),
            "coco_images": COCO_COUNT,
            "visor": {
                "protocol": VISOR_PROTOCOL,
                "scope": "pretest",
                "prompts": VISOR_PROMPTS,
                "samples_per_prompt": VISOR_SAMPLES_PER_PROMPT,
                "images_per_variant": VISOR_IMAGES,
                "num_inference_steps": VISOR_INFERENCE_STEPS,
            },
            "warnings": [
                "VISOR 200x1 at 12 steps is a checkpoint-screening protocol and is not comparable to 4-sample/50-step VISOR results.",
                "COCO-500 FID/KID/CMMD are screening metrics and are not comparable to COCO-5K.",
            ],
        },
        "baseline": {}, "checkpoints": {},
    }
    for split in ("dev", "test"):
        output["baseline"][f"vsr_{split}_bridge_off"] = read_json(
            args.run_root / f"vsr/baseline_bridge_off/{split}/summary.json"
        )
    baseline_geneval_path = (
        WORKSPACE
        / "runs/showo2_mask2former_rollback/masklogits-step500-full-20260809T090000Z"
        / "full_position/baseline/results/results.jsonl"
    )
    baseline_rows = [
        json.loads(line) for line in baseline_geneval_path.read_text().splitlines()
        if line.strip()
    ]
    baseline_prompts: dict[str, list[bool]] = defaultdict(list)
    baseline_relations: dict[str, list[bool]] = defaultdict(list)
    for row in baseline_rows:
        baseline_prompts[str(row["prompt"])].append(bool(row["correct"]))
        metadata = json.loads(row["metadata"])
        relation = str(metadata["include"][1]["position"][0])
        baseline_relations[relation].append(bool(row["correct"]))
    output["baseline"]["geneval_position72"] = {
        "source": str(baseline_geneval_path),
        "image_accuracy": sum(bool(row["correct"]) for row in baseline_rows) / len(baseline_rows),
        "prompt_pass": sum(all(values) for values in baseline_prompts.values()) / len(baseline_prompts),
        "by_relation": {
            key: {"correct": sum(values), "total": len(values), "accuracy": sum(values) / len(values)}
            for key, values in sorted(baseline_relations.items())
        },
    }
    baseline_self_cycle_path = (
        WORKSPACE / "runs/showo2_c8/c8-self-cycle-multiseed-4pair-20260810T031600Z"
        / f"seed-{SELF_CYCLE_SEED}/baseline/summary.json"
    )
    output["baseline"]["self_cycle"] = read_json(baseline_self_cycle_path)
    old_native_root = (
        WORKSPACE / "runs/showo2_native_warmup/native-warmup-500-20260807T070000Z"
        / "evaluation"
    )
    output["historical_native_warmup_step400"] = {
        "comparability": "historical_non_identical_protocol",
        "architecture": "legacy dual-route U/G adapters plus relation encoder",
        "vsr_256": read_json(old_native_root / "vsr/native_400/summary.json"),
        "position8": read_json(old_native_root / "position_paired8_v2/summary.json")["variants"]["native_400"],
        "caveat": "Do not compare these VSR-256/Position-8 values directly to current held-out VSR or Position-72 values.",
    }
    for step in STEPS:
        name = variant(step)
        geneval_results = args.run_root / f"step-{step}/geneval/{name}/results/results.jsonl"
        rows = [json.loads(line) for line in geneval_results.read_text().splitlines() if line]
        by_prompt: dict[str, list[bool]] = defaultdict(list)
        by_relation: dict[str, list[bool]] = defaultdict(list)
        for row in rows:
            by_prompt[str(row["prompt"])].append(bool(row["correct"]))
            metadata = json.loads(row["metadata"])
            relation = str(metadata["include"][1]["position"][0])
            by_relation[relation].append(bool(row["correct"]))
        output["checkpoints"][str(step)] = {
            "checkpoint": str(checkpoint_path(args, step)),
            "vsr_dev_bridge_on": read_json(args.run_root / f"step-{step}/vsr/dev/summary.json"),
            "vsr_test_bridge_on": read_json(args.run_root / f"step-{step}/vsr/test/summary.json"),
            "geneval_position72": {
                "image_accuracy": sum(bool(row["correct"]) for row in rows) / len(rows),
                "prompt_pass": sum(all(values) for values in by_prompt.values()) / len(by_prompt),
                "by_relation": {
                    key: {"correct": sum(values), "total": len(values), "accuracy": sum(values) / len(values)}
                    for key, values in sorted(by_relation.items())
                },
            },
            "self_cycle": read_json(args.run_root / f"step-{step}/self_cycle/summary.json"),
            "visor": read_json(args.run_root / f"step-{step}/visor/owlvit/summary.json"),
        }
    output["vqascore"] = read_json(args.run_root / "vqascore/summary.json")
    output["coco500_fid_kid"] = read_json(args.run_root / "quality/fid_kid/summary.json")
    output["coco500_cmmd"] = read_json(args.run_root / "quality/cmmd/summary.json")
    output["visor_baseline"] = read_json(args.run_root / "visor_baseline/owlvit/summary.json")
    write_json(args.run_root / "comparison_summary.json", output)


def main() -> int:
    args = parse_args()
    if args.max_attempts < 1 or args.min_free_gib <= 0:
        raise ValueError("invalid retry or disk-space setting")
    args.run_root.mkdir(parents=True, exist_ok=True)
    train_status = read_json(args.train_root / "status.json")
    if train_status.get("status") != "completed" or int(train_status.get("step", -1)) != 43993:
        raise RuntimeError("warmup training is not complete")
    task_ids = create_task_ids()
    queue = Queue(args, task_ids)
    env = base_env()
    eval_python = str(WORKSPACE / "venvs/showo2-eval/bin/python")
    geneval_python = str(WORKSPACE / "venvs/geneval-score/bin/python")
    vqa_python = str(WORKSPACE / "venvs/showo2-score-vqa/bin/python")
    fid_python = str(WORKSPACE / "venvs/showo2-score-fid/bin/python")
    cmmd_python = str(WORKSPACE / "venvs/showo2-score-cmmd/bin/python")
    try:
        require_free_space(args)
        coco_root = args.run_root / "quality/coco500"
        coco_manifest = coco_root / "manifest.jsonl"
        coco_reference = coco_root / "references"
        queue.run(
            "prepare_coco500",
            [eval_python, str(ROOT / "evaluation/c6_mask2former_alignment/prepare_coco500_pretest.py"),
             "--source-manifest", str(WORKSPACE / "data/showo2_coco_quality/coco5k-v1/manifest.jsonl"),
             "--source-reference-dir", str(WORKSPACE / "data/showo2_coco_quality/coco5k-v1/references"),
             "--output-dir", str(coco_root), "--count", str(COCO_COUNT)],
            lambda: line_count(coco_manifest) == COCO_COUNT and len(list(coco_reference.glob("*.jpg"))) == COCO_COUNT,
        )

        replay_checkpoint = checkpoint_path(args, 5000)
        replay_env = dict(env)
        replay_env.update({
            "RUN_DIR": str(args.run_root / "replay-step5000"),
            "RESUME_CHECKPOINT": str(args.train_root / "checkpoints/step-004000.pt"),
            "STOP_AFTER_STEP": "5000",
        })
        queue.run(
            "replay_step5000",
            [str(ROOT / "evaluation/c6_mask2former_alignment/run_g_adapter_spright_vsr.sh")],
            lambda: completed_checkpoint(replay_checkpoint, 5000),
            env=replay_env,
            active_status_paths=[args.run_root / "replay-step5000/status.json"],
        )
        for step in STEPS:
            if not completed_checkpoint(checkpoint_path(args, step), step):
                raise RuntimeError(f"invalid checkpoint for step {step}")

        for split, total in (("dev", 340), ("test", 1222)):
            run_dir = args.run_root / f"vsr/baseline_bridge_off/{split}"
            queue.run(
                f"vsr_baseline_{split}",
                [eval_python, str(ROOT / "evaluation/c6_mask2former_alignment/eval_g_adapter_vsr.py"),
                 "--run-dir", str(run_dir), "--split", split, "--bridge", "off"],
                lambda path=run_dir, size=total: vsr_valid(path, size),
                env=env, active_status_paths=[run_dir / "status.json"],
            )
        for step in STEPS:
            for split, total in (("dev", 340), ("test", 1222)):
                run_dir = args.run_root / f"step-{step}/vsr/{split}"
                queue.run(
                    f"vsr_{split}_step_{step}",
                    [eval_python, str(ROOT / "evaluation/c6_mask2former_alignment/eval_g_adapter_vsr.py"),
                     "--run-dir", str(run_dir), "--checkpoint", str(checkpoint_path(args, step)),
                     "--split", split, "--bridge", "on"],
                    lambda path=run_dir, size=total: vsr_valid(path, size),
                    env=env, active_status_paths=[run_dir / "status.json"],
                )

        position_manifest = WORKSPACE / "data/showo2_layout/position_unseen72_v1.jsonl"
        for step in STEPS:
            name = variant(step)
            run_dir = args.run_root / f"step-{step}/geneval"
            config_path = args.run_root / f"step-{step}/geneval_variant.json"
            write_json(config_path, {name: {
                "guidance_steps": 0, "inner_steps": 1, "energy_step_size": 1.0,
                "layers": [7], "attention_source": "qwen", "max_grad_rms": 0.0,
                "drsa_checkpoint": str(checkpoint_path(args, step).resolve()),
            }})
            command = [
                eval_python, "-u", str(ROOT / "evaluation/c3_layout_guidance/run_quick8_v2.py"),
                "--run-dir", str(run_dir), "--protocol-version", f"warmup-step-{step}-position72-pretest-v1",
                "--manifest", str(position_manifest), "--config", str(ROOT / "configs/showo2_1.5b_demo_432x432.yaml"),
                "--model", "showlab/show-o2-1.5B", "--vae-path", str(ROOT / "Wan2.1_VAE.pth"),
                "--variant-config-json", str(config_path), "--variants", name,
                "--samples-per-prompt", "4", "--sample-batch-size", "4", "--base-seed", "12026",
                "--guidance-scale", "5.0", "--num-inference-steps", "12",
                "--guided-guidance-steps", "0", "--inside-weight", "1.0",
                "--cross-weight", "0.0", "--relation-weight", "0.0", "--limit-prompts", "72",
                "--geneval-python", geneval_python, "--geneval-root", str(WORKSPACE / "geneval"),
                "--geneval-model-config", str(WORKSPACE / "mmdetection-2.28.2/mmdet/.mim/configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py"),
                "--geneval-model-path", str(WORKSPACE / "models/geneval"),
                "--geneval-clip-path", str(WORKSPACE / "models/geneval/clip/ViT-L-14.state_dict.pt"),
            ]
            queue.run(
                f"geneval_step_{step}", command,
                lambda path=run_dir, variant_name=name: geneval_valid(path, variant_name),
                env=env, active_status_paths=[run_dir / "status.json"],
            )

        self_dataset = WORKSPACE / "data/showo2_self_cycle/unicycle/self_built.jsonl"
        for step in STEPS:
            run_dir = args.run_root / f"step-{step}/self_cycle"
            queue.run(
                f"self_cycle_step_{step}",
                [eval_python, "-u", str(ROOT / "evaluation/c8_self_cycle/run_self_cycle.py"),
                 "--run-dir", str(run_dir), "--dataset", str(self_dataset),
                 "--variant-name", variant(step), "--drsa-checkpoint", str(checkpoint_path(args, step)),
                 "--guidance-scale", "5.0", "--num-inference-steps", "50",
                 "--generation-batch-size", "4", "--samples-per-prompt", "1",
                 "--base-seed", str(SELF_CYCLE_SEED), "--max-open-answer-tokens", "12"],
                lambda path=run_dir, checkpoint_step=step: self_cycle_valid(path, checkpoint_step),
                env=env, active_status_paths=[run_dir / "status.json"],
            )

        visor_manifest = WORKSPACE / "data/visor/sr2d_balanced200_seed20260811.jsonl"
        visor_specs = [("visor_baseline", args.run_root / "visor_baseline", "baseline", None)] + [
            (f"visor_step_{step}", args.run_root / f"step-{step}/visor", variant(step), checkpoint_path(args, step))
            for step in STEPS
        ]
        for task_id, run_dir, name, checkpoint in visor_specs:
            task_env = dict(env)
            task_env.update({
                "RUN_DIR": str(run_dir), "MANIFEST": str(visor_manifest),
                "VARIANT_NAME": name, "SAMPLES_PER_PROMPT": str(VISOR_SAMPLES_PER_PROMPT),
                "BASE_SEED": "20260811", "NUM_INFERENCE_STEPS": str(VISOR_INFERENCE_STEPS),
                "GENERATION_BATCH_SIZE": "4", "PYTHON": eval_python,
                "DRSA_CHECKPOINT": str(checkpoint) if checkpoint else "",
            })
            queue.run(
                task_id, [str(ROOT / "evaluation/c9_visor/run_variant.sh")],
                lambda path=run_dir: visor_valid(path), env=task_env,
                active_status_paths=[run_dir / "generation/status.json", run_dir / "owlvit/status.json"],
            )

        baseline_geneval = WORKSPACE / "runs/showo2_mask2former_rollback/masklogits-step500-full-20260809T090000Z/full_position/baseline/geneval"
        vqa_dir = args.run_root / "vqascore"
        vqa_command = [vqa_python, "-u", str(ROOT / "evaluation/c7_generation_benchmark/score_vqa.py"),
                       "--variant", f"baseline={baseline_geneval}"]
        for step in STEPS:
            vqa_command += ["--variant", f"{variant(step)}={args.run_root / f'step-{step}/geneval/{variant(step)}/geneval'}"]
        vqa_command += ["--output-dir", str(vqa_dir), "--model-path", str(WORKSPACE / "models/c7/Qwen2.5-VL-7B-Instruct"),
                        "--batch-size", "8", "--attn-implementation", "sdpa"]
        queue.run(
            "vqascore_all", vqa_command,
            lambda: json_completed(vqa_dir / "summary.json"),
            env=env, active_status_paths=[vqa_dir / "status.json"],
        )

        baseline_coco = args.run_root / "quality/generation/baseline/images"
        prepare_baseline_coco_view(
            WORKSPACE / "runs/showo2_coco_quality/coco5k-four-model-20260812T080000Z/generation/baseline/images",
            baseline_coco,
        )
        for step in STEPS:
            require_free_space(args)
            run_dir = args.run_root / f"quality/generation/{variant(step)}"
            queue.run(
                f"coco500_step_{step}",
                [eval_python, "-u", str(ROOT / "evaluation/c10_coco_quality/generate_coco.py"),
                 "--run-dir", str(run_dir), "--manifest", str(coco_manifest),
                 "--variant-name", variant(step), "--drsa-checkpoint", str(checkpoint_path(args, step)),
                 "--guidance-scale", "5.0", "--num-inference-steps", "50",
                 "--generation-batch-size", "4", "--limit", str(COCO_COUNT)],
                lambda path=run_dir: coco_valid(path), env=env,
                active_status_paths=[run_dir / "status.json"],
            )

        generation_roots = {"baseline": baseline_coco}
        generation_roots.update({
            variant(step): args.run_root / f"quality/generation/{variant(step)}/images"
            for step in STEPS
        })
        fid_dir = args.run_root / "quality/fid_kid"
        fid_command = [
            fid_python, "-u", str(ROOT / "evaluation/c10_coco_quality/score_fid_kid.py"),
            "--manifest", str(coco_manifest), "--reference-dir", str(coco_reference),
            "--output-dir", str(fid_dir), "--feature-cache-dir", str(args.run_root / "quality/cache/fid"),
            "--inception-path", str(WORKSPACE / "models/c10/cleanfid/inception-2015-12-05.pt"),
            "--batch-size", "64", "--num-workers", "8", "--kid-seed", "20260812",
            "--kid-subsets", "100", "--kid-max-subset-size", "500",
        ]
        for name, root in generation_roots.items():
            fid_command += ["--variant", f"{name}={root}"]
        queue.run(
            "fid_kid_all", fid_command,
            lambda: json_completed(fid_dir / "summary.json"), env=env,
            active_status_paths=[fid_dir / "status.json"],
        )

        cmmd_dir = args.run_root / "quality/cmmd"
        cmmd_command = [
            cmmd_python, "-u", str(ROOT / "evaluation/c10_coco_quality/score_cmmd.py"),
            "--manifest", str(coco_manifest), "--reference-dir", str(coco_reference),
            "--output-dir", str(cmmd_dir), "--embedding-cache-dir", str(args.run_root / "quality/cache/cmmd"),
            "--batch-size", "4", "--google-research-commit", "015539128d9a7dbe14b5f5308a198a15da808949",
            "--scenic-commit", "a179cefc7fd0a48676c15aa45b31c469e0dfc07d",
        ]
        for name, root in generation_roots.items():
            cmmd_command += ["--variant", f"{name}={root}"]
        cmmd_env = dict(env)
        cmmd_env.update({
            "PYTHONPATH": f"{WORKSPACE / 'third_party/google-research-cmmd-src'}:{WORKSPACE / 'third_party/scenic-cmmd-src'}:{env['PYTHONPATH']}",
            "HOME": str(WORKSPACE), "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        })
        queue.run(
            "cmmd_all", cmmd_command,
            lambda: json_completed(cmmd_dir / "summary.json"), env=cmmd_env,
            active_status_paths=[cmmd_dir / "status.json"],
        )

        aggregate_path = args.run_root / "comparison_summary.json"
        if not json_completed(aggregate_path):
            queue.update("running", "aggregate", [])
            aggregate(args)
        queue.completed.append("aggregate")
        queue.update("completed", None, [])
        (args.run_root / "BENCHMARK_QUEUE_COMPLETED").touch()
        return 0
    except BaseException as error:
        queue.update("failed", queue.task_ids[len(queue.completed)] if len(queue.completed) < len(queue.task_ids) else None,
                     [], f"{type(error).__name__}: {error}")
        (args.run_root / "BENCHMARK_QUEUE_FAILED").write_text(traceback.format_exc())
        raise


if __name__ == "__main__":
    raise SystemExit(main())
