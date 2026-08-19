#!/usr/bin/env python3
"""Run three official-TIFA baselines, then MF mixed-replay LoRA and MF benchmarks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
import traceback
from pathlib import Path

ROOT = Path("/root/autodl-tmp/Show-o/show-o2")
WORKSPACE = Path("/root/autodl-tmp")
TIFA_TEST = WORKSPACE / "data/showo2_self_cycle/tifa_v1/official_test_unicycle.jsonl"
REPLAY_MANIFEST = WORKSPACE / "data/drsa/mask_logits_coco_multiaxis_v1/train.jsonl"
REPLAY_SHARD_ROOT = WORKSPACE / "data/drsa/spright_coco/data"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def metric_digest(summary: dict) -> dict:
    metrics = summary.get("metrics", {})
    questions = metrics.get("questions", {})
    prompts = metrics.get("prompts", {})
    modes = metrics.get("by_answer_mode", {})
    yes_no = modes.get("yes_no", {})
    short_answer = modes.get("short_answer", {})
    return {
        "question_accuracy": questions.get("accuracy"),
        "question_mean_token_f1": questions.get("mean_token_f1"),
        "yes_no_accuracy": yes_no.get("accuracy"),
        "short_answer_accuracy": short_answer.get("accuracy"),
        "prompt_hard_score": prompts.get("hard_score"),
        "prompt_soft_score": prompts.get("soft_score"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument(
        "--lock-file",
        type=Path,
        default=WORKSPACE
        / "data/showo2_self_cycle/tifa_v1/official_tifa_lora_starts.lock.json",
    )
    parser.add_argument(
        "--data-manifest",
        type=Path,
        default=WORKSPACE
        / "data/showo2_self_cycle/tifa_v1/official_trainval_interleaved.manifest.json",
    )
    parser.add_argument("--max-steps", type=int, default=3865)
    parser.add_argument(
        "--save-steps",
        default="100,200,300,400,500,750,1000,1250,1500,2000,2500,3865",
    )
    parser.add_argument(
        "--benchmark-steps", default="100,200,500,1000,2000,3865"
    )
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--eval-seed", type=int, default=20260810)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--replay-probability", type=float, default=0.25)
    parser.add_argument("--replay-mask-weight", type=float, default=1.0)
    parser.add_argument("--replay-flow-weight", type=float, default=0.10)
    parser.add_argument("--teacher-consistency-weight", type=float, default=0.25)
    parser.add_argument("--teacher-temperature", type=float, default=1.0)
    parser.add_argument(
        "--python",
        type=Path,
        default=WORKSPACE / "venvs/showo2-eval/bin/python",
    )
    parser.add_argument(
        "--train-python",
        type=Path,
        default=WORKSPACE / "venvs/geneval-score/bin/python",
    )
    args = parser.parse_args()

    root = args.run_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    status_path = root / "queue_status.json"
    payload = {
        "status": "starting",
        "stage": "preflight",
        "current": None,
        "baseline_order": [
            "showo2_baseline",
            "s1_m_lora_start",
            "s1_mf_lora_start",
        ],
        "training_routes": ["s1_mf_lora"],
        "deferred_training_routes": ["s1_m_lora"],
        "benchmark_variants": ["s1_mf"],
        "data_manifest": str(args.data_manifest.resolve()),
        "lock_file": str(args.lock_file.resolve()),
        "tifa_test": str(TIFA_TEST.resolve()),
        "max_steps": args.max_steps,
        "save_steps": [int(x) for x in args.save_steps.split(",") if x],
        "benchmark_steps": [
            int(x) for x in args.benchmark_steps.split(",") if x
        ],
        "objective": {
            "tifa_ce": 1.0,
            "mask_replay": args.replay_mask_weight,
            "flow_replay": args.replay_flow_weight,
            "coco_teacher_consistency": args.teacher_consistency_weight,
            "teacher_temperature": args.teacher_temperature,
            "replay_probability": args.replay_probability,
        },
        "g_adapter": {
            "status": "discarded",
            "reason": "user_requested_lora_only",
        },
        "updated_at": now(),
    }

    def write_status(
        stage: str,
        current: str | None,
        *,
        status: str = "running",
        detail: dict | None = None,
        error: str | None = None,
    ) -> None:
        payload.update(
            {
                "status": status,
                "stage": stage,
                "current": current,
                "current_detail": detail or {},
                "updated_at": now(),
                "error": error,
            }
        )
        atomic(status_path, payload)

    env = os.environ.copy()
    env.update(
        {
            "OMP_NUM_THREADS": "1",
            "PYTHONPATH": str(WORKSPACE / "mmdetection-2.28.2") + ":" + str(ROOT) + ":" + env.get("PYTHONPATH", ""),
            "HF_HOME": str(WORKSPACE / "cache/huggingface"),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "CUDA_VISIBLE_DEVICES": "0",
        }
    )

    def run_monitored(
        cmd: list[str],
        log_path: Path,
        child_status: Path,
        stage: str,
        current: str,
    ) -> None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"[{now()}] command={cmd!r}\n")
            log.flush()
            process = subprocess.Popen(
                cmd,
                cwd=ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            while True:
                return_code = process.poll()
                write_status(
                    stage,
                    current,
                    detail={
                        "pid": process.pid,
                        "child_status_path": str(child_status),
                        **load(child_status),
                    },
                )
                if return_code is not None:
                    break
                time.sleep(10)
        if return_code != 0:
            raise RuntimeError(
                f"{current} failed with exit code {return_code}; see {log_path}"
            )

    try:
        failed_marker = root / "QUEUE_FAILED"
        if failed_marker.exists():
            backup = root / f"QUEUE_FAILED.before-restart-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
            failed_marker.rename(backup)
            payload["previous_failed_marker"] = str(backup)

        required = [
            args.python,
            args.train_python,
            args.lock_file,
            args.data_manifest,
            TIFA_TEST,
            REPLAY_MANIFEST,
            REPLAY_SHARD_ROOT,
            ROOT / "Wan2.1_VAE.pth",
            ROOT / "configs/showo2_1.5b_demo_432x432.yaml",
            ROOT / "evaluation/c8_self_cycle/run_self_cycle.py",
            ROOT
            / "evaluation/c6_mask2former_alignment/train_tifa12_self_cycle.py",
            ROOT
            / "evaluation/c6_mask2former_alignment/run_official_tifa_benchmark_queue.py",
        ]
        for path in required:
            if not Path(path).exists():
                raise FileNotFoundError(path)

        manifest = json.loads(args.data_manifest.read_text(encoding="utf-8"))
        train = Path(manifest["train"])
        shard_root = Path(manifest["shard_root"])
        if sha256(train) != manifest["train_sha256"]:
            raise RuntimeError("TIFA train manifest SHA256 mismatch")

        lock = json.loads(args.lock_file.read_text(encoding="utf-8"))
        starts = {str(item["branch"]): item for item in lock["starts"]}
        for branch in ("S1-M", "S1-MF"):
            item = starts[branch]
            checkpoint = Path(item["path"])
            if sha256(checkpoint) != item["sha256"]:
                raise RuntimeError(
                    f"locked checkpoint SHA256 mismatch: {checkpoint}"
                )

        payload["locked_starts"] = {
            branch: {
                "path": starts[branch]["path"],
                "sha256": starts[branch]["sha256"],
            }
            for branch in ("S1-M", "S1-MF")
        }
        write_status("baseline_tifa_test", "showo2_baseline")

        baseline_specs = [
            ("showo2_baseline", None),
            ("s1_m_lora_start", Path(starts["S1-M"]["path"])),
            ("s1_mf_lora_start", Path(starts["S1-MF"]["path"])),
        ]
        baseline_root = root / "baseline_tifa_test"
        for label, checkpoint in baseline_specs:
            out = baseline_root / label
            summary_path = out / "summary.json"
            if load(summary_path).get("status") == "completed":
                continue
            out.mkdir(parents=True, exist_ok=True)
            cmd = [
                str(args.python),
                "-u",
                str(ROOT / "evaluation/c8_self_cycle/run_self_cycle.py"),
                "--run-dir",
                str(out),
                "--dataset",
                str(TIFA_TEST),
                "--config",
                str(ROOT / "configs/showo2_1.5b_demo_432x432.yaml"),
                "--model",
                "showlab/show-o2-1.5B",
                "--variant-name",
                label,
                "--vae-path",
                str(ROOT / "Wan2.1_VAE.pth"),
                "--guidance-scale",
                "5.0",
                "--num-inference-steps",
                "50",
                "--generation-batch-size",
                "4",
                "--samples-per-prompt",
                "1",
                "--base-seed",
                str(args.eval_seed),
                "--max-open-answer-tokens",
                "12",
            ]
            if checkpoint is not None:
                cmd.extend(["--drsa-checkpoint", str(checkpoint)])
            run_monitored(
                cmd,
                out / "run.log",
                out / "status.json",
                "baseline_tifa_test",
                label,
            )
            if load(summary_path).get("status") != "completed":
                raise RuntimeError(f"{label} missing completed summary")
            (out / "BENCHMARK_COMPLETED").touch()

        baseline_results = {
            label: {
                "run_dir": str((baseline_root / label).resolve()),
                **metric_digest(load(baseline_root / label / "summary.json")),
            }
            for label, _ in baseline_specs
        }
        atomic(
            baseline_root / "comparison_summary.json",
            {
                "status": "completed",
                "protocol": "official-tifa-test-v1",
                "dataset": str(TIFA_TEST.resolve()),
                "seed": args.eval_seed,
                "num_inference_steps": 50,
                "results": baseline_results,
                "completed_at": now(),
            },
        )
        payload["baseline_results"] = baseline_results

        smoke = root / "preflight_smoke"
        if not (smoke / "TRAINING_COMPLETED").is_file():
            smoke.mkdir(parents=True, exist_ok=True)
            mf_start = starts["S1-MF"]
            smoke_cmd = [
                str(args.train_python),
                "-u",
                str(
                    ROOT
                    / "evaluation/c6_mask2former_alignment/train_tifa12_self_cycle.py"
                ),
                "--run-dir",
                str(smoke),
                "--manifest",
                str(train),
                "--shard-root",
                str(shard_root),
                "--init-checkpoint",
                mf_start["path"],
                "--label",
                "s1_mf_mixed_replay_preflight_smoke",
                "--mode",
                "smoke",
                "--max-steps",
                "1",
                "--save-steps",
                "1",
                "--seed",
                str(args.seed),
                "--learning-rate",
                str(args.learning_rate),
                "--status-every",
                "1",
                "--log-every",
                "1",
                "--allow-unpaired",
                "--generation-start",
                "pure_noise",
                "--min-valid-questions",
                "1",
                "--replay-manifest",
                str(REPLAY_MANIFEST),
                "--replay-shard-root",
                str(REPLAY_SHARD_ROOT),
                "--replay-probability",
                "1.0",
                "--replay-mask-weight",
                str(args.replay_mask_weight),
                "--replay-flow-weight",
                str(args.replay_flow_weight),
                "--teacher-consistency-weight",
                str(args.teacher_consistency_weight),
                "--teacher-temperature",
                str(args.teacher_temperature),
            ]
            run_monitored(
                smoke_cmd,
                smoke / "run.log",
                smoke / "status.json",
                "preflight_smoke",
                "s1_mf_mixed_replay_step_1",
            )
        if not (smoke / "TRAINING_COMPLETED").is_file():
            raise RuntimeError("mixed replay preflight smoke failed")
        payload["preflight_smoke"] = "completed"

        route = "s1_mf_lora"
        out = root / route
        out.mkdir(parents=True, exist_ok=True)
        route_status = out / "status.json"
        if not (
            load(route_status).get("status") == "completed"
            and int(load(route_status).get("step", -1)) == args.max_steps
        ):
            mf_start = starts["S1-MF"]
            train_cmd = [
                str(args.train_python),
                "-u",
                str(
                    ROOT
                    / "evaluation/c6_mask2former_alignment/train_tifa12_self_cycle.py"
                ),
                "--run-dir",
                str(out),
                "--manifest",
                str(train),
                "--shard-root",
                str(shard_root),
                "--init-checkpoint",
                mf_start["path"],
                "--label",
                route,
                "--max-steps",
                str(args.max_steps),
                "--save-steps",
                args.save_steps,
                "--seed",
                str(args.seed),
                "--learning-rate",
                str(args.learning_rate),
                "--status-every",
                "1",
                "--log-every",
                "1",
                "--allow-unpaired",
                "--generation-start",
                "pure_noise",
                "--min-valid-questions",
                "1",
                "--replay-manifest",
                str(REPLAY_MANIFEST),
                "--replay-shard-root",
                str(REPLAY_SHARD_ROOT),
                "--replay-probability",
                str(args.replay_probability),
                "--replay-mask-weight",
                str(args.replay_mask_weight),
                "--replay-flow-weight",
                str(args.replay_flow_weight),
                "--teacher-consistency-weight",
                str(args.teacher_consistency_weight),
                "--teacher-temperature",
                str(args.teacher_temperature),
            ]
            run_monitored(
                train_cmd,
                out / "run.log",
                route_status,
                "training",
                route,
            )
        if not (out / "TRAINING_COMPLETED").is_file():
            raise RuntimeError(f"{route} missing TRAINING_COMPLETED")

        benchmark_root = root / "mf_benchmark"
        if not (benchmark_root / "BENCHMARK_QUEUE_COMPLETED").is_file():
            benchmark_cmd = [
                str(args.python),
                "-u",
                str(
                    ROOT
                    / "evaluation/c6_mask2former_alignment/run_official_tifa_benchmark_queue.py"
                ),
                "--training-root",
                str(root),
                "--run-root",
                str(benchmark_root),
                "--route",
                "lora",
                "--variants",
                "s1_mf",
                "--steps",
                args.benchmark_steps,
                "--eval-seed",
                str(args.eval_seed),
            ]
            run_monitored(
                benchmark_cmd,
                benchmark_root / "run.log",
                benchmark_root / "queue_status.json",
                "mf_benchmark",
                "s1_mf_lora",
            )
        if not (benchmark_root / "BENCHMARK_QUEUE_COMPLETED").is_file():
            raise RuntimeError("MF benchmark missing BENCHMARK_QUEUE_COMPLETED")

        write_status("completed", None, status="completed")
        (root / "QUEUE_COMPLETED").touch()
        return 0
    except Exception as error:
        message = f"{type(error).__name__}: {error}"
        write_status(
            "failed",
            payload.get("current"),
            status="failed",
            detail=payload.get("current_detail", {}),
            error=message,
        )
        (root / "QUEUE_FAILED").touch()
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
