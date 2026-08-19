#!/usr/bin/env python3
"""Run the four locked TIFA-12 continuation starts sequentially and resumably."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT.parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--lock-file",
        type=Path,
        default=WORKSPACE
        / "data/showo2_self_cycle/tifa12_external_v1/checkpoints.lock.json",
    )
    parser.add_argument(
        "--data-manifest",
        type=Path,
        default=WORKSPACE / "data/showo2_self_cycle/tifa12_external_v1/manifest.json",
    )
    parser.add_argument("--max-steps", type=int, default=34012)
    parser.add_argument(
        "--save-steps",
        default="100,500,1000,2000,4000,6000,8000,10000,17000,25000,34012",
    )
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--python", type=Path, default=WORKSPACE / "venvs/showo2-eval/bin/python")
    return parser.parse_args()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def slug(start: dict[str, Any]) -> str:
    return f"{start['branch'].lower().replace('-', '_')}_{start['adaptation_route']}"


def main() -> int:
    args = parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    queue_status = args.run_dir / "queue_status.json"
    lock = json.loads(args.lock_file.read_text(encoding="utf-8"))
    data = json.loads(args.data_manifest.read_text(encoding="utf-8"))
    if data.get("status") != "ready":
        raise RuntimeError("external TIFA-12 data manifest is not ready")
    train = Path(data["train"])
    shard_root = Path(data["shard_root"])
    if sha256(train) != data["train_sha256"]:
        raise RuntimeError("training manifest SHA256 mismatch")
    starts = list(lock["starts"])
    if len(starts) not in (2, 4):
        raise RuntimeError("training queue requires two or four locked starts")
    for start in starts:
        checkpoint = Path(start["path"])
        if sha256(checkpoint) != start["sha256"]:
            raise RuntimeError(f"checkpoint SHA256 mismatch: {checkpoint}")

    status: dict[str, Any] = {
        "status": "running",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "current": None,
        "data_manifest": str(args.data_manifest.resolve()),
        "lock_file": str(args.lock_file.resolve()),
        "max_steps_per_route": args.max_steps,
        "tasks": [
            {"label": slug(start), "status": "waiting", "start": start}
            for start in starts
        ],
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    atomic_json(queue_status, status)
    try:
        for index, start in enumerate(starts):
            label = slug(start)
            route_dir = args.run_dir / label
            route_dir.mkdir(exist_ok=True)
            route_status_path = route_dir / "status.json"
            if route_status_path.is_file():
                route_status = json.loads(route_status_path.read_text(encoding="utf-8"))
                if (
                    route_status.get("status") == "completed"
                    and int(route_status.get("step", -1)) == args.max_steps
                ):
                    status["tasks"][index]["status"] = "completed"
                    atomic_json(queue_status, status)
                    continue
            resume: Path | None = None
            latest = route_dir / "latest_checkpoint.txt"
            if latest.is_file():
                candidate = Path(latest.read_text().strip())
                if candidate.is_file():
                    resume = candidate
            status["current"] = label
            status["tasks"][index]["status"] = "running"
            status["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            atomic_json(queue_status, status)
            command = [
                str(args.python), "-u",
                str(ROOT / "evaluation/c6_mask2former_alignment/train_tifa12_self_cycle.py"),
                "--run-dir", str(route_dir),
                "--manifest", str(train),
                "--shard-root", str(shard_root),
                "--init-checkpoint", start["path"],
                "--label", label,
                "--max-steps", str(args.max_steps),
                "--save-steps", args.save_steps,
                "--seed", str(args.seed),
                "--learning-rate", str(args.learning_rate),
                "--status-every", "1",
                "--log-every", "1",
                "--allow-unpaired",
                "--generation-start", "pure_noise",
                "--min-valid-questions", "1",
            ]
            if resume is not None:
                command.extend(("--resume-checkpoint", str(resume)))
            environment = dict(os.environ)
            environment["OMP_NUM_THREADS"] = "1"
            with (route_dir / "run.log").open("a", encoding="utf-8") as log:
                process = subprocess.run(
                    command,
                    cwd=ROOT,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            if process.returncode != 0:
                status["tasks"][index]["status"] = "failed"
                raise RuntimeError(f"{label} exited with code {process.returncode}")
            status["tasks"][index]["status"] = "completed"
            atomic_json(queue_status, status)
        status["status"] = "completed"
        status["current"] = None
        status["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        atomic_json(queue_status, status)
        (args.run_dir / "QUEUE_COMPLETED").touch()
        return 0
    except Exception as error:
        status["status"] = "failed"
        status["error"] = f"{type(error).__name__}: {error}"
        status["traceback"] = traceback.format_exc()
        status["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        atomic_json(queue_status, status)
        (args.run_dir / "QUEUE_FAILED").touch()
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
