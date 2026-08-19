#!/usr/bin/env python3
"""Download C7 scorer resources concurrently with resumable status tracking."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from huggingface_hub import snapshot_download


QWEN_REPO = "Qwen/Qwen2.5-VL-7B-Instruct"
QWEN_REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"
REWARD_REPO = "THUDM/ImageReward"
REWARD_REVISION = "5736be03b2652728fb87788c9797b0570450ab72"
BERT_REPO = "google-bert/bert-base-uncased"
BERT_REVISION = "86b5e0934494bd15c9632b12f734a8a67f723594"
COCO_URL = "http://images.cocodataset.org/zips/val2014.zip"
COCO_BYTES = 6_645_013_297


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, default=Path("/root/autodl-tmp"))
    parser.add_argument("--max-workers", type=int, default=4)
    return parser.parse_args()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def tree_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    if not path.exists():
        return 0
    total = 0
    for item in path.rglob("*"):
        if item.is_file() and not item.is_symlink():
            try:
                total += item.stat().st_size
            except FileNotFoundError:
                pass
    return total


class Tracker:
    def __init__(self, path: Path, workspace: Path, resources: dict[str, dict[str, Any]]):
        self.path = path
        self.workspace = workspace
        self.resources = resources
        self.lock = threading.Lock()
        self.started_at = now()
        self.stop_event = threading.Event()

    def update(self, name: str, **values: Any) -> None:
        with self.lock:
            self.resources[name].update(values)
            self.write_locked()

    def write_locked(self) -> None:
        usage = shutil.disk_usage(self.workspace)
        completed = sum(item.get("status") == "completed" for item in self.resources.values())
        failed = sum(item.get("status") == "failed" for item in self.resources.values())
        if failed:
            status = "failed"
        elif completed == len(self.resources):
            status = "completed"
        else:
            status = "running"
        atomic_json(
            self.path,
            {
                "status": status,
                "started_at": self.started_at,
                "updated_at": now(),
                "completed_resources": completed,
                "total_resources": len(self.resources),
                "disk": {
                    "total_bytes": usage.total,
                    "used_bytes": usage.used,
                    "free_bytes": usage.free,
                },
                "resources": self.resources,
            },
        )

    def refresh_loop(self) -> None:
        while not self.stop_event.wait(5):
            with self.lock:
                for item in self.resources.values():
                    paths = [Path(path) for path in item["paths"]]
                    item["downloaded_bytes"] = sum(tree_bytes(path) for path in paths)
                    expected = int(item.get("expected_bytes", 0))
                    if item.get("status") == "completed":
                        item["percent"] = 100.0
                    elif expected:
                        item["percent"] = min(
                            100.0, 100.0 * item["downloaded_bytes"] / expected
                        )
                self.write_locked()


def hf_download(
    repo_id: str,
    revision: str,
    local_dir: Path,
    allow_patterns: list[str] | None = None,
) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
        max_workers=4,
        allow_patterns=allow_patterns,
    )


def download_qwen(path: Path) -> None:
    hf_download(QWEN_REPO, QWEN_REVISION, path)


def download_reward(reward_path: Path, bert_path: Path) -> None:
    hf_download(
        REWARD_REPO,
        REWARD_REVISION,
        reward_path,
        ["ImageReward.pt", "med_config.json", "README.md"],
    )
    hf_download(
        BERT_REPO,
        BERT_REVISION,
        bert_path,
        ["config.json", "tokenizer.json", "tokenizer_config.json", "vocab.txt"],
    )


def validate_coco_reference(path: Path, run_dir: Path) -> None:
    tar_paths = sorted((path / "data").glob("*.tar"), key=lambda item: int(item.stem))
    if len(tar_paths) != 11:
        raise RuntimeError(f"expected 11 SPRIGHT-COCO shards, found {len(tar_paths)}")
    samples = []
    for tar_path in tar_paths:
        with tarfile.open(tar_path, "r") as archive:
            for member in archive:
                if member.isfile() and member.name.lower().endswith(".jpg"):
                    sample_id = f"{tar_path.name}:{Path(member.name).stem}"
                    samples.append({
                        "sample_id": sample_id,
                        "tar_path": str(tar_path.resolve()),
                        "member": member.name,
                    })
    if len(samples) < 40_000:
        raise RuntimeError(f"expected at least 40,000 COCO JPEGs, found {len(samples)}")
    samples.sort(key=lambda item: hashlib.sha256(item["sample_id"].encode()).hexdigest())
    selected = samples[:5_000]
    manifest = run_dir / "coco_val5000_manifest.jsonl"
    temporary = manifest.with_suffix(manifest.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in selected:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    os.replace(temporary, manifest)
    atomic_json(run_dir / "coco_reference.json", {
        "source": "SPRIGHT-T2I/spright_coco",
        "source_description": "original COCO Validation JPEGs in WebDataset tar shards",
        "root": str(path.resolve()),
        "shards": len(tar_paths),
        "jpeg_images": len(samples),
        "fid_subset_images": len(selected),
        "selection": "lowest sha256(sample_id), deterministic over all JPEG members",
        "manifest": str(manifest.resolve()),
    })

def prepare_envs(script: Path, workspace: Path) -> None:
    environment = os.environ.copy()
    environment["WORKSPACE"] = str(workspace)
    subprocess.run(["bash", str(script)], check=True, env=environment)


def run_resource(tracker: Tracker, name: str, action: Callable[[], None]) -> None:
    tracker.update(name, status="running", started_at=now(), error=None, attempt=0)
    attempt = 0
    while True:
        attempt += 1
        tracker.update(name, status="running", attempt=attempt, error=None)
        try:
            action()
            tracker.update(
                name, status="completed", completed_at=now(), percent=100.0
            )
            return
        except BaseException as error:
            delay = min(60, 5 * attempt)
            tracker.update(
                name,
                status="retrying",
                error=f"{type(error).__name__}: {error}",
                traceback=traceback.format_exc(),
                retry_delay_seconds=delay,
            )
            time.sleep(delay)


def main() -> int:
    args = parse_args()
    if args.max_workers < 1:
        raise ValueError("max-workers must be positive")
    workspace = args.workspace.resolve()
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    model_root = workspace / "models" / "c7"
    qwen_path = model_root / "Qwen2.5-VL-7B-Instruct"
    reward_path = model_root / "ImageReward-v1.0"
    bert_path = model_root / "bert-base-uncased-tokenizer"
    coco_path = workspace / "data" / "drsa" / "spright_coco"
    env_script = (
        workspace
        / "Show-o"
        / "show-o2"
        / "evaluation"
        / "c7_generation_benchmark"
        / "prepare_c7_score_envs.sh"
    )
    resources = {
        "qwen2_5_vl_7b": {
            "status": "pending",
            "expected_bytes": 16_595_981_281,
            "downloaded_bytes": tree_bytes(qwen_path),
            "paths": [str(qwen_path)],
            "repo": QWEN_REPO,
            "revision": QWEN_REVISION,
        },
        "image_reward": {
            "status": "pending",
            "expected_bytes": 1_787_580_545,
            "downloaded_bytes": tree_bytes(reward_path) + tree_bytes(bert_path),
            "paths": [str(reward_path), str(bert_path)],
            "repo": REWARD_REPO,
            "revision": REWARD_REVISION,
        },
        "coco_val2014": {
            "status": "pending",
            "expected_bytes": tree_bytes(coco_path),
            "downloaded_bytes": tree_bytes(coco_path),
            "paths": [str(coco_path)],
            "source": "local SPRIGHT-COCO original COCO Validation JPEG shards",
            "download_required": False,
        },
        "score_environments": {
            "status": "pending",
            "expected_bytes": 1_500_000_000,
            "downloaded_bytes": sum(
                tree_bytes(workspace / "venvs" / name)
                for name in (
                    "showo2-score-vqa",
                    "showo2-score-reward",
                    "showo2-score-fid",
                )
            ),
            "paths": [
                str(workspace / "venvs" / "showo2-score-vqa"),
                str(workspace / "venvs" / "showo2-score-reward"),
                str(workspace / "venvs" / "showo2-score-fid"),
            ],
            "shared_torch": "/root/miniconda3/lib/python3.12/site-packages/torch",
        },
    }
    tracker = Tracker(run_dir / "status.json", workspace, resources)
    with tracker.lock:
        tracker.write_locked()
    refresh_thread = threading.Thread(target=tracker.refresh_loop, daemon=True)
    refresh_thread.start()
    actions: dict[str, Callable[[], None]] = {
        "qwen2_5_vl_7b": lambda: download_qwen(qwen_path),
        "image_reward": lambda: download_reward(reward_path, bert_path),
        "coco_val2014": lambda: validate_coco_reference(coco_path, run_dir),
        "score_environments": lambda: prepare_envs(env_script, workspace),
    }
    errors = []
    try:
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {
                executor.submit(run_resource, tracker, name, action): name
                for name, action in actions.items()
            }
            for future in as_completed(futures):
                try:
                    future.result()
                except BaseException as error:
                    errors.append((futures[future], error))
    finally:
        tracker.stop_event.set()
        refresh_thread.join(timeout=10)
        with tracker.lock:
            for item in tracker.resources.values():
                paths = [Path(path) for path in item["paths"]]
                item["downloaded_bytes"] = sum(tree_bytes(path) for path in paths)
            tracker.write_locked()
    if errors:
        raise RuntimeError(
            "; ".join(f"{name}: {type(error).__name__}: {error}" for name, error in errors)
        )
    (run_dir / "DOWNLOAD_COMPLETED").touch()
    print(json.dumps({"status": "completed", "run_dir": str(run_dir)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

