#!/usr/bin/env python3
"""Resumable downloader for the DRSA-Adapter training data."""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import dataclasses
import hashlib
import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests


DEFAULT_ENDPOINT = "https://hf-mirror.com"
SPRIGHT_REPO = "SPRIGHT-T2I/spright_coco"
VSR_REPO = "albertvillanova/visual-spatial-reasoning"
VSR_GITHUB_REPO = "cambridgeltl/visual-spatial-reasoning"
VSR_SPLITS = ("train", "dev", "test")


@dataclasses.dataclass(frozen=True)
class DownloadSpec:
    key: str
    url: str
    destination: Path
    expected_size: int
    sha256: str | None = None
    kind: str = "file"

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "url": self.url,
            "destination": str(self.destination),
            "expected_size": self.expected_size,
            "sha256": self.sha256,
            "kind": self.kind,
        }


def _natural_key(value: str) -> list[Any]:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", value)]


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _request_json(session: requests.Session, url: str, **kwargs: Any) -> Any:
    response = session.get(url, timeout=60, **kwargs)
    response.raise_for_status()
    return response.json()


def _repo_tree(session: requests.Session, endpoint: str, repo_id: str) -> list[dict[str, Any]]:
    url = f"{endpoint}/api/datasets/{repo_id}/tree/main"
    return _request_json(
        session,
        url,
        params={"recursive": "true", "expand": "true"},
    )


def _hf_file_url(endpoint: str, repo_id: str, path: str) -> str:
    return f"{endpoint}/datasets/{repo_id}/resolve/main/{quote(path, safe='/')}"


def _tree_spec(
    *,
    endpoint: str,
    repo_id: str,
    entry: dict[str, Any],
    destination: Path,
) -> DownloadSpec:
    path = str(entry["path"])
    lfs = entry.get("lfs") or {}
    sha256 = lfs.get("oid")
    return DownloadSpec(
        key=f"{repo_id}:{path}",
        url=_hf_file_url(endpoint, repo_id, path),
        destination=destination,
        expected_size=int(entry.get("size") or lfs.get("size") or 0),
        sha256=str(sha256) if sha256 else None,
    )


def _github_blob_spec(
    session: requests.Session,
    *,
    repo_id: str,
    repo_path: str,
    destination: Path,
) -> DownloadSpec:
    headers = {"Accept": "application/vnd.github+json"}
    contents_url = f"https://api.github.com/repos/{repo_id}/contents/{repo_path}"
    metadata = _request_json(session, contents_url, headers=headers)
    return DownloadSpec(
        key=f"github:{repo_id}:{repo_path}",
        url=str(metadata["git_url"]),
        destination=destination,
        expected_size=int(metadata["size"]),
        kind="github_blob",
    )


def build_manifest(
    *,
    session: requests.Session,
    endpoint: str,
    data_root: Path,
    scope: str,
) -> list[DownloadSpec]:
    spright_tree = _repo_tree(session, endpoint, SPRIGHT_REPO)
    spright_entries = {
        str(entry["path"]): entry
        for entry in spright_tree
        if entry.get("type") == "file"
    }
    shard_paths = sorted(
        (
            path
            for path in spright_entries
            if path.startswith("data/") and path.endswith(".tar")
        ),
        key=_natural_key,
    )
    if not shard_paths:
        raise RuntimeError("SPRIGHT-COCO repository has no data/*.tar shards")
    if scope == "smoke":
        shard_paths = [min(shard_paths, key=lambda path: int(spright_entries[path]["size"]))]

    specs: list[DownloadSpec] = []
    for path in shard_paths:
        specs.append(
            _tree_spec(
                endpoint=endpoint,
                repo_id=SPRIGHT_REPO,
                entry=spright_entries[path],
                destination=data_root / "spright_coco" / path,
            )
        )
    for path in ("README.md", "LICENSE"):
        entry = spright_entries.get(path)
        if entry is not None:
            specs.append(
                _tree_spec(
                    endpoint=endpoint,
                    repo_id=SPRIGHT_REPO,
                    entry=entry,
                    destination=data_root / "spright_coco" / path,
                )
            )

    vsr_tree = _repo_tree(session, endpoint, VSR_REPO)
    vsr_entries = {
        str(entry["path"]): entry
        for entry in vsr_tree
        if entry.get("type") == "file"
    }
    image_entry = vsr_entries.get("data/vsr_images.zip")
    if image_entry is None:
        raise RuntimeError("VSR repository is missing data/vsr_images.zip")
    specs.append(
        _tree_spec(
            endpoint=endpoint,
            repo_id=VSR_REPO,
            entry=image_entry,
            destination=data_root / "vsr" / "vsr_images.zip",
        )
    )
    for split in VSR_SPLITS:
        repo_path = f"data/splits/zeroshot/{split}.jsonl"
        specs.append(
            _github_blob_spec(
                session,
                repo_id=VSR_GITHUB_REPO,
                repo_path=repo_path,
                destination=data_root / "vsr" / "annotations" / "zeroshot" / f"{split}.jsonl",
            )
        )
    return specs


class StatusTracker:
    def __init__(self, path: Path, specs: list[DownloadSpec], scope: str, data_root: Path):
        self.path = path
        self.lock = threading.Lock()
        self.payload: dict[str, Any] = {
            "status": "running",
            "scope": scope,
            "data_root": str(data_root),
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "files": {
                spec.key: {
                    "destination": str(spec.destination),
                    "expected_size": spec.expected_size,
                    "downloaded": 0,
                    "status": "pending",
                }
                for spec in specs
            },
        }
        self.flush()

    def update(self, key: str, **values: Any) -> None:
        with self.lock:
            self.payload["files"][key].update(values)
            self.payload["updated_at"] = time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
            )
            _atomic_json(self.path, self.payload)

    def finish(self, status: str, error: str | None = None) -> None:
        with self.lock:
            self.payload["status"] = status
            self.payload["updated_at"] = time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
            )
            if error:
                self.payload["error"] = error
            _atomic_json(self.path, self.payload)

    def flush(self) -> None:
        with self.lock:
            _atomic_json(self.path, self.payload)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_github_blob(spec: DownloadSpec, tracker: StatusTracker) -> None:
    if spec.destination.is_file() and spec.destination.stat().st_size == spec.expected_size:
        tracker.update(spec.key, downloaded=spec.expected_size, status="complete")
        return
    tracker.update(spec.key, status="downloading")
    response = requests.get(
        spec.url,
        headers={"Accept": "application/vnd.github+json"},
        timeout=120,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("encoding") != "base64":
        raise RuntimeError(f"Unexpected GitHub blob encoding for {spec.key}")
    content = base64.b64decode(payload["content"])
    if len(content) != spec.expected_size:
        raise RuntimeError(
            f"GitHub blob size mismatch for {spec.key}: "
            f"expected {spec.expected_size}, got {len(content)}"
        )
    spec.destination.parent.mkdir(parents=True, exist_ok=True)
    part = spec.destination.with_suffix(spec.destination.suffix + ".part")
    part.write_bytes(content)
    os.replace(part, spec.destination)
    tracker.update(spec.key, downloaded=len(content), status="complete")


def _download_file(
    spec: DownloadSpec,
    tracker: StatusTracker,
    *,
    verify_sha256: bool,
    max_attempts: int,
) -> None:
    spec.destination.parent.mkdir(parents=True, exist_ok=True)
    part = spec.destination.with_suffix(spec.destination.suffix + ".part")
    if spec.destination.is_file():
        current = spec.destination.stat().st_size
        if current == spec.expected_size:
            if verify_sha256 and spec.sha256 and _sha256(spec.destination) != spec.sha256:
                raise RuntimeError(f"SHA256 mismatch for existing file {spec.destination}")
            tracker.update(spec.key, downloaded=current, status="complete")
            return
        raise RuntimeError(
            f"Existing final file has the wrong size: {spec.destination} "
            f"({current} != {spec.expected_size})"
        )

    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            offset = part.stat().st_size if part.exists() else 0
            if offset > spec.expected_size:
                part.unlink()
                offset = 0
            headers = {"Range": f"bytes={offset}-"} if offset else {}
            tracker.update(
                spec.key,
                downloaded=offset,
                status="downloading",
                attempt=attempt,
            )
            with requests.get(
                spec.url,
                headers=headers,
                stream=True,
                timeout=(30, 120),
                allow_redirects=True,
            ) as response:
                if response.status_code == 416 and offset == spec.expected_size:
                    os.replace(part, spec.destination)
                    tracker.update(spec.key, downloaded=offset, status="complete")
                    return
                response.raise_for_status()
                append = offset > 0 and response.status_code == 206
                if offset > 0 and not append:
                    offset = 0
                mode = "ab" if append else "wb"
                downloaded = offset
                last_report = downloaded
                with part.open(mode) as handle:
                    for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                        if not chunk:
                            continue
                        handle.write(chunk)
                        downloaded += len(chunk)
                        if downloaded - last_report >= 32 * 1024 * 1024:
                            tracker.update(
                                spec.key,
                                downloaded=downloaded,
                                status="downloading",
                                attempt=attempt,
                            )
                            last_report = downloaded
            actual = part.stat().st_size
            if actual != spec.expected_size:
                raise RuntimeError(
                    f"Incomplete file {spec.destination.name}: "
                    f"expected {spec.expected_size}, got {actual}"
                )
            if verify_sha256 and spec.sha256 and _sha256(part) != spec.sha256:
                raise RuntimeError(f"SHA256 mismatch for {spec.destination.name}")
            os.replace(part, spec.destination)
            tracker.update(spec.key, downloaded=actual, status="complete")
            return
        except Exception as error:  # requests exposes several retryable subclasses.
            last_error = error
            tracker.update(
                spec.key,
                downloaded=part.stat().st_size if part.exists() else 0,
                status="retrying",
                attempt=attempt,
                error=f"{type(error).__name__}: {error}",
            )
            if attempt < max_attempts:
                time.sleep(min(30, 2 * attempt))
    raise RuntimeError(f"Download failed for {spec.key}: {last_error}")


def download_one(
    spec: DownloadSpec,
    tracker: StatusTracker,
    *,
    verify_sha256: bool,
    max_attempts: int,
) -> None:
    if spec.kind == "github_blob":
        _download_github_blob(spec, tracker)
    else:
        _download_file(
            spec,
            tracker,
            verify_sha256=verify_sha256,
            max_attempts=max_attempts,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--scope", choices=("smoke", "train"), default="train")
    parser.add_argument("--endpoint", default=os.environ.get("HF_ENDPOINT", DEFAULT_ENDPOINT))
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-attempts", type=int, default=50)
    parser.add_argument("--verify-sha256", action="store_true")
    parser.add_argument("--manifest-path", type=Path)
    parser.add_argument("--status-path", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    endpoint = args.endpoint.rstrip("/")
    data_root = args.data_root.resolve()
    manifest_path = args.manifest_path or data_root / "download_manifest.json"
    status_path = args.status_path or data_root / "download_status.json"
    with requests.Session() as session:
        specs = build_manifest(
            session=session,
            endpoint=endpoint,
            data_root=data_root,
            scope=args.scope,
        )
    _atomic_json(
        manifest_path,
        {
            "scope": args.scope,
            "endpoint": endpoint,
            "data_root": str(data_root),
            "total_files": len(specs),
            "total_bytes": sum(spec.expected_size for spec in specs),
            "files": [spec.as_dict() for spec in specs],
        },
    )
    tracker = StatusTracker(status_path, specs, args.scope, data_root)
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    download_one,
                    spec,
                    tracker,
                    verify_sha256=args.verify_sha256,
                    max_attempts=args.max_attempts,
                ): spec
                for spec in specs
            }
            for future in concurrent.futures.as_completed(futures):
                spec = futures[future]
                future.result()
                print(f"complete {spec.key}", flush=True)
    except Exception as error:
        tracker.finish("failed", f"{type(error).__name__}: {error}")
        raise
    tracker.finish("completed")
    print(
        json.dumps(
            {
                "status": "completed",
                "scope": args.scope,
                "data_root": str(data_root),
                "files": len(specs),
                "bytes": sum(spec.expected_size for spec in specs),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
