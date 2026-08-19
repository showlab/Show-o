#!/usr/bin/env python3
"""Download and validate the pinned UniCycle structured-QA resources."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


UPSTREAM_REPOSITORY = "https://github.com/XinyuJo/Unicycle"
UPSTREAM_COMMIT = "aa1cf81f791e9053e3623eabff0cbe989c614fbf"
FILES = {
    "self_built.jsonl": {
        "git_blob_sha1": "4ec0fd5725a413c14cf05aca709abbd7d4b215e5",
        "expected_rows": 689,
    },
    "tiif_style.jsonl": {
        "git_blob_sha1": "d02669c789f2305a957a8f018440e90a9407740f",
        "expected_rows": 1649,
    },
}


def parse_args() -> argparse.Namespace:
    workspace = Path(__file__).resolve().parents[4]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=workspace / "data/showo2_self_cycle/unicycle",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def git_blob_sha1(content: bytes) -> str:
    header = f"blob {len(content)}\0".encode("ascii")
    return hashlib.sha1(header + content).hexdigest()


def read_validated_rows(content: bytes, name: str) -> list[dict[str, Any]]:
    rows = []
    for line_number, raw_line in enumerate(content.decode("utf-8").splitlines(), 1):
        if not raw_line.strip():
            continue
        try:
            row = json.loads(raw_line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{name}:{line_number}: invalid JSON: {error}") from error
        prompt = row.get("short_description", row.get("short_desription"))
        qa_pairs = row.get("qa_pairs")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"{name}:{line_number}: missing short_description")
        if not isinstance(qa_pairs, list) or not qa_pairs:
            raise ValueError(f"{name}:{line_number}: missing qa_pairs")
        for qa_index, qa in enumerate(qa_pairs):
            if not isinstance(qa, dict):
                raise ValueError(f"{name}:{line_number}: qa {qa_index} is not an object")
            if not str(qa.get("question", "")).strip():
                raise ValueError(f"{name}:{line_number}: qa {qa_index} has no question")
            if qa.get("answer") is None or not str(qa["answer"]).strip():
                raise ValueError(f"{name}:{line_number}: qa {qa_index} has no answer")
        rows.append(row)
    return rows


def download(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "showo2-c8-self-cycle"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(content)
    os.replace(temporary, path)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_files: dict[str, Any] = {}
    for name, expected in FILES.items():
        path = args.output_dir / name
        if path.is_file() and not args.force:
            content = path.read_bytes()
            source = "existing"
        else:
            url = (
                "https://raw.githubusercontent.com/XinyuJo/Unicycle/"
                f"{UPSTREAM_COMMIT}/data/{name}"
            )
            content = download(url)
            atomic_write(path, content)
            source = "downloaded"
        actual_blob = git_blob_sha1(content)
        if actual_blob != expected["git_blob_sha1"]:
            raise RuntimeError(
                f"Git blob mismatch for {name}: {actual_blob} != "
                f"{expected['git_blob_sha1']}"
            )
        rows = read_validated_rows(content, name)
        if len(rows) != expected["expected_rows"]:
            raise RuntimeError(
                f"Row count mismatch for {name}: {len(rows)} != "
                f"{expected['expected_rows']}"
            )
        question_count = sum(len(row["qa_pairs"]) for row in rows)
        type_counts = Counter(str(row.get("type", "unknown")) for row in rows)
        manifest_files[name] = {
            "path": str(path.resolve()),
            "source": source,
            "bytes": len(content),
            "sha256": sha256_bytes(content),
            "git_blob_sha1": actual_blob,
            "rows": len(rows),
            "questions": question_count,
            "type_counts": dict(sorted(type_counts.items())),
        }
        print(
            json.dumps(
                {
                    "resource": name,
                    "source": source,
                    "rows": len(rows),
                    "questions": question_count,
                    "sha256": sha256_bytes(content),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    manifest = {
        "created_at": utc_now(),
        "upstream_repository": UPSTREAM_REPOSITORY,
        "upstream_commit": UPSTREAM_COMMIT,
        "upstream_license": (
            "not_declared_in_repository_metadata_at_the_pinned_commit"
        ),
        "usage": "research_evaluation_only",
        "files": manifest_files,
    }
    manifest_path = args.output_dir / "manifest.json"
    atomic_write(
        manifest_path,
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    print(f"manifest={manifest_path.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
