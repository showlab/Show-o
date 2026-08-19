#!/usr/bin/env python3
"""Shared protocol and filesystem helpers for COCO quality evaluation."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def write_json(path: Path, payload: Any) -> None:
    atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    atomic_write_text(
        path,
        "".join(
            json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n"
            for row in rows
        ),
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{line_number}: invalid JSON") from error
        if not isinstance(value, dict):
            raise TypeError(f"{path}:{line_number}: expected a JSON object")
        rows.append(value)
    return rows


def read_coco_manifest(path: Path, expected_count: int | None = None) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    if expected_count is not None and len(rows) != expected_count:
        raise ValueError(f"expected {expected_count} manifest rows, found {len(rows)}")
    required = {
        "index",
        "coco_image_id",
        "prompt",
        "seed",
        "reference_filename",
        "reference_sha256",
    }
    for index, row in enumerate(rows):
        if not required.issubset(row):
            missing = sorted(required - set(row))
            raise ValueError(f"manifest row {index} is missing {missing}")
        if int(row["index"]) != index:
            raise ValueError(f"manifest index mismatch at row {index}")
        if not str(row["prompt"]).strip():
            raise ValueError(f"manifest row {index} has an empty prompt")
        expected_name = f"{index:05d}.jpg"
        if str(row["reference_filename"]) != expected_name:
            raise ValueError(
                f"manifest row {index} reference must be {expected_name}"
            )
    ids = [str(row["coco_image_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("COCO manifest contains duplicate image IDs")
    return rows


def ensure_protocol(path: Path, payload: dict[str, Any]) -> bool:
    normalized = json.loads(json.dumps(payload, ensure_ascii=True, sort_keys=True))
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != normalized:
            raise RuntimeError(f"refusing to resume {path.parent}: protocol differs")
        return True
    write_json(path, normalized)
    return False


def generated_image_path(root: Path, index: int) -> Path:
    return root / f"{index:05d}.png"


def verify_image(path: Path, expected_sha256: str | None = None) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with Image.open(path) as image:
        image.load()
        width, height = image.size
        mode = image.mode
    digest = file_sha256(path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise RuntimeError(f"SHA-256 mismatch for {path}")
    return {
        "sha256": digest,
        "bytes": path.stat().st_size,
        "width": width,
        "height": height,
        "mode": mode,
    }


def directory_fingerprint(root: Path, filenames: list[str]) -> tuple[str, int]:
    digest = hashlib.sha256()
    total_bytes = 0
    for filename in filenames:
        path = root / filename
        info = verify_image(path)
        total_bytes += int(info["bytes"])
        digest.update(filename.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(info["sha256"]).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest(), total_bytes


def validate_flat_image_dir(
    root: Path,
    expected_filenames: list[str],
    *,
    reject_extras: bool = True,
) -> None:
    if not root.is_dir():
        raise FileNotFoundError(root)
    expected = set(expected_filenames)
    actual = {
        path.name
        for path in root.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    }
    missing = sorted(expected - actual)
    extras = sorted(actual - expected)
    if missing:
        raise RuntimeError(f"{root} is missing {len(missing)} images: {missing[:5]}")
    if reject_extras and extras:
        raise RuntimeError(f"{root} has {len(extras)} unexpected images: {extras[:5]}")

