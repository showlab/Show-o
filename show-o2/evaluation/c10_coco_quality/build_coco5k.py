#!/usr/bin/env python3
"""Build a deterministic paired COCO-5K manifest and real-image cache."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

from common import file_sha256, utc_now, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=5000)
    parser.add_argument("--base-seed", type=int, default=20260812)
    parser.add_argument(
        "--selection-salt", default="showo2-coco-quality-v1-20260812"
    )
    return parser.parse_args()


def numeric_tar_key(path: Path) -> tuple[int, str]:
    return (int(path.stem), path.name) if path.stem.isdigit() else (10**9, path.name)


def selection_hash(salt: str, image_id: str) -> str:
    return hashlib.sha256(f"{salt}\0{image_id}".encode("utf-8")).hexdigest()


def load_json_member(archive: tarfile.TarFile, name: str) -> dict[str, Any]:
    handle = archive.extractfile(name)
    if handle is None:
        raise FileNotFoundError(f"{archive.name}:{name}")
    value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"{archive.name}:{name}: expected a JSON object")
    return value


def discover(archives: list[Path], salt: str) -> list[dict[str, Any]]:
    records = []
    for archive_path in archives:
        with tarfile.open(archive_path, "r") as archive:
            names = {member.name for member in archive.getmembers() if member.isfile()}
            metadata_names = sorted(
                name for name in names if name.endswith(".metadata.json")
            )
            for metadata_name in metadata_names:
                stem = metadata_name[: -len(".metadata.json")]
                caption_name = f"{stem}.json"
                image_name = next(
                    (
                        f"{stem}{suffix}"
                        for suffix in (".jpg", ".jpeg", ".png")
                        if f"{stem}{suffix}" in names
                    ),
                    None,
                )
                if caption_name not in names or image_name is None:
                    raise RuntimeError(
                        f"incomplete sample {stem!r} in {archive_path}"
                    )
                metadata = load_json_member(archive, metadata_name)
                caption = load_json_member(archive, caption_name)
                if str(metadata.get("image_origin", "")).upper() != "COCO":
                    continue
                image_id = str(metadata.get("image_id", "")).strip()
                prompt = str(caption.get("original_caption", "")).strip()
                if not image_id or not prompt:
                    raise ValueError(f"sample {stem!r} has no COCO ID or caption")
                records.append(
                    {
                        "coco_image_id": image_id,
                        "prompt": prompt,
                        "source_archive": str(archive_path.resolve()),
                        "image_member": image_name,
                        "caption_member": caption_name,
                        "metadata_member": metadata_name,
                        "selection_hash": selection_hash(salt, image_id),
                    }
                )
    ids = [record["coco_image_id"] for record in records]
    if len(ids) != len(set(ids)):
        raise RuntimeError("SPRIGHT COCO archives contain duplicate COCO image IDs")
    return records


def extract_selected(
    records: list[dict[str, Any]], reference_dir: Path
) -> list[dict[str, Any]]:
    by_archive: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for index, record in enumerate(records):
        by_archive[str(record["source_archive"])].append((index, record))
    reference_dir.mkdir(parents=True, exist_ok=True)
    completed = 0
    for archive_name in sorted(by_archive, key=lambda value: numeric_tar_key(Path(value))):
        with tarfile.open(archive_name, "r") as archive:
            for index, record in by_archive[archive_name]:
                handle = archive.extractfile(str(record["image_member"]))
                if handle is None:
                    raise FileNotFoundError(
                        f"{archive_name}:{record['image_member']}"
                    )
                payload = handle.read()
                digest = hashlib.sha256(payload).hexdigest()
                with Image.open(io.BytesIO(payload)) as image:
                    image.load()
                    width, height = image.size
                    mode = image.mode
                    image_format = image.format
                if image_format != "JPEG":
                    raise ValueError(
                        f"official COCO reference is not JPEG: {record['image_member']}"
                    )
                destination = reference_dir / f"{index:05d}.jpg"
                if destination.is_file():
                    if file_sha256(destination) != digest:
                        raise RuntimeError(f"existing reference differs: {destination}")
                else:
                    temporary = destination.with_suffix(".jpg.tmp")
                    temporary.write_bytes(payload)
                    os.replace(temporary, destination)
                record.update(
                    {
                        "index": index,
                        "reference_filename": destination.name,
                        "reference_sha256": digest,
                        "reference_bytes": len(payload),
                        "reference_width": width,
                        "reference_height": height,
                        "reference_mode": mode,
                    }
                )
                completed += 1
                if completed % 100 == 0 or completed == len(records):
                    print(
                        json.dumps(
                            {
                                "phase": "extract_references",
                                "completed": completed,
                                "total": len(records),
                                "percent": 100.0 * completed / len(records),
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
    return records


def main() -> None:
    args = parse_args()
    if args.count < 1:
        raise ValueError("count must be positive")
    archives = sorted(args.archive_dir.glob("*.tar"), key=numeric_tar_key)
    if not archives:
        raise FileNotFoundError(f"no tar archives in {args.archive_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    status_path = args.output_dir / "status.json"
    write_json(
        status_path,
        {
            "status": "running",
            "phase": "discover",
            "archives": len(archives),
            "requested_count": args.count,
            "updated_at": utc_now(),
        },
    )
    discovered = discover(archives, args.selection_salt)
    if len(discovered) < args.count:
        raise RuntimeError(f"only found {len(discovered)} COCO samples")
    selected = sorted(
        discovered,
        key=lambda row: (row["selection_hash"], row["coco_image_id"]),
    )[: args.count]
    for index, record in enumerate(selected):
        record["seed"] = args.base_seed + index
    selected = extract_selected(selected, args.output_dir / "references")
    manifest = args.output_dir / "manifest.jsonl"
    write_jsonl(manifest, selected)
    archive_metadata = []
    for archive in archives:
        archive_metadata.append(
            {
                "path": str(archive.resolve()),
                "bytes": archive.stat().st_size,
                "sha256": file_sha256(archive),
            }
        )
    protocol = {
        "protocol_version": "showo2-coco5k-quality-paired-v1",
        "created_at": utc_now(),
        "selection": "ascending SHA-256(selection_salt + NUL + coco_image_id)",
        "selection_salt": args.selection_salt,
        "count": args.count,
        "base_seed": args.base_seed,
        "seed_formula": "base_seed + manifest_index",
        "caption_field": "SPRIGHT-COCO original_caption",
        "reference_source": "COCO 2014 validation images in SPRIGHT-COCO",
        "archives": archive_metadata,
        "manifest": str(manifest.resolve()),
        "manifest_sha256": file_sha256(manifest),
        "reference_dir": str((args.output_dir / "references").resolve()),
    }
    write_json(args.output_dir / "protocol.json", protocol)
    summary = {
        "status": "completed",
        "phase": "completed",
        "samples": len(selected),
        "discovered_samples": len(discovered),
        "manifest_sha256": protocol["manifest_sha256"],
        "reference_bytes": sum(int(row["reference_bytes"]) for row in selected),
        "updated_at": utc_now(),
    }
    write_json(status_path, summary)
    write_json(args.output_dir / "summary.json", {**summary, "protocol": protocol})
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
