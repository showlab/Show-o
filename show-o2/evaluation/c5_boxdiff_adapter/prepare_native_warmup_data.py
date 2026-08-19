#!/usr/bin/env python3
"""Validate local data used by the native-objective adapter warm-up smoke."""

from __future__ import annotations

import argparse
import json
import tarfile
import zipfile
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root", type=Path, default=Path("/root/autodl-tmp/data/drsa")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--spright-limit", type=int, default=2)
    parser.add_argument("--vsr-limit", type=int, default=2)
    return parser.parse_args()


def inspect_tar(path: Path) -> dict[str, Any]:
    with tarfile.open(path, "r") as archive:
        members = archive.getmembers()
        names = [member.name for member in members[:20]]
    return {"path": str(path), "bytes": path.stat().st_size, "members": len(members), "sample_names": names}


def inspect_zip(path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
    return {"path": str(path), "bytes": path.stat().st_size, "members": len(names), "sample_names": names[:20]}


def main() -> int:
    args = parse_args()
    root = args.data_root.expanduser().resolve()
    spright_dir = root / "spright_coco" / "data"
    vsr_annotations = root / "vsr" / "annotations" / "zeroshot" / "train.jsonl"
    vsr_images = root / "vsr" / "vsr_images.zip"
    shards = sorted(spright_dir.glob("*.tar"))
    if not shards:
        raise FileNotFoundError(f"No SPRIGHT tar shards under {spright_dir}")
    if not vsr_annotations.is_file() or not vsr_images.is_file():
        raise FileNotFoundError("VSR annotations or image zip is missing")
    spright_samples = [inspect_tar(path) for path in shards[: max(1, args.spright_limit)]]
    with vsr_annotations.open(encoding="utf-8") as handle:
        vsr_rows = [json.loads(line) for _, line in zip(range(max(1, args.vsr_limit)), handle) if line.strip()]
    if not vsr_rows:
        raise ValueError("VSR annotation file has no readable rows")
    payload = {
        "status": "ok",
        "protocol": "native-objective-warmup-data-v1",
        "official_native_full_data_available": False,
        "note": "Local SPRIGHT original_caption and VSR are native-objective proxy data, not official 66M/100M Show-o2 pretraining data.",
        "spright": {
            "root": str(spright_dir),
            "shard_count": len(shards),
            "total_bytes": sum(path.stat().st_size for path in shards),
            "inspected_shards": spright_samples,
        },
        "vsr": {
            "annotations": str(vsr_annotations),
            "images": str(vsr_images),
            "annotation_bytes": vsr_annotations.stat().st_size,
            "image_bytes": vsr_images.stat().st_size,
            "inspected_rows": vsr_rows,
            "zip": inspect_zip(vsr_images),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps({"status": "ok", "output": str(args.output), "spright_shards": len(shards), "vsr_rows_checked": len(vsr_rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
