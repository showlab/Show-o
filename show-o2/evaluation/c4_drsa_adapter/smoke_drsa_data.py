#!/usr/bin/env python3
"""Validate SPRIGHT-COCO and VSR without extracting their image archives."""

from __future__ import annotations

import argparse
import collections
import io
import json
import re
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import Any, Iterator

from PIL import Image

SHOWO2_ROOT = Path(__file__).resolve().parents[2]
if str(SHOWO2_ROOT) not in sys.path:
    sys.path.insert(0, str(SHOWO2_ROOT))

RELATION_PHRASES = (
    "at the left side of",
    "at the right side of",
    "in front of",
    "far away from",
    "next to",
    "left of",
    "right of",
    "on top of",
    "above",
    "below",
    "behind",
    "under",
    "inside",
    "outside",
    "near",
    "far from",
    "adjacent",
    "larger than",
    "smaller than",
)


def _natural_key(path: Path) -> list[Any]:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", path.name)]


def _member_key(name: str) -> tuple[str, str] | None:
    basename = Path(name).name
    if basename.endswith(".metadata.json"):
        return basename[: -len(".metadata.json")], "metadata"
    if basename.endswith(".json"):
        return basename[: -len(".json")], "caption"
    if basename.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
        return basename.rsplit(".", 1)[0], "image"
    return None


def iter_spright_samples(tar_path: Path, limit: int) -> Iterator[dict[str, Any]]:
    pending: dict[str, dict[str, bytes]] = collections.defaultdict(dict)
    with tarfile.open(tar_path, mode="r|*") as archive:
        for member in archive:
            if not member.isfile():
                continue
            parsed = _member_key(member.name)
            if parsed is None:
                continue
            stem, kind = parsed
            extracted = archive.extractfile(member)
            if extracted is None:
                continue
            pending[stem][kind] = extracted.read()
            if {"image", "caption", "metadata"}.issubset(pending[stem]):
                raw = pending.pop(stem)
                yield {"id": stem, **raw}
                limit -= 1
                if limit == 0:
                    return


def validate_spright(data_root: Path, limit: int) -> dict[str, Any]:
    shards = sorted((data_root / "spright_coco" / "data").glob("*.tar"), key=_natural_key)
    if not shards:
        raise FileNotFoundError("No completed SPRIGHT-COCO tar shard found")
    relation_counts: collections.Counter[str] = collections.Counter()
    samples: list[dict[str, Any]] = []
    for shard in shards:
        for raw in iter_spright_samples(shard, limit - len(samples)):
            caption = json.loads(raw["caption"])
            metadata = json.loads(raw["metadata"])
            spatial_caption = caption.get("spatial_caption")
            original_caption = caption.get("original_caption")
            if not isinstance(spatial_caption, str) or not spatial_caption.strip():
                raise ValueError(f"Missing spatial_caption in sample {raw['id']}")
            if not isinstance(original_caption, str) or not original_caption.strip():
                raise ValueError(f"Missing original_caption in sample {raw['id']}")
            with Image.open(io.BytesIO(raw["image"])) as image:
                image.load()
                width, height = image.size
                mode = image.mode
            lowered = spatial_caption.lower()
            matched = [phrase for phrase in RELATION_PHRASES if phrase in lowered]
            relation_counts.update(matched)
            samples.append(
                {
                    "id": raw["id"],
                    "shard": shard.name,
                    "width": width,
                    "height": height,
                    "mode": mode,
                    "relations": matched,
                    "image_origin": metadata.get("image_origin"),
                    "image_id": metadata.get("image_id"),
                }
            )
            if len(samples) >= limit:
                break
        if len(samples) >= limit:
            break
    if len(samples) < limit:
        raise ValueError(f"Requested {limit} SPRIGHT samples but found {len(samples)}")
    return {
        "status": "ok",
        "shards_available": len(shards),
        "samples_checked": len(samples),
        "samples_with_known_relation": sum(bool(sample["relations"]) for sample in samples),
        "relation_counts": dict(relation_counts),
        "sample_preview": samples[:3],
    }


def _zip_basenames(path: Path) -> set[str]:
    with zipfile.ZipFile(path) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise ValueError(f"Corrupt VSR zip entry: {bad}")
        return {Path(name).name for name in archive.namelist() if not name.endswith("/")}


def validate_vsr(data_root: Path, limit: int, require_images: bool) -> dict[str, Any]:
    annotation_path = data_root / "vsr" / "annotations" / "zeroshot" / "train.jsonl"
    if not annotation_path.is_file():
        raise FileNotFoundError(f"VSR annotation missing: {annotation_path}")
    image_zip = data_root / "vsr" / "vsr_images.zip"
    if require_images and not image_zip.is_file():
        raise FileNotFoundError(f"VSR image archive missing: {image_zip}")
    image_names = _zip_basenames(image_zip) if image_zip.is_file() else set()
    required = {"image", "caption", "label", "relation", "subj", "obj"}
    relation_counts: collections.Counter[str] = collections.Counter()
    rows_checked = 0
    missing_images = 0
    role_text_mismatches = 0
    labels: collections.Counter[int] = collections.Counter()
    with annotation_path.open() as handle:
        for line in handle:
            row = json.loads(line)
            missing = required - row.keys()
            if missing:
                raise ValueError(f"VSR row is missing fields: {sorted(missing)}")
            caption = str(row["caption"]).lower()
            if str(row["subj"]).lower() not in caption or str(row["obj"]).lower() not in caption:
                role_text_mismatches += 1
            relation_counts[str(row["relation"])] += 1
            labels[int(row["label"])] += 1
            if image_names and Path(str(row["image"])).name not in image_names:
                missing_images += 1
            rows_checked += 1
            if rows_checked >= limit:
                break
    if rows_checked < limit:
        raise ValueError(f"Requested {limit} VSR rows but found {rows_checked}")
    if image_names and missing_images:
        raise ValueError(f"{missing_images} sampled VSR images are absent from the zip")
    return {
        "status": "ok",
        "rows_checked": rows_checked,
        "image_archive_checked": bool(image_names),
        "images_in_archive": len(image_names),
        "missing_images": missing_images,
        "role_text_mismatches": role_text_mismatches,
        "labels": dict(labels),
        "relation_counts": dict(relation_counts),
    }


def validate_torch_loaders(data_root: Path, require_vsr_images: bool) -> dict[str, Any]:
    # Defer torch import so archive-only checks work in low-memory containers.
    from datasets.drsa_dataset import SprightCocoIterableDataset, VsrRelationDataset

    spright = SprightCocoIterableDataset(
        data_root / "spright_coco" / "data",
        shuffle_shards=False,
        shuffle_buffer=0,
    )
    spright_record = next(iter(spright))
    result: dict[str, Any] = {
        "spright": {
            "sample_id": spright_record["sample_id"],
            "image_size": list(spright_record["image"].size),
            "has_spatial_caption": bool(spright_record["spatial_caption"]),
            "has_original_caption": bool(spright_record["original_caption"]),
        }
    }
    image_zip = data_root / "vsr" / "vsr_images.zip"
    if image_zip.is_file():
        vsr = VsrRelationDataset(
            data_root / "vsr" / "annotations" / "zeroshot" / "train.jsonl",
            image_zip,
        )
        vsr_record = vsr[0]
        result["vsr"] = {
            "rows": len(vsr),
            "image_size": list(vsr_record["image"].size),
            "relation": vsr_record["relation"],
            "subject": vsr_record["subj"],
            "object": vsr_record["obj"],
            "label": vsr_record["label"],
        }
    elif require_vsr_images:
        raise FileNotFoundError(image_zip)
    else:
        result["vsr"] = {"status": "image archive pending"}
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--spright-samples", type=int, default=16)
    parser.add_argument("--vsr-samples", type=int, default=64)
    parser.add_argument("--require-vsr-images", action="store_true")
    parser.add_argument("--skip-torch-loaders", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.spright_samples < 1 or args.vsr_samples < 1:
        raise ValueError("Smoke sample counts must be positive")
    result: dict[str, Any] = {
        "status": "ok",
        "data_root": str(args.data_root.resolve()),
        "spright_coco": validate_spright(args.data_root, args.spright_samples),
        "vsr": validate_vsr(args.data_root, args.vsr_samples, args.require_vsr_images),
    }
    if args.skip_torch_loaders:
        result["torch_loaders"] = {"status": "skipped by --skip-torch-loaders"}
    else:
        result["torch_loaders"] = validate_torch_loaders(
            args.data_root, args.require_vsr_images
        )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
