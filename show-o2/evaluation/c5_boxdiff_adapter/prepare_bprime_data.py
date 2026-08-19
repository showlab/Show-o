#!/usr/bin/env python3
"""Build balanced B-prime spatial supervision from SPRIGHT images and COCO boxes."""

from __future__ import annotations

import argparse
import collections
import hashlib
import itertools
import json
import random
import re
import tarfile
import time
import zipfile
from pathlib import Path
from typing import Any


RELATION_TEXT = {"left": "left of", "right": "right of", "above": "above", "below": "below"}
RELATIONS = tuple(RELATION_TEXT)
COCO_ID_RE = re.compile(r"(\d{12})(?=\.[A-Za-z]+$)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spright-root", type=Path, default=Path("/root/autodl-tmp/data/drsa/spright_coco/data"))
    parser.add_argument(
        "--coco-annotations-zip",
        type=Path,
        default=Path("/root/autodl-tmp/datasets/COCO2017/downloads/modelscope/annotations_trainval2017.zip"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--max-train-per-relation", type=int, default=2000)
    parser.add_argument("--max-validation-per-relation", type=int, default=256)
    parser.add_argument("--min-visible-fraction", type=float, default=0.70)
    parser.add_argument("--min-box-area", type=float, default=0.01)
    parser.add_argument("--min-axis-gap", type=float, default=0.12)
    parser.add_argument("--dominance-ratio", type=float, default=1.25)
    parser.add_argument("--max-box-iou", type=float, default=0.20)
    parser.add_argument("--max-min-box-overlap", type=float, default=0.50)
    return parser.parse_args()


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_coco(annotation_zip: Path):
    images: dict[int, dict[str, Any]] = {}
    annotations: dict[int, list[dict[str, Any]]] = collections.defaultdict(list)
    categories: dict[int, str] = {}
    names = ("annotations/instances_train2017.json", "annotations/instances_val2017.json")
    with zipfile.ZipFile(annotation_zip) as archive:
        for name in names:
            with archive.open(name) as handle:
                payload = json.load(handle)
            split = Path(name).stem.removeprefix("instances_")
            for category in payload["categories"]:
                categories[int(category["id"])] = str(category["name"])
            for image in payload["images"]:
                item = dict(image)
                item["coco2017_split"] = split
                images[int(image["id"])] = item
            for annotation in payload["annotations"]:
                if int(annotation.get("iscrowd", 0)) == 0:
                    annotations[int(annotation["image_id"])].append(annotation)
            del payload
    return images, annotations, categories


def crop_box(bbox, width, height, *, min_visible_fraction, min_box_area):
    x, y, box_width, box_height = (float(value) for value in bbox)
    if box_width <= 0.0 or box_height <= 0.0:
        return None
    side = float(min(width, height))
    crop_x = (float(width) - side) * 0.5
    crop_y = (float(height) - side) * 0.5
    left, top = max(x, crop_x), max(y, crop_y)
    right, bottom = min(x + box_width, crop_x + side), min(y + box_height, crop_y + side)
    if right <= left or bottom <= top:
        return None
    visible = (right - left) * (bottom - top) / (box_width * box_height)
    normalized = (
        (left - crop_x) / side,
        (top - crop_y) / side,
        (right - crop_x) / side,
        (bottom - crop_y) / side,
    )
    area = (normalized[2] - normalized[0]) * (normalized[3] - normalized[1])
    if visible < min_visible_fraction or area < min_box_area:
        return None
    return tuple(max(0.0, min(1.0, value)) for value in normalized)


def object_record(annotation, box, categories):
    category_id = int(annotation["category_id"])
    return {
        "phrase": categories[category_id],
        "box": list(box),
        "category_id": category_id,
        "annotation_id": int(annotation["id"]),
    }


def center(item):
    x0, y0, x1, y1 = item["box"]
    return 0.5 * (x0 + x1), 0.5 * (y0 + y1)


def box_overlap(first, second):
    ax0, ay0, ax1, ay1 = first["box"]
    bx0, by0, bx1, by1 = second["box"]
    intersection = max(0.0, min(ax1, bx1) - max(ax0, bx0)) * max(
        0.0, min(ay1, by1) - max(ay0, by0)
    )
    first_area = (ax1 - ax0) * (ay1 - ay0)
    second_area = (bx1 - bx0) * (by1 - by0)
    union = first_area + second_area - intersection
    return (
        intersection / max(union, 1e-12),
        intersection / max(min(first_area, second_area), 1e-12),
    )


def directional_pair(
    first, second, *, min_axis_gap, dominance_ratio, max_box_iou, max_min_box_overlap
):
    iou, min_overlap = box_overlap(first, second)
    if iou > max_box_iou or min_overlap > max_min_box_overlap:
        return None
    first_x, first_y = center(first)
    second_x, second_y = center(second)
    dx, dy = second_x - first_x, second_y - first_y
    abs_x, abs_y = abs(dx), abs(dy)
    if abs_x >= min_axis_gap and abs_x >= dominance_ratio * abs_y:
        return ("left", first, second, abs_x) if dx > 0.0 else ("left", second, first, abs_x)
    if abs_y >= min_axis_gap and abs_y >= dominance_ratio * abs_x:
        return ("above", first, second, abs_y) if dy > 0.0 else ("above", second, first, abs_y)
    return None


def invert_relation(relation: str) -> str:
    return {"left": "right", "right": "left", "above": "below", "below": "above"}[relation]


def make_record(*, shard, stem, image_id, image_info, relation, first, second, caption):
    prompt = f"a photo of {first['phrase']} {RELATION_TEXT[relation]} {second['phrase']}"
    return {
        "sample_id": f"{shard}:{stem}:{relation}",
        "shard": shard,
        "tar_stem": stem,
        "image_id": image_id,
        "image_origin": "COCO",
        "coco2017_split": image_info["coco2017_split"],
        "source_width": int(image_info["width"]),
        "source_height": int(image_info["height"]),
        "prompt": prompt,
        "relation": relation,
        "objects": [first, second],
        "original_caption": str(caption.get("original_caption", "")),
        "source_spatial_caption": str(caption.get("spatial_caption", "")),
        "box_source": "COCO instances train2017/val2017, center-crop adjusted",
    }


def records_for_sample(*, shard, stem, metadata, caption, images, annotations, categories, args):
    match = COCO_ID_RE.search(str(metadata.get("image_id", "")))
    if metadata.get("image_origin") != "COCO" or match is None:
        return [], "not_coco"
    image_id = int(match.group(1))
    image_info = images.get(image_id)
    if image_info is None:
        return [], "missing_coco2017_id"
    grouped = collections.defaultdict(list)
    for annotation in annotations.get(image_id, ()):
        box = crop_box(
            annotation["bbox"],
            int(image_info["width"]),
            int(image_info["height"]),
            min_visible_fraction=args.min_visible_fraction,
            min_box_area=args.min_box_area,
        )
        if box is not None:
            grouped[int(annotation["category_id"])].append((annotation, box))
    unique = [
        object_record(items[0][0], items[0][1], categories)
        for items in grouped.values()
        if len(items) == 1
    ]
    if len(unique) < 2:
        return [], "fewer_than_two_unique_objects"
    horizontal, vertical = [], []
    for first, second in itertools.combinations(unique, 2):
        parsed = directional_pair(
            first,
            second,
            min_axis_gap=args.min_axis_gap,
            dominance_ratio=args.dominance_ratio,
            max_box_iou=args.max_box_iou,
            max_min_box_overlap=args.max_min_box_overlap,
        )
        if parsed is not None:
            (horizontal if parsed[0] == "left" else vertical).append(parsed)
    selected = []
    if horizontal:
        selected.append(max(horizontal, key=lambda item: item[3]))
    if vertical:
        selected.append(max(vertical, key=lambda item: item[3]))
    if not selected:
        return [], "ambiguous_geometry"
    output = []
    for relation, first, second, _strength in selected:
        common = dict(shard=shard, stem=stem, image_id=image_id, image_info=image_info, caption=caption)
        output.append(make_record(relation=relation, first=first, second=second, **common))
        output.append(make_record(
            relation=invert_relation(relation), first=second, second=first, **common
        ))
    return output, None


def scan_spright(args, images, annotations, categories):
    candidates = []
    counters = collections.Counter()
    shards = sorted(args.spright_root.glob("*.tar"), key=lambda path: int(path.stem))
    if not shards:
        raise FileNotFoundError(f"No SPRIGHT shards found in {args.spright_root}")
    for shard_index, shard in enumerate(shards, start=1):
        captions = {}
        with tarfile.open(shard, mode="r:") as archive:
            for member in archive:
                if not member.isfile():
                    continue
                if member.name.endswith(".metadata.json"):
                    stem, kind = member.name[: -len(".metadata.json")], "metadata"
                elif member.name.endswith(".json"):
                    stem, kind = member.name[: -len(".json")], "caption"
                else:
                    continue
                extracted = archive.extractfile(member)
                if extracted is None:
                    continue
                payload = json.load(extracted)
                if kind == "caption":
                    captions[stem] = payload
                    continue
                counters["samples_seen"] += 1
                rows, reason = records_for_sample(
                    shard=shard.name,
                    stem=stem,
                    metadata=payload,
                    caption=captions.pop(stem, {}),
                    images=images,
                    annotations=annotations,
                    categories=categories,
                    args=args,
                )
                if reason is None:
                    counters["accepted_images"] += 1
                    candidates.extend(rows)
                else:
                    counters[f"rejected_{reason}"] += 1
        print(
            f"scanned {shard_index}/{len(shards)} {shard.name}: "
            f"samples={counters['samples_seen']} candidates={len(candidates)}",
            flush=True,
        )
    return candidates, dict(counters)


def is_validation_image(image_id, seed, fraction):
    digest = hashlib.sha256(f"{seed}:{image_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64) < fraction


def balance(rows, *, seed, cap):
    grouped = {relation: [] for relation in RELATIONS}
    for row in rows:
        grouped[row["relation"]].append(row)
    before = {relation: len(values) for relation, values in grouped.items()}
    take = min(min(before.values()), cap)
    if take < 1:
        raise RuntimeError(f"Cannot balance spatial rows: {before}")
    selected = []
    for index, relation in enumerate(RELATIONS):
        random.Random(seed + 1009 * index).shuffle(grouped[relation])
        selected.extend(grouped[relation][:take])
    random.Random(seed + 7919).shuffle(selected)
    return selected, before


def main() -> int:
    args = parse_args()
    if not 0.0 < args.validation_fraction < 1.0:
        raise ValueError("validation-fraction must be in (0, 1)")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    images, annotations, categories = load_coco(args.coco_annotations_zip)
    candidates, rejection_counts = scan_spright(args, images, annotations, categories)
    train_candidates, validation_candidates = [], []
    for row in candidates:
        target = validation_candidates if is_validation_image(
            int(row["image_id"]), args.seed, args.validation_fraction
        ) else train_candidates
        target.append(row)
    train, train_before = balance(train_candidates, seed=args.seed, cap=args.max_train_per_relation)
    validation, validation_before = balance(
        validation_candidates, seed=args.seed + 1, cap=args.max_validation_per_relation
    )
    train_path, validation_path = args.output_dir / "train.jsonl", args.output_dir / "validation.jsonl"
    write_jsonl(train_path, train)
    write_jsonl(validation_path, validation)
    train_images = {int(row["image_id"]) for row in train}
    validation_images = {int(row["image_id"]) for row in validation}
    if train_images & validation_images:
        raise AssertionError("train/validation image leakage detected")
    audit = {
        "protocol": "bprime-coco-real-box-balanced-v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": time.time() - started,
        "config": vars(args) | {
            "spright_root": str(args.spright_root),
            "coco_annotations_zip": str(args.coco_annotations_zip),
            "output_dir": str(args.output_dir),
        },
        "coco_image_ids": len(images),
        "coco_annotated_image_ids": len(annotations),
        "rejection_counts": rejection_counts,
        "candidate_rows": len(candidates),
        "train_direction_counts_before_balance": train_before,
        "validation_direction_counts_before_balance": validation_before,
        "train_rows": len(train),
        "validation_rows": len(validation),
        "train_unique_images": len(train_images),
        "validation_unique_images": len(validation_images),
        "image_id_overlap": 0,
        "train_sha256": sha256(train_path),
        "validation_sha256": sha256(validation_path),
    }
    atomic_json(args.output_dir / "audit.json", audit)
    print(json.dumps(audit, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
