#!/usr/bin/env python3
"""Build a balanced single/dual-axis Mask-logits manifest from SPRIGHT and COCO."""

from __future__ import annotations

import argparse
import collections
import hashlib
import itertools
import json
import random
import sys
import tarfile
import time
import zipfile
from pathlib import Path
from typing import Any, Iterable

try:
    import ijson
except ImportError as error:
    raise SystemExit(
        "prepare_multiaxis_manifest.py requires ijson; run it with the local "
        "/root/autodl-tmp/venvs/toolvista environment"
    ) from error

ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.c5_boxdiff_adapter.geneval_geometry import (
    POSITION_THRESHOLD,
    RELATION_THRESHOLD,
    geneval_relation_scores,
)
from evaluation.c5_boxdiff_adapter.prepare_bprime_data import (
    COCO_ID_RE,
    box_overlap,
    crop_box,
    object_record,
)


RELATIONS = ("left", "right", "above", "below")
INVERSE_RELATION = {
    "left": "right",
    "right": "left",
    "above": "below",
    "below": "above",
}
RELATION_TEXT = {
    "left": "to the left of",
    "right": "to the right of",
    "above": "above",
    "below": "below",
}
GROUPS = ("diagonal_negative", "diagonal_positive", "horizontal", "vertical")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spright-root",
        type=Path,
        default=WORKSPACE / "data/drsa/spright_coco/data",
    )
    parser.add_argument(
        "--coco-annotations-zip",
        type=Path,
        default=WORKSPACE
        / "datasets/COCO2017/downloads/modelscope/annotations_trainval2017.zip",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--train-rows", type=int, default=17100)
    parser.add_argument("--validation-rows", type=int, default=976)
    parser.add_argument("--min-visible-fraction", type=float, default=0.70)
    parser.add_argument("--min-box-area", type=float, default=0.01)
    parser.add_argument("--max-box-iou", type=float, default=0.20)
    parser.add_argument("--max-min-box-overlap", type=float, default=0.50)
    parser.add_argument("--max-selected-pairs-per-image", type=int, default=2)
    parser.add_argument(
        "--position-threshold", type=float, default=POSITION_THRESHOLD
    )
    parser.add_argument(
        "--relation-threshold", type=float, default=RELATION_THRESHOLD
    )
    parser.add_argument("--training-margin", type=float, default=0.10)
    return parser.parse_args()


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
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


def is_validation_image(image_id: int, seed: int, fraction: float) -> bool:
    digest = hashlib.sha256(f"{seed}:{image_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64) < fraction


def load_spright_samples(root: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    shards = sorted(root.glob("*.tar"), key=lambda path: int(path.stem))
    if not shards:
        raise FileNotFoundError(f"No SPRIGHT shards found in {root}")
    samples = []
    counters = collections.Counter()
    for shard_index, shard in enumerate(shards, start=1):
        pending: dict[str, dict[str, dict[str, Any]]] = collections.defaultdict(dict)
        with tarfile.open(shard, mode="r:") as archive:
            for member in archive:
                if not member.isfile():
                    continue
                if member.name.endswith(".metadata.json"):
                    stem = member.name[: -len(".metadata.json")]
                    kind = "metadata"
                elif member.name.endswith(".json"):
                    stem = member.name[: -len(".json")]
                    kind = "caption"
                else:
                    continue
                extracted = archive.extractfile(member)
                if extracted is None:
                    continue
                pending[stem][kind] = json.load(extracted)
                if not {"metadata", "caption"}.issubset(pending[stem]):
                    continue
                payload = pending.pop(stem)
                metadata, caption = payload["metadata"], payload["caption"]
                counters["samples_seen"] += 1
                match = COCO_ID_RE.search(str(metadata.get("image_id", "")))
                if metadata.get("image_origin") != "COCO" or match is None:
                    counters["rejected_not_coco"] += 1
                    continue
                samples.append({
                    "shard": shard.name,
                    "tar_stem": stem,
                    "image_id": int(match.group(1)),
                    "original_caption": str(caption.get("original_caption", "")),
                    "source_spatial_caption": str(caption.get("spatial_caption", "")),
                })
        print(
            f"SPRIGHT {shard_index}/{len(shards)} {shard.name}: "
            f"samples={counters['samples_seen']} coco={len(samples)}",
            flush=True,
        )
    counters["coco_samples"] = len(samples)
    counters["unique_coco_image_ids"] = len({sample["image_id"] for sample in samples})
    return samples, dict(counters)


def stream_relevant_coco(
    annotation_zip: Path, relevant_ids: set[int]
) -> tuple[
    dict[int, dict[str, Any]],
    dict[int, list[dict[str, Any]]],
    dict[int, str],
]:
    images: dict[int, dict[str, Any]] = {}
    annotations: dict[int, list[dict[str, Any]]] = collections.defaultdict(list)
    categories: dict[int, str] = {}
    members = (
        "annotations/instances_train2017.json",
        "annotations/instances_val2017.json",
    )
    with zipfile.ZipFile(annotation_zip) as archive:
        for member in members:
            split = Path(member).stem.removeprefix("instances_")
            with archive.open(member) as handle:
                for item in ijson.items(handle, "images.item", use_float=True):
                    image_id = int(item["id"])
                    if image_id in relevant_ids:
                        row = dict(item)
                        row["coco2017_split"] = split
                        images[image_id] = row
            with archive.open(member) as handle:
                for item in ijson.items(handle, "annotations.item", use_float=True):
                    image_id = int(item["image_id"])
                    if image_id in relevant_ids and int(item.get("iscrowd", 0)) == 0:
                        annotations[image_id].append(dict(item))
            if not categories:
                with archive.open(member) as handle:
                    for item in ijson.items(handle, "categories.item", use_float=True):
                        categories[int(item["id"])] = str(item["name"])
            print(
                f"COCO {split}: images={len(images)} "
                f"annotated_images={len(annotations)}",
                flush=True,
            )
    return images, annotations, categories


def relation_group(relations: tuple[str, ...]) -> str:
    if len(relations) == 1:
        return "horizontal" if relations[0] in ("left", "right") else "vertical"
    horizontal, vertical = relations
    if (horizontal, vertical) in (("left", "above"), ("right", "below")):
        return "diagonal_negative"
    return "diagonal_positive"


def pair_relations(
    first: dict[str, Any],
    second: dict[str, Any],
    *,
    position_threshold: float,
    relation_threshold: float,
) -> tuple[tuple[str, ...], dict[str, float]]:
    scores = dict(geneval_relation_scores(
        first["box"], second["box"], position_threshold=position_threshold
    ))
    horizontal = max(("left", "right"), key=scores.get)
    vertical = max(("above", "below"), key=scores.get)
    relations = []
    if scores[horizontal] > relation_threshold:
        relations.append(horizontal)
    if scores[vertical] > relation_threshold:
        relations.append(vertical)
    return tuple(relations), scores


def pair_candidates(
    samples: list[dict[str, Any]],
    images: dict[int, dict[str, Any]],
    annotations: dict[int, list[dict[str, Any]]],
    categories: dict[int, str],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    candidates: list[dict[str, Any]] = []
    counters = collections.Counter()
    for sample in samples:
        image_id = int(sample["image_id"])
        image_info = images.get(image_id)
        if image_info is None:
            counters["missing_coco_image"] += 1
            continue
        grouped: dict[int, list[tuple[dict[str, Any], tuple[float, ...]]]] = (
            collections.defaultdict(list)
        )
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
            counters["fewer_than_two_unique_objects"] += 1
            continue
        found = False
        for first, second in itertools.combinations(unique, 2):
            iou, min_overlap = box_overlap(first, second)
            if iou > args.max_box_iou or min_overlap > args.max_min_box_overlap:
                counters["overlapping_pair"] += 1
                continue
            relations, scores = pair_relations(
                first,
                second,
                position_threshold=args.position_threshold,
                relation_threshold=args.relation_threshold,
            )
            if not relations:
                counters["below_relation_threshold"] += 1
                continue
            found = True
            group = relation_group(relations)
            target_scores = {relation: scores[relation] for relation in relations}
            quality = min(target_scores.values())
            candidate = {
                **sample,
                "image_info": image_info,
                "first": first,
                "second": second,
                "relations": relations,
                "source_scores": target_scores,
                "group": group,
                "quality": quality,
            }
            candidates.append(candidate)
            counters[f"eligible_pair_{group}"] += 1
        counters["eligible_images" if found else "images_without_eligible_pair"] += 1
    counters["candidate_pairs"] = len(candidates)
    counters["candidate_unique_images"] = len({int(row["image_id"]) for row in candidates})
    return candidates, dict(counters)


def select_balanced_pairs(
    candidates: Iterable[dict[str, Any]],
    *,
    target_rows: int,
    seed: int,
    max_pairs_per_image: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if target_rows < 2 or target_rows % 2:
        raise ValueError("target row count must be a positive even number")
    if max_pairs_per_image < 1:
        raise ValueError("max-pairs-per-image must be positive")
    target_pairs = target_rows // 2
    grouped = {group: [] for group in GROUPS}
    for candidate in candidates:
        grouped[str(candidate["group"])].append(candidate)
    for index, group in enumerate(GROUPS):
        random.Random(seed + 1009 * index).shuffle(grouped[group])

    base, remainder = divmod(target_pairs, len(GROUPS))
    quotas = {
        group: base + int(index < remainder)
        for index, group in enumerate(GROUPS)
    }
    selected = []
    selected_image_counts: collections.Counter[int] = collections.Counter()
    cursors = {group: 0 for group in GROUPS}

    def take_from(group: str, limit: int) -> int:
        taken = 0
        values = grouped[group]
        while cursors[group] < len(values) and taken < limit:
            candidate = values[cursors[group]]
            cursors[group] += 1
            image_id = int(candidate["image_id"])
            if selected_image_counts[image_id] >= max_pairs_per_image:
                continue
            selected_image_counts[image_id] += 1
            selected.append(candidate)
            taken += 1
        return taken

    shortfall = 0
    for group in GROUPS:
        shortfall += quotas[group] - take_from(group, quotas[group])

    while len(selected) < target_pairs:
        progress = False
        for group in GROUPS:
            if len(selected) >= target_pairs:
                break
            if take_from(group, 1):
                progress = True
        if not progress:
            raise RuntimeError(
                f"not enough bounded image pairs for {target_pairs}; got {len(selected)}"
            )
    random.Random(seed + 7919).shuffle(selected)
    counts = collections.Counter(str(item["group"]) for item in selected)
    return selected, {
        "target_rows": target_rows,
        "target_pairs": target_pairs,
        "available_by_group": {
            group: len(grouped[group]) for group in GROUPS
        },
        "requested_pair_quotas": quotas,
        "initial_quota_shortfall": shortfall,
        "selected_pairs_by_group": dict(counts),
        "selected_single_axis_pairs": counts["horizontal"] + counts["vertical"],
        "selected_dual_axis_pairs": (
            counts["diagonal_negative"] + counts["diagonal_positive"]
        ),
        "selected_unique_images": len(selected_image_counts),
        "maximum_selected_pairs_per_image": max(selected_image_counts.values()),
    }


def prompt_for(first: str, second: str, relations: tuple[str, ...]) -> str:
    relation_text = " and ".join(RELATION_TEXT[relation] for relation in relations)
    return f"a photo of {first} {relation_text} {second}"


def make_record(
    candidate: dict[str, Any],
    *,
    first: dict[str, Any],
    second: dict[str, Any],
    relations: tuple[str, ...],
    scores: dict[str, float],
    args: argparse.Namespace,
) -> dict[str, Any]:
    relation_id = "+".join(relations)
    image_info = candidate["image_info"]
    return {
        "sample_id": (
            f"{candidate['shard']}:{candidate['tar_stem']}:{relation_id}:"
            f"{first['annotation_id']}-{second['annotation_id']}"
        ),
        "shard": candidate["shard"],
        "tar_stem": candidate["tar_stem"],
        "image_id": int(candidate["image_id"]),
        "image_origin": "COCO",
        "coco2017_split": image_info["coco2017_split"],
        "source_width": int(image_info["width"]),
        "source_height": int(image_info["height"]),
        "prompt": prompt_for(first["phrase"], second["phrase"], relations),
        "relation": relations[0],
        "relations": list(relations),
        "relation_schema": "orthogonal-multilabel-v1",
        "objects": [first, second],
        "original_caption": candidate["original_caption"],
        "source_spatial_caption": candidate["source_spatial_caption"],
        "box_source": "COCO instances train2017/val2017, center-crop adjusted",
        "geneval_supervision": {
            "expected_relation": relations[0],
            "expected_relations": list(relations),
            "source_box_score": scores[relations[0]],
            "source_box_scores": {
                relation: scores[relation] for relation in relations
            },
            "position_threshold": args.position_threshold,
            "evaluation_threshold": args.relation_threshold,
            "training_margin": args.training_margin,
            "training_target": args.relation_threshold + args.training_margin,
        },
    }


def expand_directed_rows(
    pairs: list[dict[str, Any]], args: argparse.Namespace
) -> list[dict[str, Any]]:
    rows = []
    for candidate in pairs:
        relations = tuple(candidate["relations"])
        scores = dict(candidate["source_scores"])
        rows.append(make_record(
            candidate,
            first=candidate["first"],
            second=candidate["second"],
            relations=relations,
            scores=scores,
            args=args,
        ))
        inverse_relations = tuple(INVERSE_RELATION[value] for value in relations)
        inverse_scores = {
            INVERSE_RELATION[value]: scores[value] for value in relations
        }
        rows.append(make_record(
            candidate,
            first=candidate["second"],
            second=candidate["first"],
            relations=inverse_relations,
            scores=inverse_scores,
            args=args,
        ))
    return rows


def split_audit(rows: list[dict[str, Any]], selection: dict[str, Any]) -> dict[str, Any]:
    relation_counts = collections.Counter()
    pattern_counts = collections.Counter()
    for row in rows:
        relation_counts.update(row["relations"])
        pattern_counts["+".join(row["relations"])] += 1
    return {
        **selection,
        "output_rows": len(rows),
        "unique_images": len({int(row["image_id"]) for row in rows}),
        "single_axis_rows": sum(len(row["relations"]) == 1 for row in rows),
        "dual_axis_rows": sum(len(row["relations"]) == 2 for row in rows),
        "relation_target_counts": dict(relation_counts),
        "relation_pattern_counts": dict(pattern_counts),
    }


def main() -> int:
    args = parse_args()
    if not 0.0 < args.validation_fraction < 1.0:
        raise ValueError("validation-fraction must be in (0, 1)")
    if args.relation_threshold < 0.0 or args.training_margin < 0.0:
        raise ValueError("relation threshold and training margin must be non-negative")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    samples, spright_audit = load_spright_samples(args.spright_root)
    relevant_ids = {int(sample["image_id"]) for sample in samples}
    images, annotations, categories = stream_relevant_coco(
        args.coco_annotations_zip, relevant_ids
    )
    candidates, candidate_audit = pair_candidates(
        samples, images, annotations, categories, args
    )

    train_candidates = []
    validation_candidates = []
    for candidate in candidates:
        target = (
            validation_candidates
            if is_validation_image(
                int(candidate["image_id"]), args.seed, args.validation_fraction
            )
            else train_candidates
        )
        target.append(candidate)

    train_pairs, train_selection = select_balanced_pairs(
        train_candidates,
        target_rows=args.train_rows,
        seed=args.seed,
        max_pairs_per_image=args.max_selected_pairs_per_image,
    )
    validation_pairs, validation_selection = select_balanced_pairs(
        validation_candidates,
        target_rows=args.validation_rows,
        seed=args.seed + 1,
        max_pairs_per_image=args.max_selected_pairs_per_image,
    )
    train_rows = expand_directed_rows(train_pairs, args)
    validation_rows = expand_directed_rows(validation_pairs, args)
    train_path = args.output_dir / "train.jsonl"
    validation_path = args.output_dir / "validation.jsonl"
    write_jsonl(train_path, train_rows)
    write_jsonl(validation_path, validation_rows)

    train_images = {int(row["image_id"]) for row in train_rows}
    validation_images = {int(row["image_id"]) for row in validation_rows}
    if train_images & validation_images:
        raise AssertionError("train/validation image leakage detected")

    train_audit = split_audit(train_rows, train_selection)
    validation_audit = split_audit(validation_rows, validation_selection)
    train_audit["sha256"] = sha256(train_path)
    validation_audit["sha256"] = sha256(validation_path)
    audit = {
        "protocol": "mask-logits-orthogonal-multilabel-v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": time.time() - started,
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "schema": {
            "relation": "legacy primary direction",
            "relations": "one horizontal and/or one vertical direction",
            "maximum_relations_per_row": 2,
            "same_axis_mutual_exclusion": True,
            "orthogonal_coexistence": True,
        },
        "spright": spright_audit,
        "coco": {
            "relevant_image_ids": len(relevant_ids),
            "loaded_images": len(images),
            "loaded_annotated_images": len(annotations),
            "categories": len(categories),
        },
        "candidates": candidate_audit,
        "splits": {
            "train": train_audit,
            "validation": validation_audit,
        },
        "train_validation_image_overlap": 0,
        "examples": train_rows[:4],
    }
    atomic_json(args.output_dir / "audit.json", audit)
    print(json.dumps(audit, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
