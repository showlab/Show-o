#!/usr/bin/env python3
"""Build a decontaminated TIFA-12 training set from SPRIGHT and COCO truth."""

from __future__ import annotations

import argparse
import collections
import hashlib
import itertools
import json
import random
import re
import sys
import tarfile
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:
    import ijson
except ImportError as error:
    raise SystemExit(
        "prepare_external_tifa12_data.py requires ijson; use the toolvista venv"
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


DIMENSIONS = (
    "object",
    "animal/human",
    "attribute",
    "activity",
    "spatial",
    "location",
    "color",
    "counting",
    "food",
    "material",
    "shape",
    "other",
)
RELATION_TEXT = {
    "left": "to the left of",
    "right": "to the right of",
    "above": "above",
    "below": "below",
}
RELATION_AXIS = {
    "left": ("left", "right"),
    "right": ("left", "right"),
    "above": ("above", "below"),
    "below": ("above", "below"),
}
NUMBER_WORDS = ("one", "two", "three", "four")
ANIMAL_HUMAN = {
    "person", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
    "bear", "zebra", "giraffe",
}
FOOD = {
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake",
}
COLORS = (
    "black", "white", "red", "green", "blue", "yellow", "orange",
    "purple", "pink", "brown", "gray", "grey",
)
MATERIALS = {
    "wood": ("wood", "wooden"),
    "metal": ("metal", "metallic"),
    "plastic": ("plastic",),
    "glass": ("glass",),
    "leather": ("leather",),
    "fabric": ("fabric", "cloth"),
    "concrete": ("concrete",),
    "brick": ("brick",),
    "ceramic": ("ceramic",),
    "paper": ("paper",),
}
SHAPES = {
    "round": ("round", "circular"),
    "rectangular": ("rectangular", "rectangle-shaped"),
    "square": ("square",),
    "triangular": ("triangular", "triangle-shaped"),
    "cylindrical": ("cylindrical", "cylinder-shaped"),
    "octagonal": ("octagonal", "octagon-shaped"),
}
SHAPE_PRIORS = {
    "sports ball": "round",
    "frisbee": "round",
    "donut": "round",
    "pizza": "round",
    "stop sign": "octagonal",
    "book": "rectangular",
    "cell phone": "rectangular",
    "keyboard": "rectangular",
    "tv": "rectangular",
    "traffic light": "rectangular",
    "bottle": "cylindrical",
}
ACTIVITIES = (
    "riding", "sitting", "standing", "walking", "running", "playing",
    "eating", "cooking", "holding", "flying", "driving", "skiing",
    "surfing", "skateboarding", "sleeping", "reading", "talking",
    "working", "jumping", "swimming", "cutting", "throwing", "catching",
    "grazing", "parking", "posing", "waiting",
)
LOCATIONS = (
    "living room", "dining room", "tennis court", "baseball field",
    "train station", "parking lot", "bathroom", "kitchen", "bedroom",
    "restaurant", "airport", "runway", "beach", "street", "road",
    "sidewalk", "field", "park", "room", "office", "store", "ocean",
    "snow", "water",
)
INDOOR_LOCATIONS = {
    "living room", "dining room", "bathroom", "kitchen", "bedroom",
    "restaurant", "room", "office", "store",
}
OUTDOOR_LOCATIONS = {
    "tennis court", "baseball field", "parking lot", "runway", "beach",
    "street", "road", "sidewalk", "field", "park", "ocean", "snow",
    "water",
}
WEATHER = {
    "snowy": ("snowy", "in the snow"),
    "rainy": ("rainy", "in the rain"),
    "sunny": ("sunny",),
    "cloudy": ("cloudy", "overcast"),
}


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
    parser.add_argument(
        "--tifa-source",
        type=Path,
        default=WORKSPACE
        / "data/showo2_self_cycle/tifa_v1/source/tifa_v1.0_question_answers.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--min-visible-fraction", type=float, default=0.70)
    parser.add_argument("--min-box-area", type=float, default=0.01)
    parser.add_argument("--max-box-iou", type=float, default=0.20)
    parser.add_argument("--max-min-box-overlap", type=float, default=0.50)
    parser.add_argument("--position-threshold", type=float, default=POSITION_THRESHOLD)
    parser.add_argument("--relation-threshold", type=float, default=RELATION_THRESHOLD)
    parser.add_argument("--min-rows", type=int, default=30000)
    parser.add_argument("--min-spatial-rows", type=int, default=6000)
    parser.add_argument("--min-rare-questions", type=int, default=100)
    return parser.parse_args()


def normalized(value: Any) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(value).casefold()))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(path)


def plural(noun: str) -> str:
    if noun.endswith("s"):
        return noun
    if noun.endswith("y") and len(noun) > 1 and noun[-2] not in "aeiou":
        return noun[:-1] + "ies"
    return noun + "s"


def stable_validation(image_id: int, seed: int, fraction: float) -> bool:
    value = int.from_bytes(
        hashlib.sha256(f"{seed}:{image_id}".encode()).digest()[:8], "big"
    )
    return value / float(2**64) < fraction


def deterministic_choices(
    answer: str, pool: Sequence[str], *, key: str, count: int = 4
) -> tuple[list[str], int]:
    values = [value for value in dict.fromkeys(pool) if value != answer]
    random.Random(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")).shuffle(values)
    choices = [answer, *values[: count - 1]]
    if len(choices) < 2:
        raise ValueError(f"not enough choices for {answer!r}")
    return choices, 0


def qa(
    dimension: str,
    element: str,
    question: str,
    answer: str,
    choices: Sequence[str],
    target_index: int,
    source: str,
) -> dict[str, Any]:
    if dimension not in DIMENSIONS:
        raise ValueError(dimension)
    if not 0 <= target_index < len(choices) or choices[target_index] != answer:
        raise ValueError(f"invalid answer target for {question!r}")
    return {
        "answer": answer,
        "choices": list(choices),
        "element": element,
        "element_type": dimension,
        "question": question,
        "source": source,
        "target_index": target_index,
    }


def load_forbidden(tifa_source: Path) -> tuple[set[int], set[str], dict[str, int]]:
    rows = json.loads(tifa_source.read_text(encoding="utf-8"))
    image_ids = {
        int(row["coco_val_id"])
        for row in rows
        if str(row.get("id", "")).startswith("coco_") and row.get("coco_val_id")
    }
    captions = {normalized(row["caption"]) for row in rows if row.get("caption")}
    local_files = [
        *sorted((WORKSPACE / "data/showo2_layout").glob("*.jsonl")),
        *sorted((WORKSPACE / "data/visor").glob("*.jsonl")),
        *sorted((WORKSPACE / "data/showo2_self_cycle/unicycle").glob("*.jsonl")),
    ]
    local_prompts: set[str] = set()
    for path in local_files:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                for field in ("prompt", "caption", "short_description"):
                    if row.get(field):
                        local_prompts.add(normalized(row[field]))
    captions.update(local_prompts)
    return image_ids, captions, {
        "official_tifa_questions": len(rows),
        "official_tifa_coco_image_ids": len(image_ids),
        "official_tifa_captions": len({normalized(row["caption"]) for row in rows}),
        "local_evaluation_prompts": len(local_prompts),
    }


def load_spright(root: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    shards = sorted(root.glob("*.tar"), key=lambda path: int(path.stem))
    if not shards:
        raise FileNotFoundError(root)
    rows: list[dict[str, Any]] = []
    counters: collections.Counter[str] = collections.Counter()
    for shard_index, shard in enumerate(shards, 1):
        pending: dict[str, dict[str, Any]] = collections.defaultdict(dict)
        image_stems: set[str] = set()
        with tarfile.open(shard, mode="r:") as archive:
            for member in archive:
                if not member.isfile():
                    continue
                if member.name.endswith(".jpg"):
                    image_stems.add(member.name[:-4])
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
                if extracted is not None:
                    pending[stem][kind] = json.load(extracted)
        for stem, payload in pending.items():
            if stem not in image_stems or not {"metadata", "caption"}.issubset(payload):
                counters["incomplete_tar_sample"] += 1
                continue
            metadata = payload["metadata"]
            match = COCO_ID_RE.search(str(metadata.get("image_id", "")))
            if metadata.get("image_origin") != "COCO" or match is None:
                counters["not_coco"] += 1
                continue
            caption = payload["caption"]
            rows.append({
                "image_id": int(match.group(1)),
                "original_caption": str(caption.get("original_caption", "")).strip(),
                "shard": shard.name,
                "spatial_caption": str(caption.get("spatial_caption", "")).strip(),
                "tar_stem": stem,
            })
        print(
            f"SPRIGHT {shard_index}/{len(shards)} rows={len(rows)} "
            f"incomplete={counters['incomplete_tar_sample']}",
            flush=True,
        )
    counters["rows"] = len(rows)
    counters["unique_image_ids"] = len({row["image_id"] for row in rows})
    return rows, dict(counters)


def load_coco(
    annotation_zip: Path, relevant_ids: set[int]
) -> tuple[dict[int, dict[str, Any]], dict[int, list[dict[str, Any]]], dict[int, str]]:
    import zipfile

    images: dict[int, dict[str, Any]] = {}
    annotations: dict[int, list[dict[str, Any]]] = collections.defaultdict(list)
    categories: dict[int, str] = {}
    with zipfile.ZipFile(annotation_zip) as archive:
        for member in (
            "annotations/instances_train2017.json",
            "annotations/instances_val2017.json",
        ):
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
                f"COCO {split} relevant_images={len(images)} "
                f"annotated={len(annotations)}",
                flush=True,
            )
    return images, annotations, categories


def visible_objects(
    image: Mapping[str, Any],
    annotations: Sequence[Mapping[str, Any]],
    categories: Mapping[int, str],
    *,
    min_visible_fraction: float,
    min_box_area: float,
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for annotation in annotations:
        box = crop_box(
            annotation["bbox"],
            int(image["width"]),
            int(image["height"]),
            min_visible_fraction=min_visible_fraction,
            min_box_area=min_box_area,
        )
        if box is None:
            continue
        record = object_record(dict(annotation), box, categories)
        grouped[str(record["phrase"])].append(record)
    return grouped


def best_spatial_pair(
    grouped: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    max_box_iou: float,
    max_min_box_overlap: float,
    position_threshold: float,
    relation_threshold: float,
) -> dict[str, Any] | None:
    unique = [values[0] for values in grouped.values() if len(values) == 1]
    candidates = []
    for first, second in itertools.combinations(unique, 2):
        iou, min_overlap = box_overlap(first, second)
        if iou > max_box_iou or min_overlap > max_min_box_overlap:
            continue
        scores = dict(geneval_relation_scores(
            first["box"], second["box"], position_threshold=position_threshold
        ))
        horizontal = max(("left", "right"), key=scores.get)
        vertical = max(("above", "below"), key=scores.get)
        relations = tuple(
            relation
            for relation in (horizontal, vertical)
            if scores[relation] > relation_threshold
        )
        if relations:
            candidates.append((min(scores[value] for value in relations), first, second, relations, scores))
    if not candidates:
        return None
    quality, first, second, relations, scores = max(candidates, key=lambda item: item[0])
    return {
        "first": dict(first),
        "second": dict(second),
        "relations": list(relations),
        "scores": {name: scores[name] for name in relations},
        "quality": quality,
    }


def terms_in(text: str, terms: Sequence[str]) -> list[str]:
    padded = f" {normalized(text)} "
    return [term for term in terms if f" {normalized(term)} " in padded]


def first_term(text: str, terms: Sequence[str]) -> str | None:
    values = terms_in(text, terms)
    return values[0] if values else None


def associated_term(text: str, objects: Sequence[str], terms: Sequence[str]) -> tuple[str, str] | None:
    clean = normalized(text)
    for object_name in objects:
        object_key = normalized(object_name)
        matches = []
        for term in terms:
            term_key = normalized(term)
            if re.search(rf"\b{re.escape(term_key)}(?:\s+[a-z]+){{0,2}}\s+{re.escape(object_key)}\b", clean):
                matches.append(term)
        if len(matches) == 1:
            return object_name, matches[0]
    return None


def material_claim(text: str, objects: Sequence[str]) -> tuple[str, str] | None:
    matches = []
    for answer, aliases in MATERIALS.items():
        match = associated_term(text, objects, aliases)
        if match is not None:
            matches.append((match[0], answer))
    return matches[0] if len(matches) == 1 else None


def shape_claim(text: str, objects: Sequence[str]) -> tuple[str, str, str] | None:
    matches = []
    for answer, aliases in SHAPES.items():
        match = associated_term(text, objects, aliases)
        if match is not None:
            matches.append((match[0], answer, "spright_explicit_caption_claim"))
    if len(matches) == 1:
        return matches[0]
    prior_matches = [
        (object_name, SHAPE_PRIORS[object_name], "conservative_category_shape_prior")
        for object_name in objects if object_name in SHAPE_PRIORS
    ]
    return prior_matches[0] if len(prior_matches) == 1 else None


def build_record(
    sample: Mapping[str, Any],
    image: Mapping[str, Any],
    grouped: Mapping[str, Sequence[Mapping[str, Any]]],
    pair: Mapping[str, Any] | None,
    categories: Sequence[str],
    *,
    seed: int,
) -> dict[str, Any] | None:
    object_names = sorted(grouped, key=lambda name: (-len(grouped[name]), name))
    if not object_names:
        return None
    if pair is not None:
        primary = [str(pair["first"]["phrase"]), str(pair["second"]["phrase"])]
    else:
        primary = object_names[:2]
    key = f"{seed}:{sample['image_id']}:{sample['shard']}:{sample['tar_stem']}"
    questions: list[dict[str, Any]] = []
    prompt_claims: list[str] = []

    if pair is not None:
        relation_phrase = " and ".join(RELATION_TEXT[value] for value in pair["relations"])
        prompt_claims.append(f"A {primary[0]} is {relation_phrase} a {primary[1]}.")
        for relation in pair["relations"]:
            choices = list(RELATION_AXIS[relation])
            questions.append(qa(
                "spatial", relation,
                f"Where is the {primary[0]} relative to the {primary[1]}?",
                relation, choices, choices.index(relation), "coco_box_geometry",
            ))
    elif len(primary) == 2:
        prompt_claims.append(f"A photo containing a {primary[0]} and a {primary[1]}.")
    else:
        prompt_claims.append(f"A photo containing a {primary[0]}.")

    object_answer = primary[0]
    absent_categories = [name for name in categories if name not in grouped]
    object_choices, target = deterministic_choices(
        object_answer, [object_answer, *absent_categories], key=key + ":object"
    )
    questions.append(qa(
        "object", object_answer, "Which object is present in the image?",
        object_answer, object_choices, target, "coco_instance_category",
    ))

    count_candidates = {
        name: len(values) for name, values in grouped.items() if 1 <= len(values) <= 4
    }
    if count_candidates:
        count_object = max(count_candidates, key=lambda name: (count_candidates[name], name))
        count = count_candidates[count_object]
        prompt_claims.append(
            f"There {'is' if count == 1 else 'are'} {NUMBER_WORDS[count - 1]} "
            f"{count_object if count == 1 else plural(count_object)}."
        )
        questions.append(qa(
            "counting", str(count), f"How many {plural(count_object)} are present?",
            NUMBER_WORDS[count - 1], NUMBER_WORDS, count - 1,
            "coco_visible_instance_count",
        ))

    present_set = set(object_names)
    for dimension, vocabulary in (("animal/human", ANIMAL_HUMAN), ("food", FOOD)):
        answer = next((name for name in primary + object_names if name in vocabulary), None)
        if answer is None:
            continue
        distractors = sorted(vocabulary - present_set)
        choices, target = deterministic_choices(
            answer, [answer, *distractors], key=key + ":" + dimension
        )
        question = (
            "Which living subject is present in the image?"
            if dimension == "animal/human" else "Which food is present in the image?"
        )
        questions.append(qa(
            dimension, answer, question, answer, choices, target,
            "coco_instance_category",
        ))

    if pair is not None:
        first_area = float(pair["first"]["box"][2] - pair["first"]["box"][0]) * float(
            pair["first"]["box"][3] - pair["first"]["box"][1]
        )
        second_area = float(pair["second"]["box"][2] - pair["second"]["box"][0]) * float(
            pair["second"]["box"][3] - pair["second"]["box"][1]
        )
        if max(first_area, second_area) / max(min(first_area, second_area), 1e-9) >= 1.4:
            larger = primary[0] if first_area > second_area else primary[1]
            smaller = primary[1] if larger == primary[0] else primary[0]
            prompt_claims.append(f"The {larger} appears larger than the {smaller}.")
            questions.append(qa(
                "attribute", "relative size",
                f"Which appears larger, the {primary[0]} or the {primary[1]}?",
                larger, primary, primary.index(larger), "coco_box_relative_area",
            ))

    source_text = f"{sample['original_caption']} {sample['spatial_caption']}"
    present_activities = terms_in(source_text, ACTIVITIES)
    if present_activities:
        activity = present_activities[0]
        choices, target = deterministic_choices(
            activity, [activity, *[x for x in ACTIVITIES if x not in present_activities]],
            key=key + ":activity",
        )
        prompt_claims.append(f"Someone or something is {activity}.")
        questions.append(qa(
            "activity", activity, "What activity is taking place?", activity,
            choices, target, "spright_explicit_caption_claim",
        ))

    present_locations = terms_in(source_text, LOCATIONS)
    location = present_locations[0] if present_locations else None
    if location is not None:
        choices, target = deterministic_choices(
            location, [location, *[x for x in LOCATIONS if x not in present_locations]],
            key=key + ":location",
        )
        prompt_claims.append(f"The scene is at or in a {location}.")
        questions.append(qa(
            "location", location, "Where does the scene take place?", location,
            choices, target, "spright_explicit_caption_claim",
        ))

    color_match = associated_term(source_text, object_names, COLORS)
    if color_match is not None:
        object_name, color = color_match
        choices, target = deterministic_choices(color, COLORS, key=key + ":color")
        prompt_claims.append(f"The {object_name} is {color}.")
        questions.append(qa(
            "color", color, f"What color is the {object_name}?", color,
            choices, target, "spright_explicit_caption_claim",
        ))

    material = material_claim(source_text, object_names)
    if material is not None:
        object_name, answer = material
        choices, target = deterministic_choices(answer, sorted(MATERIALS), key=key + ":material")
        prompt_claims.append(f"The {object_name} is made of {answer}.")
        questions.append(qa(
            "material", answer, f"What material is the {object_name} made of?",
            answer, choices, target, "spright_explicit_caption_claim",
        ))

    shape = shape_claim(source_text, object_names)
    if shape is not None:
        object_name, answer, source = shape
        choices, target = deterministic_choices(answer, sorted(SHAPES), key=key + ":shape")
        prompt_claims.append(f"The {object_name} has a {answer} shape.")
        questions.append(qa(
            "shape", answer, f"What shape is the {object_name}?", answer,
            choices, target, source,
        ))

    other_answer: str | None = None
    other_choices: Sequence[str] = ("indoors", "outdoors")
    if location in INDOOR_LOCATIONS:
        other_answer = "indoors"
    elif location in OUTDOOR_LOCATIONS:
        other_answer = "outdoors"
    if other_answer is not None:
        prompt_claims.append(f"This is an {other_answer} scene.")
        questions.append(qa(
            "other", "scene type", "Is this scene indoors or outdoors?",
            other_answer, other_choices, other_choices.index(other_answer),
            "explicit_location_scene_type",
        ))
    else:
        weather = next(
            (answer for answer, aliases in WEATHER.items() if first_term(source_text, aliases)),
            None,
        )
        if weather is not None:
            choices, target = deterministic_choices(weather, sorted(WEATHER), key=key + ":weather")
            prompt_claims.append(f"The weather appears {weather}.")
            questions.append(qa(
                "other", "weather", "What is the weather like?", weather,
                choices, target, "spright_explicit_caption_claim",
            ))

    dimensions = sorted({question["element_type"] for question in questions})
    if len(dimensions) < 2:
        return None
    caption = " ".join(dict.fromkeys(prompt_claims))
    return {
        "caption": caption,
        "coco2017_split": image["coco2017_split"],
        "dimensions": dimensions,
        "image_id": int(sample["image_id"]),
        "image_origin": "SPRIGHT-COCO",
        "qa_pairs": questions,
        "relation_count": len(pair["relations"]) if pair is not None else 0,
        "relations": list(pair["relations"]) if pair is not None else [],
        "sample_id": f"spright_{sample['image_id']}_{sample['shard']}_{sample['tar_stem']}",
        "shard": sample["shard"],
        "source_original_caption": sample["original_caption"],
        "source_spatial_caption": sample["spatial_caption"],
        "tar_stem": sample["tar_stem"],
    }


def audit_rows(
    train: Sequence[Mapping[str, Any]],
    validation: Sequence[Mapping[str, Any]],
    forbidden_ids: set[int],
    forbidden_captions: set[str],
) -> dict[str, Any]:
    all_rows = [*train, *validation]
    dimensions = collections.Counter(
        question["element_type"] for row in all_rows for question in row["qa_pairs"]
    )
    sources = collections.Counter(
        question["source"] for row in all_rows for question in row["qa_pairs"]
    )
    relations = collections.Counter(
        relation for row in all_rows for relation in row.get("relations", ())
    )
    train_ids = {int(row["image_id"]) for row in train}
    validation_ids = {int(row["image_id"]) for row in validation}
    target_errors = sum(
        int(question["choices"][question["target_index"]] != question["answer"])
        for row in all_rows for question in row["qa_pairs"]
    )
    return {
        "caption_overlap_with_forbidden_evaluation": sum(
            normalized(row["caption"]) in forbidden_captions for row in all_rows
        ),
        "dimension_counts": dict(sorted(dimensions.items())),
        "dimension_count": len(dimensions),
        "image_overlap_with_official_tifa": len(
            {int(row["image_id"]) for row in all_rows} & forbidden_ids
        ),
        "qa_source_counts": dict(sorted(sources.items())),
        "question_count": sum(len(row["qa_pairs"]) for row in all_rows),
        "relation_counts": dict(sorted(relations.items())),
        "spatial_rows": sum("spatial" in row["dimensions"] for row in all_rows),
        "split_image_overlap": len(train_ids & validation_ids),
        "splits": {
            "train": {
                "rows": len(train),
                "unique_images": len(train_ids),
                "questions": sum(len(row["qa_pairs"]) for row in train),
            },
            "validation": {
                "rows": len(validation),
                "unique_images": len(validation_ids),
                "questions": sum(len(row["qa_pairs"]) for row in validation),
            },
        },
        "target_errors": target_errors,
        "unique_images": len(train_ids | validation_ids),
    }


def main() -> int:
    args = parse_args()
    if not 0.0 < args.validation_fraction < 0.5:
        raise ValueError("validation-fraction must be in (0, 0.5)")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    forbidden_ids, forbidden_captions, tifa_stats = load_forbidden(args.tifa_source)
    samples, spright_stats = load_spright(args.spright_root)
    images, annotations, categories_by_id = load_coco(
        args.coco_annotations_zip, {int(row["image_id"]) for row in samples}
    )
    category_names = sorted(set(categories_by_id.values()))

    counters: collections.Counter[str] = collections.Counter()
    train: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    for index, sample in enumerate(samples, 1):
        image_id = int(sample["image_id"])
        if image_id in seen_ids:
            counters["duplicate_image_id"] += 1
            continue
        seen_ids.add(image_id)
        if image_id in forbidden_ids:
            counters["excluded_tifa_image"] += 1
            continue
        if (
            normalized(sample["original_caption"]) in forbidden_captions
            or normalized(sample["spatial_caption"]) in forbidden_captions
        ):
            counters["excluded_forbidden_caption"] += 1
            continue
        image = images.get(image_id)
        if image is None:
            counters["missing_coco_image"] += 1
            continue
        grouped = visible_objects(
            image,
            annotations.get(image_id, ()),
            categories_by_id,
            min_visible_fraction=args.min_visible_fraction,
            min_box_area=args.min_box_area,
        )
        pair = best_spatial_pair(
            grouped,
            max_box_iou=args.max_box_iou,
            max_min_box_overlap=args.max_min_box_overlap,
            position_threshold=args.position_threshold,
            relation_threshold=args.relation_threshold,
        )
        row = build_record(sample, image, grouped, pair, category_names, seed=args.seed)
        if row is None:
            counters["insufficient_dimensions"] += 1
            continue
        if normalized(row["caption"]) in forbidden_captions:
            counters["excluded_generated_forbidden_caption"] += 1
            continue
        destination = validation if stable_validation(
            image_id, args.seed, args.validation_fraction
        ) else train
        destination.append(row)
        counters["spatial_rows" if pair is not None else "general_rows"] += 1
        if index % 5000 == 0:
            print(
                f"BUILD {index}/{len(samples)} train={len(train)} "
                f"validation={len(validation)} spatial={counters['spatial_rows']}",
                flush=True,
            )

    random.Random(args.seed).shuffle(train)
    random.Random(args.seed + 1).shuffle(validation)
    audit = audit_rows(train, validation, forbidden_ids, forbidden_captions)
    errors = []
    if len(train) < args.min_rows:
        errors.append(f"train rows {len(train)} < {args.min_rows}")
    if audit["spatial_rows"] < args.min_spatial_rows:
        errors.append(f"spatial rows {audit['spatial_rows']} < {args.min_spatial_rows}")
    if set(audit["dimension_counts"]) != set(DIMENSIONS):
        errors.append("not all official TIFA dimensions are present")
    for dimension in ("material", "shape", "other"):
        count = int(audit["dimension_counts"].get(dimension, 0))
        if count < args.min_rare_questions:
            errors.append(f"{dimension} questions {count} < {args.min_rare_questions}")
    for name in (
        "caption_overlap_with_forbidden_evaluation",
        "image_overlap_with_official_tifa",
        "split_image_overlap",
        "target_errors",
    ):
        if audit[name] != 0:
            errors.append(f"{name}={audit[name]}")

    train_path = args.output_dir / "train.jsonl"
    validation_path = args.output_dir / "validation.jsonl"
    write_jsonl(train_path, train)
    write_jsonl(validation_path, validation)
    audit.update({
        "build_counters": dict(sorted(counters.items())),
        "elapsed_seconds": time.time() - started,
        "errors": errors,
        "inputs": {
            "coco_annotations_zip": str(args.coco_annotations_zip.resolve()),
            "spright_root": str(args.spright_root.resolve()),
            "tifa_source": str(args.tifa_source.resolve()),
        },
        "protocol": "external-spright-coco-tifa12-v1",
        "seed": args.seed,
        "spright": spright_stats,
        "status": "ready" if not errors else "failed",
        "tifa_forbidden": tifa_stats,
        "validation_fraction": args.validation_fraction,
    })
    audit_path = args.output_dir / "audit.json"
    atomic_json(audit_path, audit)
    manifest = {
        "audit": str(audit_path.resolve()),
        "audit_sha256": sha256(audit_path),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dimensions": list(DIMENSIONS),
        "license": "Intel Research Use License v6; research-only",
        "protocol": audit["protocol"],
        "shard_root": str(args.spright_root.resolve()),
        "status": audit["status"],
        "train": str(train_path.resolve()),
        "train_sha256": sha256(train_path),
        "validation": str(validation_path.resolve()),
        "validation_sha256": sha256(validation_path),
    }
    atomic_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps({"audit": audit, "manifest": manifest}, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
