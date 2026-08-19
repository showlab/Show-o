#!/usr/bin/env python3
"""Build a deterministic GenEval manifest for condition-localization probes."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


DEFAULT_TAGS = (
    "single_object",
    "two_object",
    "counting",
    "colors",
    "position",
    "color_attr",
)
COLORS = ("red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "black", "white")
NUMBER_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}
OPPOSITE_POSITION = {"left of": "right of", "right of": "left of", "above": "below", "below": "above"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--base-prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-tag", type=int, default=0,
                        help="Unique prompts per category; 0 keeps every prompt")
    parser.add_argument("--images-per-prompt", type=int, default=0,
                        help="Generated images per prompt; 0 keeps every image")
    parser.add_argument("--outcomes", choices=("all", "correct", "incorrect"), default="all")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--tags", nargs="+", default=list(DEFAULT_TAGS))
    return parser.parse_args()


def prompt_id(row: dict) -> str:
    return Path(row["filename"]).parent.parent.name


def choose_other(value: str, choices: tuple[str, ...], salt: int) -> str:
    candidates = [candidate for candidate in choices if candidate != value]
    return candidates[salt % len(candidates)]


def make_counterfactual(row: dict, peer: dict, index: int) -> tuple[str, dict]:
    tag = row["tag"]
    include = row["include"]
    if tag == "single_object":
        source = include[0]["class"]
        target = peer["include"][0]["class"]
        return f"a photo of a {target}", {"operation": "replace_object", "from": source, "to": target}

    if tag == "two_object":
        first, second = include[0]["class"], include[1]["class"]
        peer_classes = [item["class"] for item in peer["include"]]
        target = next((name for name in peer_classes if name not in {first, second}), peer_classes[-1])
        return (
            f"a photo of a {first} and a {target}",
            {"operation": "replace_second_object", "from": second, "to": target},
        )

    if tag == "counting":
        count = int(include[0]["count"])
        excluded = row.get("exclude") or []
        alternate = int(excluded[0]["count"]) if excluded and "count" in excluded[0] else (3 if count == 2 else 2)
        old_word = NUMBER_WORDS[count]
        new_word = NUMBER_WORDS[alternate]
        prompt = row["prompt"].replace(old_word, new_word, 1)
        return prompt, {"operation": "replace_count", "from": count, "to": alternate}

    if tag == "colors":
        source = include[0]["color"]
        target = choose_other(source, COLORS, index)
        prompt = f"a photo of a {target} {include[0]['class']}"
        return prompt, {"operation": "replace_color", "from": source, "to": target}

    if tag == "position":
        positioned = next(item for item in include if "position" in item)
        relation, reference_index = positioned["position"]
        target = OPPOSITE_POSITION[relation]
        reference = include[int(reference_index)]["class"]
        prompt = f"a photo of a {positioned['class']} {target} a {reference}"
        return prompt, {"operation": "reverse_relation", "from": relation, "to": target}

    if tag == "color_attr":
        first, second = include[0], include[1]
        prompt = f"a photo of a {second['color']} {first['class']} and a {first['color']} {second['class']}"
        return (
            prompt,
            {
                "operation": "swap_color_assignment",
                "from": [first["color"], second["color"]],
                "to": [second["color"], first["color"]],
            },
        )

    raise ValueError(f"Unsupported tag: {tag}")


def main() -> None:
    args = parse_args()
    if args.per_tag < 0 or args.images_per_prompt < 0:
        raise ValueError("--per-tag and --images-per-prompt must be non-negative")
    if not args.results.is_file() or not args.base_prompts.is_file():
        raise FileNotFoundError("GenEval results or base prompt metadata is missing")

    base_rows = {}
    with args.base_prompts.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if line.strip():
                base_rows[f"{index:05d}"] = json.loads(line)

    candidates = defaultdict(lambda: defaultdict(list))
    with args.results.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            result = json.loads(line)
            if result.get("tag") not in args.tags:
                continue
            if args.outcomes == "correct" and not bool(result.get("correct")):
                continue
            if args.outcomes == "incorrect" and bool(result.get("correct")):
                continue
            identifier = prompt_id(result)
            if identifier not in base_rows:
                raise RuntimeError(f"No base prompt metadata for GenEval prompt {identifier}")
            image = Path(result["filename"])
            if not image.is_file():
                raise FileNotFoundError(f"Missing GenEval image at line {line_number}: {image}")
            candidates[result["tag"]][identifier].append(result)

    missing = [
        tag for tag in args.tags
        if not candidates[tag] or (args.per_tag > 0 and len(candidates[tag]) < args.per_tag)
    ]
    if missing:
        raise RuntimeError(f"Not enough matching, unique GenEval prompts for: {', '.join(missing)}")

    rng = random.Random(args.seed)
    selected_by_tag: dict[str, list[dict]] = {}
    for tag in args.tags:
        identifiers = sorted(candidates[tag])
        rng.shuffle(identifiers)
        if args.per_tag > 0:
            identifiers = identifiers[:args.per_tag]
        selected = []
        for identifier in identifiers:
            selected.append({
                "prompt_id": identifier,
                "results": sorted(candidates[tag][identifier], key=lambda row: row["filename"]),
                **base_rows[identifier],
            })
        selected_by_tag[tag] = selected

    prompt_rows = []
    max_prompts = max(len(rows) for rows in selected_by_tag.values())
    for offset in range(max_prompts):
        for tag in args.tags:
            if offset >= len(selected_by_tag[tag]):
                continue
            row = selected_by_tag[tag][offset]
            peer = selected_by_tag[tag][(offset + 1) % len(selected_by_tag[tag])]
            if len(selected_by_tag[tag]) == 1:
                peer_pool = [
                    {"prompt_id": identifier, **base_rows[identifier]}
                    for identifier in sorted(candidates[tag])
                    if identifier != row["prompt_id"]
                ]
                peer = peer_pool[0]
            counterfactual, edit = make_counterfactual(row, peer, offset)
            prompt_rows.append((row, tag, counterfactual, edit))

    output_rows = []
    for row, tag, counterfactual, edit in prompt_rows:
        results = row["results"][:]
        if args.images_per_prompt > 0 and len(results) > args.images_per_prompt:
            results = rng.sample(results, args.images_per_prompt)
            results.sort(key=lambda result: result["filename"])
        for result in results:
            output_rows.append({
                "index": len(output_rows),
                "sample_id": f"{row['prompt_id']}-{Path(result['filename']).stem}",
                "prompt_id": row["prompt_id"],
                "tag": tag,
                "correct": bool(result["correct"]),
                "reason": result.get("reason", ""),
                "image": str(Path(result["filename"]).resolve()),
                "matched_prompt": row["prompt"],
                "counterfactual_prompt": counterfactual,
                "counterfactual_edit": edit,
                "source_metadata": {key: row[key] for key in ("include", "exclude") if key in row},
            })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({
        "status": "ok",
        "output": str(args.output),
        "samples": len(output_rows),
        "unique_prompts": len({row["prompt_id"] for row in output_rows}),
        "correct_images": sum(bool(row["correct"]) for row in output_rows),
        "incorrect_images": sum(not bool(row["correct"]) for row in output_rows),
        "per_tag": {
            tag: {
                "prompts": len({row["prompt_id"] for row in output_rows if row["tag"] == tag}),
                "images": sum(row["tag"] == tag for row in output_rows),
            }
            for tag in args.tags
        },
    }, sort_keys=True))


if __name__ == "__main__":
    main()
