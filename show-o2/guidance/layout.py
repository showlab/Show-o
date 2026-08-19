"""Prompt-to-layout parsing and tokenizer alignment for guided generation."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable


Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class ObjectConstraint:
    phrase: str
    box: Box
    char_span: tuple[int, int]
    token_indices: tuple[int, ...] = ()


@dataclass(frozen=True)
class LayoutSpec:
    prompt: str
    relation: str
    objects: tuple[ObjectConstraint, ...]

    def require_tokens(self) -> "LayoutSpec":
        missing = [item.phrase for item in self.objects if not item.token_indices]
        if missing:
            raise ValueError(f"Layout objects have no mapped tokens: {missing}")
        return self


_RELATIONS = (
    ("to the left of", "left"),
    ("on the left of", "left"),
    ("left of", "left"),
    ("to the right of", "right"),
    ("on the right of", "right"),
    ("right of", "right"),
    ("on top of", "above"),
    ("above", "above"),
    ("underneath", "below"),
    ("under", "below"),
    ("below", "below"),
)
_RELATION_PATTERN = "|".join(re.escape(alias) for alias, _ in _RELATIONS)
_POSITION_RE = re.compile(
    rf"(?P<first>.+?)\s+(?P<relation>{_RELATION_PATTERN})\s+(?P<second>.+?)(?:[.!?]|$)",
    flags=re.IGNORECASE,
)


def _clean_object(raw: str, absolute_start: int) -> tuple[str, tuple[int, int]]:
    prefix = re.compile(
        r"^\s*(?:(?:a|an|the)\s+)?(?:(?:photo|image|picture)\s+of\s+)?"
        r"(?:(?:a|an|the)\s+)?",
        flags=re.IGNORECASE,
    )
    suffix = re.compile(r"\s*(?:in the image|in the picture)\s*$", flags=re.IGNORECASE)
    start_match = prefix.match(raw)
    local_start = start_match.end() if start_match else 0
    cleaned = suffix.sub("", raw[local_start:]).strip()
    if not cleaned:
        raise ValueError(f"Could not extract an object phrase from {raw!r}")
    phrase_offset = raw.lower().find(cleaned.lower(), local_start)
    start = absolute_start + phrase_offset
    return cleaned, (start, start + len(cleaned))


def _relation_boxes(relation: str, margin: float, gap: float) -> tuple[Box, Box]:
    if not 0.0 <= margin < 0.25:
        raise ValueError("margin must be in [0, 0.25)")
    if not 0.0 <= gap < 0.5:
        raise ValueError("gap must be in [0, 0.5)")
    low = margin
    high = 1.0 - margin
    middle_low = 0.5 - gap * 0.5
    middle_high = 0.5 + gap * 0.5
    left = (low, low, middle_low, high)
    right = (middle_high, low, high, high)
    top = (low, low, high, middle_low)
    bottom = (low, middle_high, high, high)
    return {
        "left": (left, right),
        "right": (right, left),
        "above": (top, bottom),
        "below": (bottom, top),
    }[relation]


def parse_position_layout(prompt: str, margin: float = 0.08, gap: float = 0.10) -> LayoutSpec:
    match = _POSITION_RE.search(prompt)
    if match is None:
        raise ValueError(
            "Could not parse a spatial relation. Use a prompt containing left of, right of, "
            "above, or below, or provide --layout-json."
        )
    alias = match.group("relation").lower()
    relation = dict(_RELATIONS)[alias]
    first, first_span = _clean_object(match.group("first"), match.start("first"))
    second, second_span = _clean_object(match.group("second"), match.start("second"))
    first_box, second_box = _relation_boxes(relation, margin, gap)
    return LayoutSpec(
        prompt=prompt,
        relation=relation,
        objects=(
            ObjectConstraint(first, first_box, first_span),
            ObjectConstraint(second, second_box, second_span),
        ),
    )


def _validate_box(value: Iterable[Any]) -> Box:
    values = tuple(float(item) for item in value)
    if len(values) != 4:
        raise ValueError(f"A box must contain four coordinates, got {values}")
    x0, y0, x1, y1 = values
    if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
        raise ValueError(f"Box coordinates must be normalized and ordered, got {values}")
    return values


def layout_from_mapping(prompt: str, payload: dict[str, Any]) -> LayoutSpec:
    objects = []
    for item in payload.get("objects", ()):
        phrase = str(item["phrase"])
        match = re.search(re.escape(phrase), prompt, flags=re.IGNORECASE)
        if match is None:
            raise ValueError(f"Layout phrase {phrase!r} does not occur in the prompt")
        objects.append(
            ObjectConstraint(
                phrase=prompt[match.start():match.end()],
                box=_validate_box(item["box"]),
                char_span=(match.start(), match.end()),
            )
        )
    if not objects:
        raise ValueError("layout JSON must contain a non-empty objects list")
    return LayoutSpec(
        prompt=prompt,
        relation=str(payload.get("relation", "explicit")),
        objects=tuple(objects),
    )


def load_layout_json(prompt: str, value: str) -> LayoutSpec:
    candidate = Path(value)
    if candidate.is_file():
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    else:
        payload = json.loads(value)
    if not isinstance(payload, dict):
        raise TypeError("layout JSON must be an object")
    return layout_from_mapping(prompt, payload)


def map_layout_tokens(layout: LayoutSpec, tokenizer: Any, sequence_offset: int = 1) -> LayoutSpec:
    encoded = tokenizer(
        layout.prompt,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    offsets = encoded.get("offset_mapping")
    if offsets is None:
        raise TypeError("The tokenizer must support return_offsets_mapping")
    mapped = []
    for item in layout.objects:
        start, end = item.char_span
        indices = tuple(
            index + sequence_offset
            for index, (token_start, token_end) in enumerate(offsets)
            if token_end > start and token_start < end and token_end > token_start
        )
        if not indices:
            raise ValueError(
                f"Could not map object phrase {item.phrase!r} at span {item.char_span} to tokens"
            )
        mapped.append(replace(item, token_indices=indices))
    return replace(layout, objects=tuple(mapped)).require_tokens()


def layout_to_dict(layout: LayoutSpec) -> dict[str, Any]:
    return {
        "prompt": layout.prompt,
        "relation": layout.relation,
        "objects": [
            {
                "phrase": item.phrase,
                "box": list(item.box),
                "char_span": list(item.char_span),
                "token_indices": list(item.token_indices),
            }
            for item in layout.objects
        ],
    }
