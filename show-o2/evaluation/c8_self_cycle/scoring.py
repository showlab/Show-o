#!/usr/bin/env python3
"""Deterministic question typing and answer scoring for self-cycle evaluation."""

from __future__ import annotations

import re
import string
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable


YES_ALIASES = {"yes", "y", "true", "present"}
NO_ALIASES = {"no", "n", "false", "absent"}
ARTICLES = {"a", "an", "the"}
NUMBER_WORDS = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
    "thirteen": "13",
    "fourteen": "14",
    "fifteen": "15",
    "sixteen": "16",
    "seventeen": "17",
    "eighteen": "18",
    "nineteen": "19",
    "twenty": "20",
}
TOKEN_ALIASES = {
    "grey": "gray",
    "wooden": "wood",
    "metallic": "metal",
}
OPTION_MARKER = re.compile(r"(?<![A-Za-z0-9])([A-Da-d])[\.)\]:]\s*")
OPTION_ANSWER = re.compile(r"(?:^|\b)([A-Da-d])(?:\b|[\.)\]:])")


@dataclass(frozen=True)
class QuestionSpec:
    mode: str
    reference: str
    candidates: tuple[str, ...]
    options: dict[str, str]


def normalize_answer(value: object) -> str:
    """Normalize short English VQA answers without semantic model calls."""
    text = unicodedata.normalize("NFKC", str(value or "")).lower().strip()
    text = text.replace("-", " ").replace("/", " ")
    table = str.maketrans({character: " " for character in string.punctuation})
    tokens = text.translate(table).split()
    normalized = []
    for token in tokens:
        if token in ARTICLES:
            continue
        token = NUMBER_WORDS.get(token, token)
        token = TOKEN_ALIASES.get(token, token)
        normalized.append(token)
    return " ".join(normalized)


def parse_options(question: str) -> dict[str, str]:
    matches = list(OPTION_MARKER.finditer(question))
    if len(matches) < 2:
        return {}
    options: dict[str, str] = {}
    for index, match in enumerate(matches):
        label = match.group(1).upper()
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(question)
        value = question[start:end].strip(" \t\r\n,;")
        if label in options or not value:
            return {}
        options[label] = value
    return options


def canonical_yes_no(value: object) -> str | None:
    normalized = normalize_answer(value)
    first = normalized.split()[0] if normalized else ""
    if first in YES_ALIASES:
        return "yes"
    if first in NO_ALIASES:
        return "no"
    return None


def canonical_option(value: object, options: dict[str, str]) -> str | None:
    raw = str(value or "").strip()
    match = OPTION_ANSWER.search(raw)
    if match and match.group(1).upper() in options:
        return match.group(1).upper()
    normalized = normalize_answer(raw)
    for label, option_text in options.items():
        if normalized and normalized == normalize_answer(option_text):
            return label
    return None


def classify_question(question: str, reference: object) -> QuestionSpec:
    options = parse_options(question)
    if options:
        canonical = canonical_option(reference, options)
        if canonical is not None:
            return QuestionSpec(
                mode="multiple_choice",
                reference=canonical,
                candidates=tuple(options),
                options=options,
            )
    yes_no = canonical_yes_no(reference)
    if yes_no is not None:
        return QuestionSpec(
            mode="yes_no",
            reference=yes_no,
            candidates=("yes", "no"),
            options={},
        )
    return QuestionSpec(
        mode="short_answer",
        reference=normalize_answer(reference),
        candidates=(),
        options={},
    )


def token_f1(prediction: object, reference: object) -> float:
    predicted_tokens = normalize_answer(prediction).split()
    reference_tokens = normalize_answer(reference).split()
    if not predicted_tokens or not reference_tokens:
        return float(predicted_tokens == reference_tokens)
    common = Counter(predicted_tokens) & Counter(reference_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted_tokens)
    recall = overlap / len(reference_tokens)
    return 2.0 * precision * recall / (precision + recall)


def score_answer(prediction: object, spec: QuestionSpec) -> dict[str, Any]:
    if spec.mode == "multiple_choice":
        canonical = canonical_option(prediction, spec.options)
        prediction_normalized = canonical or normalize_answer(prediction)
        correct = canonical == spec.reference
        f1 = float(correct)
    elif spec.mode == "yes_no":
        canonical = canonical_yes_no(prediction)
        prediction_normalized = canonical or normalize_answer(prediction)
        correct = canonical == spec.reference
        f1 = float(correct)
    else:
        prediction_normalized = normalize_answer(prediction)
        correct = bool(spec.reference) and prediction_normalized == spec.reference
        f1 = token_f1(prediction_normalized, spec.reference)
    return {
        "prediction_normalized": prediction_normalized,
        "reference_normalized": spec.reference,
        "correct": bool(correct),
        "token_f1": float(f1),
    }


def aggregate_results(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    records = list(rows)
    by_mode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_prompt: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_mode[str(row["answer_mode"])].append(row)
        by_type[str(row.get("type", "unknown"))].append(row)
        by_prompt[(int(row["record_index"]), int(row["sample_index"]))].append(row)

    def summarize(items: list[dict[str, Any]]) -> dict[str, Any]:
        total = len(items)
        return {
            "questions": total,
            "correct": sum(bool(item["correct"]) for item in items),
            "accuracy": (
                sum(bool(item["correct"]) for item in items) / total if total else 0.0
            ),
            "mean_token_f1": (
                sum(float(item["token_f1"]) for item in items) / total
                if total else 0.0
            ),
        }

    prompt_soft = [
        sum(bool(item["correct"]) for item in items) / len(items)
        for items in by_prompt.values()
        if items
    ]
    prompt_hard = [float(value == 1.0) for value in prompt_soft]
    return {
        "questions": summarize(records),
        "prompts": {
            "count": len(prompt_soft),
            "soft_score": sum(prompt_soft) / len(prompt_soft) if prompt_soft else 0.0,
            "hard_score": sum(prompt_hard) / len(prompt_hard) if prompt_hard else 0.0,
        },
        "by_answer_mode": {
            key: summarize(value) for key, value in sorted(by_mode.items())
        },
        "by_type": {key: summarize(value) for key, value in sorted(by_type.items())},
    }
