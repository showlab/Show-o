#!/usr/bin/env python3
"""Deterministic lexical metrics for caption reconstruction cycles."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Any


TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")


def tokens(value: object) -> list[str]:
    text = unicodedata.normalize("NFKC", str(value or "")).lower()
    return TOKEN_PATTERN.findall(text)


def token_prf(prediction: object, reference: object) -> dict[str, float]:
    predicted = tokens(prediction)
    expected = tokens(reference)
    if not predicted or not expected:
        exact = float(predicted == expected)
        return {"precision": exact, "recall": exact, "f1": exact}
    overlap = sum((Counter(predicted) & Counter(expected)).values())
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )
    return {"precision": precision, "recall": recall, "f1": f1}


def rouge_l_f1(prediction: object, reference: object) -> float:
    predicted = tokens(prediction)
    expected = tokens(reference)
    if not predicted or not expected:
        return float(predicted == expected)
    previous = [0] * (len(expected) + 1)
    for left in predicted:
        current = [0]
        for index, right in enumerate(expected, 1):
            if left == right:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(current[-1], previous[index]))
        previous = current
    lcs = previous[-1]
    precision = lcs / len(predicted)
    recall = lcs / len(expected)
    return (
        2.0 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )


def lexical_scores(prediction: object, reference: object) -> dict[str, Any]:
    values = token_prf(prediction, reference)
    return {
        "token_precision": values["precision"],
        "token_recall": values["recall"],
        "token_f1": values["f1"],
        "rouge_l_f1": rouge_l_f1(prediction, reference),
    }
