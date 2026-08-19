"""Pure VISOR geometry and aggregation helpers."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Sequence


RELATIONS = ("to the left of", "to the right of", "above", "below")


def relation_holds(
    first: Sequence[float], second: Sequence[float], relation: str
) -> bool:
    """Apply the official VISOR centroid-sign rule."""
    if len(first) != 2 or len(second) != 2:
        raise ValueError("centroids must have two coordinates")
    if relation not in RELATIONS:
        raise ValueError(f"unsupported VISOR relation: {relation!r}")
    dx = float(first[0]) - float(second[0])
    dy = float(first[1]) - float(second[1])
    return {
        "to the left of": dx < 0.0,
        "to the right of": dx > 0.0,
        "above": dy < 0.0,
        "below": dy > 0.0,
    }[relation]


def score_detections(
    labels: Sequence[str],
    centroids: Sequence[Sequence[float]],
    object_a: str,
    object_b: str,
    relation: str,
) -> dict[str, Any]:
    """Match the official any-detected-pair VISOR decision."""
    if len(labels) != len(centroids):
        raise ValueError("labels and centroids must have matching lengths")
    indices_a = [index for index, label in enumerate(labels) if label == object_a]
    indices_b = [index for index, label in enumerate(labels) if label == object_b]
    detected_a = bool(indices_a)
    detected_b = bool(indices_b)
    spatial_correct = any(
        relation_holds(centroids[index_a], centroids[index_b], relation)
        for index_a in indices_a
        for index_b in indices_b
    )
    return {
        "detected_a": detected_a,
        "detected_b": detected_b,
        "detected_both": detected_a and detected_b,
        "recall": int(detected_a) + int(detected_b),
        "spatial_correct": spatial_correct,
        "joint_success": detected_a and detected_b and spatial_correct,
    }


def _aggregate_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    detected_a = sum(bool(row["detected_a"]) for row in rows)
    detected_b = sum(bool(row["detected_b"]) for row in rows)
    detected_both = sum(bool(row["detected_both"]) for row in rows)
    successes = sum(bool(row["joint_success"]) for row in rows)
    return {
        "images": count,
        "object_accuracy_a": 100.0 * detected_a / count if count else 0.0,
        "object_accuracy_b": 100.0 * detected_b / count if count else 0.0,
        "object_accuracy_both": 100.0 * detected_both / count if count else 0.0,
        "visor_cond": 100.0 * successes / detected_both if detected_both else 0.0,
        "visor_uncond": 100.0 * successes / count if count else 0.0,
        "detected_both": detected_both,
        "joint_successes": successes,
    }


def aggregate_visor(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate official metrics plus relation and VISOR-at-least-N splits."""
    records = list(rows)
    if not records:
        raise ValueError("VISOR aggregation requires at least one image result")
    keys = [(int(row["unique_id"]), int(row["sample_index"])) for row in records]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate unique_id/sample_index in VISOR results")
    if any(str(row["relation"]) not in RELATIONS for row in records):
        raise ValueError("VISOR result contains an unsupported relation")

    by_relation: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_prompt: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_relation[str(row["relation"])].append(row)
        by_prompt[int(row["unique_id"])].append(row)

    sample_counts = {len(items) for items in by_prompt.values()}
    if len(sample_counts) != 1:
        raise ValueError("every VISOR prompt must have the same number of samples")
    samples_per_prompt = next(iter(sample_counts))
    at_least = {
        str(required): 100.0
        * sum(
            sum(bool(row["joint_success"]) for row in items) >= required
            for items in by_prompt.values()
        )
        / len(by_prompt)
        for required in range(1, samples_per_prompt + 1)
    }
    return {
        **_aggregate_rows(records),
        "prompts": len(by_prompt),
        "samples_per_prompt": samples_per_prompt,
        "visor_at_least_n": at_least,
        "by_relation": {
            relation: _aggregate_rows(by_relation.get(relation, []))
            for relation in RELATIONS
        },
    }

