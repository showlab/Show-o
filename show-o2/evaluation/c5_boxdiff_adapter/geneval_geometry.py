"""Pure-Python geometry matching GenEval's relative-position evaluator."""

from __future__ import annotations

import math
from typing import Mapping, Sequence


POSITION_THRESHOLD = 0.1
RELATION_THRESHOLD = 0.5
RELATIONS = ("left", "right", "above", "below")


def box_geometry(box: Sequence[float]) -> tuple[float, float, float, float]:
    if len(box) != 4:
        raise ValueError(f"expected xyxy box, got {box!r}")
    x0, y0, x1, y1 = (float(value) for value in box)
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"invalid xyxy box {box!r}")
    return 0.5 * (x0 + x1), 0.5 * (y0 + y1), x1 - x0, y1 - y0


def geneval_relation_scores(
    box_a: Sequence[float],
    box_b: Sequence[float],
    *,
    position_threshold: float = POSITION_THRESHOLD,
) -> Mapping[str, float]:
    """Return GenEval's four directional scores for A relative to B.

    GenEval declares a relation present when its score is greater than 0.5.
    This is the scalar form of ``geneval/evaluation/evaluate_images.py``'s
    ``relative_position`` function.
    """
    ax, ay, aw, ah = box_geometry(box_a)
    bx, by, bw, bh = box_geometry(box_b)
    dx, dy = ax - bx, ay - by
    norm = math.hypot(dx, dy)
    if norm < 1e-12:
        return {relation: 0.0 for relation in RELATIONS}
    tolerance_x = float(position_threshold) * (aw + bw)
    tolerance_y = float(position_threshold) * (ah + bh)
    return {
        "left": max(-dx - tolerance_x, 0.0) / norm,
        "right": max(dx - tolerance_x, 0.0) / norm,
        "above": max(-dy - tolerance_y, 0.0) / norm,
        "below": max(dy - tolerance_y, 0.0) / norm,
    }


def geneval_relation_score(
    box_a: Sequence[float],
    box_b: Sequence[float],
    relation: str,
    *,
    position_threshold: float = POSITION_THRESHOLD,
) -> float:
    if relation not in RELATIONS:
        raise ValueError(f"unsupported relation {relation!r}")
    return geneval_relation_scores(
        box_a, box_b, position_threshold=position_threshold
    )[relation]
