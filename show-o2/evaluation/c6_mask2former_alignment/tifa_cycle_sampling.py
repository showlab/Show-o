"""Leak-free, position-balanced sampling for TIFA self-cycle questions."""

from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from typing import Any, Mapping, Sequence

from evaluation.c6_mask2former_alignment.tifa_cycle import (
    select_multidimensional_questions,
)


def normalized(value: Any) -> str:
    return " ".join(str(value).strip().casefold().split()).strip(" .,!?:;\"'")


def cycle_question_seed(base_seed: int, sample_id: str, cycle_step: int) -> int:
    payload = f"{base_seed}\0{sample_id}\0{cycle_step}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def filter_caption_copy_questions(
    caption: str, qa_pairs: Sequence[Mapping[str, Any]]
) -> tuple[Mapping[str, Any], ...]:
    caption_key = normalized(caption)
    return tuple(
        qa
        for qa in qa_pairs
        if len(caption_key) < 20
        or caption_key not in normalized(qa.get("question", ""))
    )


def shuffle_choice_positions(
    qa: Mapping[str, Any], *, seed: int
) -> dict[str, Any]:
    choices = list(qa["choices"])
    target = int(qa["target_index"])
    if not 0 <= target < len(choices):
        raise ValueError(f"invalid target_index={target} for {len(choices)} choices")
    permutation = list(range(len(choices)))
    random.Random(seed).shuffle(permutation)
    shuffled = dict(qa)
    shuffled["choices"] = [choices[index] for index in permutation]
    shuffled["target_index"] = permutation.index(target)
    shuffled["source_target_index"] = target
    shuffled["choice_permutation"] = permutation
    return shuffled


def prepare_cycle_questions(
    row: Mapping[str, Any],
    *,
    max_questions: int,
    base_seed: int,
    cycle_step: int,
) -> tuple[dict[str, Any], ...]:
    """Select dimensions first and deterministically reshuffle every answer slot."""
    sample_id = str(row["sample_id"])
    seed = cycle_question_seed(base_seed, sample_id, cycle_step)
    filtered = filter_caption_copy_questions(str(row["caption"]), row["qa_pairs"])
    if not filtered:
        filtered = tuple(row["qa_pairs"])
    selected = select_multidimensional_questions(
        filtered, max_questions=max_questions, seed=seed
    )
    return tuple(
        shuffle_choice_positions(qa, seed=seed + 104_729 * (index + 1))
        for index, qa in enumerate(selected)
    )


def prepare_spatial_cycle_questions(
    row: Mapping[str, Any],
    *,
    max_questions: int,
    min_non_spatial_dimensions: int,
    base_seed: int,
    cycle_step: int,
) -> tuple[dict[str, Any], ...]:
    """Select one or two orthogonal spatial questions plus non-spatial dimensions."""
    if max_questions < 1 + min_non_spatial_dimensions:
        raise ValueError(
            "max_questions cannot satisfy the requested non-spatial dimensions"
        )
    if min_non_spatial_dimensions < 1:
        raise ValueError("min_non_spatial_dimensions must be positive")

    seed = cycle_question_seed(
        base_seed, str(row["sample_id"]), cycle_step
    )
    filtered = filter_caption_copy_questions(str(row["caption"]), row["qa_pairs"])
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for qa in filtered:
        groups[str(qa.get("element_type", "unknown")).strip().casefold()].append(qa)
    if not groups.get("spatial"):
        return prepare_cycle_questions(
            row,
            max_questions=max_questions,
            base_seed=base_seed,
            cycle_step=cycle_step,
        )

    rng = random.Random(seed)
    spatial_candidates = list(groups.pop("spatial"))
    rng.shuffle(spatial_candidates)
    selected: list[Mapping[str, Any]] = [spatial_candidates.pop()]
    axis_by_relation = {
        "left": "horizontal",
        "right": "horizontal",
        "above": "vertical",
        "below": "vertical",
    }
    first_axis = axis_by_relation.get(
        normalized(selected[0].get("element", ""))
    )
    if (
        first_axis is not None
        and len(selected) + min_non_spatial_dimensions < max_questions
    ):
        for index, candidate in enumerate(spatial_candidates):
            candidate_axis = axis_by_relation.get(
                normalized(candidate.get("element", ""))
            )
            if candidate_axis is not None and candidate_axis != first_axis:
                selected.append(spatial_candidates.pop(index))
                break

    dimensions = sorted(groups)
    rng.shuffle(dimensions)
    # Caption-copy filtering can remove a dimension; use all remaining groups.
    min_non_spatial_dimensions = min(min_non_spatial_dimensions, len(dimensions))
    remaining: list[Mapping[str, Any]] = list(spatial_candidates)
    for dimension in dimensions:
        candidates = list(groups[dimension])
        rng.shuffle(candidates)
        selected.append(candidates.pop())
        remaining.extend(candidates)
        if len(selected) == max_questions:
            break
    if len(selected) < max_questions:
        rng.shuffle(remaining)
        selected.extend(remaining[: max_questions - len(selected)])

    return tuple(
        shuffle_choice_positions(qa, seed=seed + 104_729 * (index + 1))
        for index, qa in enumerate(selected)
    )
