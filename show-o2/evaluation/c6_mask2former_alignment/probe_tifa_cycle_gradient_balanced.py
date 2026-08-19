#!/usr/bin/env python3
"""Run the full TIFA gradient probe with deterministic option reshuffling."""

from __future__ import annotations

from evaluation.c6_mask2former_alignment import probe_tifa_cycle_gradient as probe
from evaluation.c6_mask2former_alignment.tifa_cycle_sampling import (
    shuffle_choice_positions,
)


ORIGINAL_SELECT = probe.select_multidimensional_questions


def balanced_select(qa_pairs, *, max_questions, seed):
    selected = ORIGINAL_SELECT(
        qa_pairs, max_questions=max_questions, seed=seed
    )
    return tuple(
        shuffle_choice_positions(qa, seed=seed + 104_729 * (index + 1))
        for index, qa in enumerate(selected)
    )


if __name__ == "__main__":
    probe.select_multidimensional_questions = balanced_select
    raise SystemExit(probe.main())
