#!/usr/bin/env python3
"""Unit tests for caption-cycle lexical metrics."""

from __future__ import annotations

import unittest

from evaluation.c8_self_cycle.caption_metrics import (
    lexical_scores,
    rouge_l_f1,
    token_prf,
    tokens,
)


class CaptionMetricsTest(unittest.TestCase):
    def test_tokens(self) -> None:
        self.assertEqual(tokens("A dark-blue vase!"), ["a", "dark", "blue", "vase"])

    def test_token_prf(self) -> None:
        values = token_prf("blue vase", "a dark blue vase")
        self.assertEqual(values["precision"], 1.0)
        self.assertEqual(values["recall"], 0.5)
        self.assertAlmostEqual(values["f1"], 2.0 / 3.0)

    def test_rouge_l(self) -> None:
        self.assertEqual(rouge_l_f1("red vase", "red vase"), 1.0)
        self.assertLess(rouge_l_f1("vase red", "red vase"), 1.0)

    def test_lexical_scores(self) -> None:
        values = lexical_scores("a wooden table", "a wood table")
        self.assertEqual(set(values), {
            "token_precision", "token_recall", "token_f1", "rouge_l_f1"
        })


if __name__ == "__main__":
    unittest.main()
