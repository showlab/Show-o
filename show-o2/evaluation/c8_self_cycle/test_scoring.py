#!/usr/bin/env python3
"""Unit tests for deterministic self-cycle answer scoring."""

from __future__ import annotations

import unittest

from evaluation.c8_self_cycle.scoring import (
    aggregate_results,
    classify_question,
    normalize_answer,
    parse_options,
    score_answer,
    token_f1,
)


class ScoringTest(unittest.TestCase):
    def test_multiple_choice(self) -> None:
        question = "Where is the cat? A. Left B. Right C. Above D. Below"
        spec = classify_question(question, "B")
        self.assertEqual(spec.mode, "multiple_choice")
        self.assertEqual(spec.options["C"], "Above")
        self.assertTrue(score_answer("The answer is B.", spec)["correct"])
        self.assertFalse(score_answer("A", spec)["correct"])

    def test_option_text_reference(self) -> None:
        question = "Material? A) Glass B) Metal C) Wood D) Plastic"
        spec = classify_question(question, "wood")
        self.assertEqual(spec.reference, "C")
        self.assertTrue(score_answer("C", spec)["correct"])

    def test_yes_no(self) -> None:
        spec = classify_question("Is there a red ball?", "Yes")
        self.assertEqual(spec.mode, "yes_no")
        self.assertTrue(score_answer("yes.", spec)["correct"])
        self.assertFalse(score_answer("No", spec)["correct"])

    def test_short_answer_normalization(self) -> None:
        spec = classify_question("What material?", "wood")
        self.assertEqual(spec.mode, "short_answer")
        self.assertTrue(score_answer("The wooden.", spec)["correct"])
        self.assertEqual(normalize_answer("Two objects"), "2 objects")
        self.assertAlmostEqual(token_f1("red metal box", "red box"), 0.8)

    def test_parse_options_requires_multiple_markers(self) -> None:
        self.assertEqual(parse_options("Version A. is shown"), {})

    def test_aggregate(self) -> None:
        rows = [
            {
                "record_index": 0,
                "sample_index": 0,
                "answer_mode": "yes_no",
                "type": "negation",
                "correct": True,
                "token_f1": 1.0,
            },
            {
                "record_index": 0,
                "sample_index": 0,
                "answer_mode": "yes_no",
                "type": "negation",
                "correct": False,
                "token_f1": 0.0,
            },
        ]
        summary = aggregate_results(rows)
        self.assertEqual(summary["questions"]["accuracy"], 0.5)
        self.assertEqual(summary["prompts"]["soft_score"], 0.5)
        self.assertEqual(summary["prompts"]["hard_score"], 0.0)


if __name__ == "__main__":
    unittest.main()
