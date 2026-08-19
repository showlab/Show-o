from __future__ import annotations

import unittest

from evaluation.c6_mask2former_alignment.tifa_cycle_sampling import (
    cycle_question_seed,
    filter_caption_copy_questions,
    prepare_cycle_questions,
    prepare_spatial_cycle_questions,
    shuffle_choice_positions,
)


class TifaCycleSamplingTest(unittest.TestCase):
    def setUp(self):
        self.row = {
            "sample_id": "sample-7",
            "caption": "A red dog runs beside two blue cars.",
            "qa_pairs": [
                {
                    "question": "A red dog runs beside two blue cars?",
                    "choices": ["yes", "no"],
                    "target_index": 0,
                    "element_type": "other",
                },
                {
                    "question": "What color is the dog?",
                    "choices": ["red", "green", "black", "white"],
                    "target_index": 0,
                    "element_type": "color",
                },
                {
                    "question": "How many cars are visible?",
                    "choices": ["two", "one", "three", "four"],
                    "target_index": 0,
                    "element_type": "counting",
                },
                {
                    "question": "What is the dog doing?",
                    "choices": ["running", "sleeping", "eating", "sitting"],
                    "target_index": 0,
                    "element_type": "activity",
                },
            ],
        }

    def test_caption_copy_filter_normalizes_terminal_punctuation(self):
        filtered = filter_caption_copy_questions(
            self.row["caption"], self.row["qa_pairs"]
        )
        self.assertEqual(len(filtered), 3)
        self.assertNotIn("other", {qa["element_type"] for qa in filtered})

    def test_shuffle_preserves_answer_and_records_permutation(self):
        qa = self.row["qa_pairs"][1]
        shuffled = shuffle_choice_positions(qa, seed=3)
        self.assertEqual(
            shuffled["choices"][shuffled["target_index"]],
            qa["choices"][qa["target_index"]],
        )
        self.assertEqual(sorted(shuffled["choice_permutation"]), [0, 1, 2, 3])

    def test_cycle_sampling_is_reproducible_and_step_dependent(self):
        first = prepare_cycle_questions(
            self.row, max_questions=3, base_seed=11, cycle_step=5
        )
        repeated = prepare_cycle_questions(
            self.row, max_questions=3, base_seed=11, cycle_step=5
        )
        later = prepare_cycle_questions(
            self.row, max_questions=3, base_seed=11, cycle_step=6
        )
        self.assertEqual(first, repeated)
        self.assertNotEqual(first, later)
        self.assertEqual(len({qa["element_type"] for qa in first}), 3)

    def test_seed_changes_with_route_independent_inputs_only(self):
        self.assertEqual(
            cycle_question_seed(9, "id", 3), cycle_question_seed(9, "id", 3)
        )
        self.assertNotEqual(
            cycle_question_seed(9, "id", 3), cycle_question_seed(9, "id", 4)
        )


if __name__ == "__main__":
    unittest.main()
