import unittest

from evaluation.c6_mask2former_alignment.tifa_cycle_sampling import (
    prepare_spatial_cycle_questions,
)


class Tifa12SpatialSamplingTest(unittest.TestCase):
    def test_dual_axis_row_keeps_both_axes_and_two_non_spatial_dimensions(self):
        row = {
            "sample_id": "dual-axis",
            "caption": "A cat is left of and above a dog.",
            "qa_pairs": [
                {
                    "element": "left",
                    "element_type": "spatial",
                    "question": "Where is the cat horizontally?",
                    "choices": ["left", "right"],
                    "target_index": 0,
                },
                {
                    "element": "above",
                    "element_type": "spatial",
                    "question": "Where is the cat vertically?",
                    "choices": ["above", "below"],
                    "target_index": 0,
                },
                {
                    "element": "cat",
                    "element_type": "object",
                    "question": "Which object is present?",
                    "choices": ["cat", "car", "bus", "cup"],
                    "target_index": 0,
                },
                {
                    "element": "one",
                    "element_type": "counting",
                    "question": "How many cats are present?",
                    "choices": ["one", "two", "three", "four"],
                    "target_index": 0,
                },
            ],
        }
        selected = prepare_spatial_cycle_questions(
            row,
            max_questions=4,
            min_non_spatial_dimensions=2,
            base_seed=20260817,
            cycle_step=1,
        )
        spatial = [question for question in selected if question["element_type"] == "spatial"]
        self.assertEqual(len(selected), 4)
        self.assertEqual({question["element"] for question in spatial}, {"left", "above"})
        self.assertEqual(
            {question["element_type"] for question in selected if question["element_type"] != "spatial"},
            {"object", "counting"},
        )


if __name__ == "__main__":
    unittest.main()
