import unittest

from evaluation.c6_mask2former_alignment.prepare_external_tifa12_data import (
    DIMENSIONS,
    associated_term,
    best_spatial_pair,
    deterministic_choices,
    normalized,
    stable_validation,
)


class ExternalTifa12DataTest(unittest.TestCase):
    def test_normalization_is_case_and_punctuation_insensitive(self):
        self.assertEqual(normalized(" A Red-Car! "), "a red car")

    def test_choice_builder_keeps_answer_as_a_valid_target(self):
        choices, target = deterministic_choices(
            "dog", ["cat", "dog", "horse", "cow"], key="sample"
        )
        self.assertEqual(choices[target], "dog")
        self.assertEqual(len(choices), 4)
        self.assertEqual(len(set(choices)), 4)

    def test_caption_attribute_requires_object_association(self):
        self.assertEqual(
            associated_term("A red sports car is parked.", ["car"], ["red", "blue"]),
            ("car", "red"),
        )
        self.assertIsNone(
            associated_term("A red car is beside a bus.", ["bus"], ["red", "blue"])
        )

    def test_geometry_can_keep_orthogonal_relations(self):
        grouped = {
            "cat": [{"phrase": "cat", "box": [0.05, 0.05, 0.25, 0.25]}],
            "dog": [{"phrase": "dog", "box": [0.70, 0.70, 0.90, 0.90]}],
        }
        pair = best_spatial_pair(
            grouped,
            max_box_iou=0.2,
            max_min_box_overlap=0.5,
            position_threshold=0.1,
            relation_threshold=0.5,
        )
        self.assertIsNotNone(pair)
        self.assertEqual(pair["relations"], ["left", "above"])

    def test_split_is_deterministic(self):
        self.assertEqual(
            stable_validation(123, 20260817, 0.05),
            stable_validation(123, 20260817, 0.05),
        )

    def test_taxonomy_matches_tifa(self):
        self.assertEqual(len(DIMENSIONS), 12)
        self.assertIn("spatial", DIMENSIONS)
        self.assertIn("animal/human", DIMENSIONS)


if __name__ == "__main__":
    unittest.main()
