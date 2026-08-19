from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from evaluation.c6_mask2former_alignment.prepare_tifa_cycle_data import (
    annotate_cycle_record,
    grouped_records,
    question_axis_directions,
    source_family,
)


class CompleteTifaPreparationTest(unittest.TestCase):
    def test_source_family_accepts_all_official_sources(self):
        self.assertEqual(source_family("coco_7"), "coco")
        self.assertEqual(source_family("drawbench_7"), "drawbench")
        self.assertEqual(source_family("paintskill_7"), "paintskill")
        self.assertEqual(source_family("partiprompt_7"), "partiprompt")
        with self.assertRaises(ValueError):
            source_family("unknown_7")

    def test_axis_directions_use_spatial_element_not_distractors(self):
        self.assertEqual(
            question_axis_directions(
                {"element_type": "spatial", "element": "left to"}
            ),
            ("left",),
        )
        self.assertEqual(
            question_axis_directions(
                {"element_type": "spatial", "element": "on top of"}
            ),
            ("above",),
        )
        self.assertEqual(
            question_axis_directions(
                {"element_type": "spatial", "element": "underneath"}
            ),
            ("below",),
        )
        self.assertEqual(
            question_axis_directions(
                {"element_type": "spatial", "element": "near"}
            ),
            (),
        )
        self.assertEqual(
            question_axis_directions(
                {"element_type": "object", "element": "left shoe"}
            ),
            (),
        )

    def test_grouped_record_tracks_source_spatial_and_direction(self):
        payload = [
            {
                "id": "paintskill_0",
                "caption": "a cup left of a plate",
                "question": "is there a cup?",
                "choices": ["yes", "no"],
                "answer": "yes",
                "element_type": "object",
                "element": "cup",
            },
            {
                "id": "paintskill_0",
                "caption": "a cup left of a plate",
                "question": "is the cup left of the plate?",
                "choices": ["yes", "no"],
                "answer": "yes",
                "element_type": "spatial",
                "element": "left",
            },
        ]
        row = grouped_records(payload)[0]
        self.assertEqual(row["source_family"], "paintskill")
        self.assertEqual(row["spatial_question_count"], 1)
        self.assertEqual(row["axis_directions"], ["left"])

    def test_cycle_annotation_distinguishes_paired_and_unpaired(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "000000000007.jpg"
            image.write_bytes(b"test")
            common = {
                "caption": "caption",
                "dimensions": ["object", "spatial"],
                "spatial_question_count": 1,
                "axis_directions": ["left"],
                "qa_pairs": [],
            }
            paired = annotate_cycle_record(
                {
                    **common,
                    "sample_id": "coco_0",
                    "source_family": "coco",
                    "coco_val_id": "7",
                },
                root,
            )
            self.assertTrue(paired["has_reference_image"])
            self.assertEqual(paired["gate_policy"], "real_image_strict")
            self.assertEqual(paired["teacher_acceptance_status"], "not_required")

            unpaired = annotate_cycle_record(
                {
                    **common,
                    "sample_id": "paintskill_0",
                    "source_family": "paintskill",
                    "coco_val_id": None,
                },
                root,
            )
            self.assertFalse(unpaired["has_reference_image"])
            self.assertEqual(unpaired["gate_policy"], "unpaired_answer_target")
            self.assertEqual(unpaired["teacher_acceptance_status"], "not_built")
            self.assertNotIn("image_path", unpaired)


if __name__ == "__main__":
    unittest.main()
