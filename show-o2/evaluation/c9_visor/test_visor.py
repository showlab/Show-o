#!/usr/bin/env python3
"""Tests for deterministic VISOR manifest and metric logic."""

from __future__ import annotations

import unittest

from evaluation.c9_visor.prepare_manifest import manifest_row, select_records
from evaluation.c9_visor.visor_metrics import aggregate_visor, score_detections


class VisorMetricTest(unittest.TestCase):
    def test_any_pair_matches_official_rule(self) -> None:
        result = score_detections(
            ["cat", "cat", "chair"],
            [[30.0, 50.0], [80.0, 50.0], [60.0, 50.0]],
            "cat",
            "chair",
            "to the left of",
        )
        self.assertTrue(result["detected_both"])
        self.assertTrue(result["spatial_correct"])
        self.assertTrue(result["joint_success"])

    def test_missing_object_is_not_success(self) -> None:
        result = score_detections(
            ["cat"], [[20.0, 20.0]], "cat", "chair", "above"
        )
        self.assertTrue(result["detected_a"])
        self.assertFalse(result["detected_b"])
        self.assertFalse(result["joint_success"])

    def test_aggregate(self) -> None:
        rows = []
        for sample_index, success in enumerate((True, True, False, False)):
            rows.append(
                {
                    "unique_id": 10,
                    "sample_index": sample_index,
                    "relation": "above",
                    "detected_a": True,
                    "detected_b": True,
                    "detected_both": True,
                    "joint_success": success,
                }
            )
        summary = aggregate_visor(rows)
        self.assertEqual(summary["object_accuracy_both"], 100.0)
        self.assertEqual(summary["visor_cond"], 50.0)
        self.assertEqual(summary["visor_uncond"], 50.0)
        self.assertEqual(summary["visor_at_least_n"]["2"], 100.0)
        self.assertEqual(summary["visor_at_least_n"]["3"], 0.0)


class ManifestTest(unittest.TestCase):
    def test_pair_selection_is_group_complete(self) -> None:
        groups = {
            ("cat", "chair"): [{"unique_id": index} for index in range(8)],
            ("dog", "table"): [{"unique_id": index} for index in range(8, 16)],
        }
        selected = select_records(groups, pair_count=1, seed=7)
        self.assertEqual(len(selected), 8)
        self.assertIn(
            {int(item["unique_id"]) for item in selected},
            (set(range(8)), set(range(8, 16))),
        )

    def test_manifest_row(self) -> None:
        row = manifest_row(
            {
                "unique_id": 11,
                "text": "a cat above a chair",
                "obj_1_attributes": ["cat"],
                "obj_2_attributes": ["chair"],
                "rel_type": "above",
            },
            3,
        )
        self.assertEqual(row["manifest_index"], 3)
        self.assertEqual(row["pair_key"], ["cat", "chair"])


if __name__ == "__main__":
    unittest.main()
