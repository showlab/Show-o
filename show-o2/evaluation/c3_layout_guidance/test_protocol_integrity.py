#!/usr/bin/env python3
"""Tests for paired layout-guidance experiment bookkeeping."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evaluation.c3_layout_guidance.run_quick8_v2 import (
    ensure_protocol,
    load_recorded_pair_hashes,
    pending_sample_indices,
)
from evaluation.c3_layout_guidance.summarize_boxdiff_seed_ab import main as summarize


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


class PairingIntegrityTest(unittest.TestCase):
    def test_existing_image_without_pairing_record_is_regenerated(self):
        with tempfile.TemporaryDirectory() as temporary:
            prompt_dir = Path(temporary) / "baseline/geneval/00000"
            image = prompt_dir / "samples/00000.png"
            image.parent.mkdir(parents=True)
            image.write_bytes(b"incomplete-run-image")
            self.assertEqual(
                pending_sample_indices(prompt_dir, {}, 10, "baseline", 1),
                [0],
            )
            self.assertEqual(
                pending_sample_indices(
                    prompt_dir,
                    {"10:0": {"baseline": "a" * 64}},
                    10,
                    "baseline",
                    1,
                ),
                [],
            )

    def test_resume_requires_an_identical_protocol(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "protocol.json"
            protocol = {"variants": {"baseline": {"layers": (20,)}}}
            self.assertFalse(ensure_protocol(path, protocol))
            self.assertTrue(ensure_protocol(path, protocol))
            with self.assertRaisesRegex(RuntimeError, "protocol.json differs"):
                ensure_protocol(
                    path,
                    {"variants": {"baseline": {"layers": (19,)}}},
                )

    def test_resume_merges_persisted_and_chunk_pairing_records(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            digest = "a" * 64
            write_json(
                run_dir / "paired_initial_latents.json",
                {"10:0": {"baseline": digest}},
            )
            write_json(
                run_dir / "guided/raw/00000/run-00000.json",
                {
                    "source_index": 10,
                    "sample_indices": [0],
                    "initial_latent_sha256": [digest],
                },
            )

            self.assertEqual(
                load_recorded_pair_hashes(run_dir, ("baseline", "guided")),
                {"10:0": {"baseline": digest, "guided": digest}},
            )

    def test_resume_rejects_conflicting_chunk_pairing_record(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            write_json(
                run_dir / "paired_initial_latents.json",
                {"10:0": {"baseline": "a" * 64}},
            )
            write_json(
                run_dir / "guided/raw/00000/run-00000.json",
                {
                    "source_index": 10,
                    "sample_indices": [0],
                    "initial_latent_sha256": ["b" * 64],
                },
            )

            with self.assertRaisesRegex(RuntimeError, "mismatch across variants"):
                load_recorded_pair_hashes(run_dir, ("baseline", "guided"))

    def test_summary_marks_missing_expected_pair_as_invalid(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            manifest = run_dir / "manifest.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "local_index": 0,
                        "source_index": 10,
                        "relation": "left of",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            write_json(
                run_dir / "protocol.json",
                {
                    "protocol_version": "test",
                    "manifest": str(manifest),
                    "variants": {"baseline": {}, "guided": {}},
                    "samples_per_prompt": 1,
                },
            )
            for variant in ("baseline", "guided"):
                result_path = run_dir / variant / "results/results.jsonl"
                result_path.parent.mkdir(parents=True, exist_ok=True)
                result_path.write_text(
                    json.dumps(
                        {
                            "filename": str(
                                run_dir
                                / variant
                                / "geneval/00000/samples/00000.png"
                            ),
                            "correct": variant == "baseline",
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
            write_json(run_dir / "paired_initial_latents.json", {})
            output = run_dir / "summary.json"

            with patch.object(
                sys,
                "argv",
                ["summarize", "--run-dir", str(run_dir), "--output", str(output)],
            ):
                summarize()

            summary = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(summary["status"], "invalid_pairing")
            pairing = summary["initial_latent_pairing"]
            self.assertEqual(pairing["expected_pairs"], 1)
            self.assertEqual(pairing["absent_keys"], ["10:0"])


if __name__ == "__main__":
    unittest.main()
