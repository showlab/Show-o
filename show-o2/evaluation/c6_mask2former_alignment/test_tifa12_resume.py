import json
import tempfile
import unittest
from pathlib import Path

from evaluation.c6_mask2former_alignment.train_tifa12_self_cycle import (
    restore_metric_history,
)


class Tifa12ResumeTest(unittest.TestCase):
    def test_restore_truncates_future_steps_and_recovers_counters(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.jsonl"
            rows = [
                {
                    "step": 1,
                    "gate_valid_questions": 3,
                    "question_count": 4,
                    "update_applied": True,
                },
                {
                    "step": 2,
                    "gate_valid_questions": 1,
                    "question_count": 4,
                    "update_applied": False,
                },
                {
                    "step": 3,
                    "gate_valid_questions": 4,
                    "question_count": 4,
                    "update_applied": True,
                },
            ]
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            totals = restore_metric_history(path, 2)
            self.assertEqual(totals, (4, 8, 1, 1))
            retained = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual([row["step"] for row in retained], [1, 2])


if __name__ == "__main__":
    unittest.main()
