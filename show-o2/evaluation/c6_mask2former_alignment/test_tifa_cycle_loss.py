from __future__ import annotations

import unittest

import torch

from evaluation.c6_mask2former_alignment.tifa_cycle_loss import (
    per_question_tifa_cycle_loss,
)


class TifaCycleLossTest(unittest.TestCase):
    def test_invalid_teacher_question_does_not_drop_valid_question(self):
        student = torch.tensor(
            [[0.0, 1.0, 9.0, 9.0], [1.0, 0.0, 9.0, 9.0]],
            requires_grad=True,
        )
        teacher = torch.tensor(
            [[0.0, 2.0, 9.0, 9.0], [0.0, 2.0, 9.0, 9.0]]
        )
        targets = torch.tensor([1, 0])
        mask = torch.tensor(
            [[True, True, False, False], [True, True, False, False]]
        )
        loss, details = per_question_tifa_cycle_loss(
            student, teacher, targets, mask
        )
        loss.backward()
        self.assertTrue(details["step_valid"])
        self.assertEqual(details["teacher_valid"], [True, False])
        self.assertGreater(float(loss), 0.0)
        self.assertTrue(torch.equal(student.grad[1], torch.zeros(4)))

    def test_minimum_valid_question_threshold_skips_step(self):
        student = torch.tensor(
            [[0.0, 1.0], [1.0, 0.0]], requires_grad=True
        )
        teacher = torch.tensor([[0.0, 2.0], [0.0, 2.0]])
        loss, details = per_question_tifa_cycle_loss(
            student,
            teacher,
            torch.tensor([1, 0]),
            torch.ones((2, 2), dtype=torch.bool),
            min_valid_questions=2,
        )
        self.assertFalse(details["step_valid"])
        self.assertEqual(float(loss), 0.0)


if __name__ == "__main__":
    unittest.main()
