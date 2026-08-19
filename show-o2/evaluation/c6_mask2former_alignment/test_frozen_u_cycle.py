import unittest

import torch

from evaluation.c6_mask2former_alignment.frozen_u_cycle import (
    INVERSE_RELATION,
    RELATION_OPTIONS,
    build_question_batch,
    build_relation_choice_batch,
    prompt_u_task_loss,
    relation_option_token_ids,
    relation_questions,
    yes_no_token_ids,
)


class FakeTokenizer:
    pad_token_id = 0

    def __call__(
        self,
        text,
        *,
        add_special_tokens=False,
        truncation=False,
        max_length=None,
    ):
        del add_special_tokens
        if text == " yes":
            return {"input_ids": [41]}
        if text == " no":
            return {"input_ids": [42]}
        if text in ("A", "B", "C", "D"):
            return {"input_ids": [50 + ord(text) - ord("A")]}
        values = [100 + index for index, _ in enumerate(str(text).split())]
        if truncation and max_length is not None:
            values = values[:max_length]
        return {"input_ids": values}


class FrozenUCycleTest(unittest.TestCase):
    def test_inverse_relation_is_involution(self):
        for relation, inverse in INVERSE_RELATION.items():
            self.assertEqual(INVERSE_RELATION[inverse], relation)

    def test_relation_questions_include_positive_inverse_and_swap(self):
        questions = relation_questions(("cat", "dog"), "left")
        self.assertEqual([item.name for item in questions], [
            "positive", "inverse_negative", "swapped_inverse"
        ])
        self.assertEqual([item.target for item in questions], [0, 1, 0])
        self.assertEqual(
            [(item.subject, item.relation, item.object) for item in questions],
            [("cat", "left", "dog"), ("cat", "right", "dog"), ("dog", "right", "cat")],
        )

    def test_question_batch_has_answer_free_prefix_and_fixed_image_span(self):
        tokenizer = FakeTokenizer()
        batch = build_question_batch(
            ("cat", "dog"),
            "above",
            tokenizer,
            {"bos_id": 1, "boi_id": 2, "eoi_id": 3, "img_pad_id": 4},
            max_text_tokens=32,
            sequence_length=766,
            device=torch.device("cpu"),
        )
        self.assertEqual(tuple(batch.text_tokens.shape), (3, 766))
        self.assertEqual(batch.modality_positions.tolist(), [[[2, 730]]] * 3)
        self.assertEqual(batch.targets.tolist(), [0, 1, 0])
        self.assertTrue((batch.answer_logit_indices < 766).all())
        for row, index in zip(batch.text_tokens, batch.answer_logit_indices):
            self.assertNotIn(int(row[index]), (41, 42))

    def test_yes_no_tokens_follow_vsr_leading_space_contract(self):
        self.assertEqual(yes_no_token_ids(FakeTokenizer()), (41, 42))

    def test_relation_choice_uses_fixed_direction_coordinates(self):
        tokenizer = FakeTokenizer()
        batch = build_relation_choice_batch(
            ("cat", "dog"),
            "below",
            tokenizer,
            {"bos_id": 1, "boi_id": 2, "eoi_id": 3, "img_pad_id": 4},
            max_text_tokens=64,
            sequence_length=798,
            device=torch.device("cpu"),
        )
        self.assertEqual(RELATION_OPTIONS, ("left", "right", "above", "below"))
        self.assertEqual(batch.targets.tolist(), [3])
        self.assertEqual(relation_option_token_ids(tokenizer), (50, 51, 52, 53))
        self.assertIn("Where is cat relative to dog", batch.questions[0].statement)
        self.assertIn("A. left. B. right. C. above. D. below.", batch.questions[0].statement)

    def test_strict_teacher_gate_masks_invalid_teacher(self):
        student = torch.tensor([[0.2, -0.2], [0.2, -0.2], [0.2, -0.2]], requires_grad=True)
        teacher = torch.tensor([[2.0, -1.0], [-1.0, 2.0], [-1.0, 2.0]])
        targets = torch.tensor([0, 1, 0])
        loss, details = prompt_u_task_loss(student, teacher, targets)
        self.assertEqual(float(loss), 0.0)
        self.assertFalse(details["teacher_strict_valid"])
        loss.backward()
        self.assertEqual(float(student.grad.abs().sum()), 0.0)

    def test_valid_teacher_allows_gradient_to_student(self):
        student = torch.tensor([[0.2, -0.2], [0.2, -0.2], [0.2, -0.2]], requires_grad=True)
        teacher = torch.tensor([[2.0, -1.0], [-1.0, 2.0], [2.0, -1.0]])
        targets = torch.tensor([0, 1, 0])
        loss, details = prompt_u_task_loss(
            student, teacher, targets, gate_margin=0.5
        )
        self.assertGreater(float(loss), 0.0)
        self.assertTrue(details["teacher_strict_valid"])
        loss.backward()
        self.assertGreater(float(student.grad.abs().sum()), 0.0)

    def test_none_gate_retains_teacher_audit_but_trains(self):
        student = torch.zeros((3, 2), requires_grad=True)
        teacher = torch.zeros((3, 2))
        targets = torch.tensor([0, 1, 0])
        loss, details = prompt_u_task_loss(
            student, teacher, targets, gate_policy="none"
        )
        self.assertAlmostEqual(float(loss), 0.693147, places=5)
        self.assertTrue(details["gate_valid"])
        self.assertFalse(details["teacher_strict_valid"])

    def test_four_way_teacher_gate_and_gradient(self):
        student = torch.zeros((1, 4), requires_grad=True)
        teacher = torch.tensor([[0.1, -0.2, 1.4, 0.3]])
        loss, details = prompt_u_task_loss(
            student, teacher, torch.tensor([2]), gate_margin=0.5
        )
        self.assertTrue(details["teacher_strict_valid"])
        self.assertGreater(float(loss), 0.0)
        loss.backward()
        self.assertGreater(float(student.grad.abs().sum()), 0.0)

    def test_four_way_gate_rejects_wrong_teacher(self):
        student = torch.zeros((1, 4), requires_grad=True)
        teacher = torch.tensor([[1.4, -0.2, 0.1, 0.3]])
        loss, details = prompt_u_task_loss(
            student, teacher, torch.tensor([2]), gate_margin=0.0
        )
        self.assertFalse(details["teacher_strict_valid"])
        self.assertEqual(float(loss), 0.0)


if __name__ == "__main__":
    unittest.main()
