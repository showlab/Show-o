from __future__ import annotations

import unittest

import torch

from evaluation.c6_mask2former_alignment.tifa_cycle import (
    build_tifa_question_batch,
    option_token_ids,
    select_multidimensional_questions,
    tifa_next_token_loss,
)


class DummyTokenizer:
    pad_token_id = 0

    def __init__(self):
        self.prompts = []

    def __call__(self, text, add_special_tokens=False):
        del add_special_tokens
        if text in ("A", "B", "C", "D"):
            return {"input_ids": [10 + ord(text) - ord("A")]}
        self.prompts.append(text)
        return {"input_ids": list(range(100, 100 + len(text.split())))}


class TifaCycleTest(unittest.TestCase):
    def setUp(self):
        self.qa_pairs = [
            {
                "question": "What color is the board?",
                "choices": ["white", "black", "red", "blue"],
                "target_index": 0,
                "element_type": "color",
            },
            {
                "question": "Is a person visible?",
                "choices": ["yes", "no"],
                "target_index": 0,
                "element_type": "animal/human",
            },
            {
                "question": "What is the person doing?",
                "choices": ["walking", "sleeping", "swimming", "driving"],
                "target_index": 0,
                "element_type": "activity",
            },
        ]

    def test_option_ids_are_single_and_unique(self):
        self.assertEqual(option_token_ids(DummyTokenizer()), (10, 11, 12, 13))

    def test_sampler_selects_distinct_dimensions_first(self):
        selected = select_multidimensional_questions(
            self.qa_pairs, max_questions=2, seed=7
        )
        self.assertEqual(len(selected), 2)
        self.assertEqual(len({row["element_type"] for row in selected}), 2)

    def test_batch_masks_two_choice_questions_and_excludes_caption(self):
        tokenizer = DummyTokenizer()
        token_ids = {"bos_id": 1, "boi_id": 2, "img_pad_id": 3, "eoi_id": 4}
        batch = build_tifa_question_batch(
            self.qa_pairs[:2],
            tokenizer,
            token_ids,
            max_text_tokens=80,
            sequence_length=814,
            device=torch.device("cpu"),
        )
        self.assertEqual(batch.targets.tolist(), [0, 0])
        self.assertEqual(
            batch.candidate_mask.tolist(),
            [[True, True, True, True], [True, True, False, False]],
        )
        self.assertTrue(all("generation caption" not in text for text in tokenizer.prompts))
        self.assertTrue(all("Question about this image" in text for text in tokenizer.prompts))

    def test_masked_loss_backpropagates_only_through_valid_choices(self):
        logits = torch.tensor(
            [[0.1, 0.2, 99.0, 99.0], [0.1, 0.2, 0.3, 0.4]],
            requires_grad=True,
        )
        targets = torch.tensor([0, 3])
        mask = torch.tensor(
            [[True, True, False, False], [True, True, True, True]]
        )
        loss, details = tifa_next_token_loss(logits, targets, mask)
        loss.backward()
        self.assertGreater(float(loss), 0.0)
        self.assertEqual(logits.grad[0, 2:].tolist(), [0.0, 0.0])
        self.assertEqual(details["gate_policy"], "none")

    def test_strict_gate_uses_real_image_teacher_logits(self):
        student = torch.tensor([[0.0, 1.0, 0.0, 0.0]], requires_grad=True)
        target = torch.tensor([1])
        mask = torch.tensor([[True, True, False, False]])
        accepted, details = tifa_next_token_loss(
            student,
            target,
            mask,
            teacher_logits=torch.tensor([[0.0, 2.0, 5.0, 5.0]]),
            gate_policy="strict",
            gate_margin=0.5,
        )
        self.assertTrue(details["gate_valid"])
        self.assertGreater(float(accepted), 0.0)
        rejected, details = tifa_next_token_loss(
            student,
            target,
            mask,
            teacher_logits=torch.tensor([[2.0, 0.0, 5.0, 5.0]]),
            gate_policy="strict",
        )
        self.assertFalse(details["gate_valid"])
        self.assertEqual(float(rejected), 0.0)


if __name__ == "__main__":
    unittest.main()
