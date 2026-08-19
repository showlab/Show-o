import unittest

import torch

from evaluation.c6_mask2former_alignment.online_detector_loss import (
    LOSS_NAMES,
    TRAINING_LOSS_NAMES,
    alignment_losses,
    class_conditioned_outputs,
    compose_loss,
    compose_training_loss,
)


class OnlineDetectorLossTest(unittest.TestCase):
    def logits(self):
        classes = torch.full((1, 4, 4), -5.0)
        masks = torch.full((1, 4, 4, 4), -6.0)
        classes[0, 0, 0], classes[0, 1, 1] = 6.0, 6.0
        classes[0, 2:, -1] = 6.0
        masks[0, 0, :, :2] = 6.0
        masks[0, 1, :, 2:] = 6.0
        return classes, masks

    def diagonal_logits(self, first_above: bool):
        classes = torch.full((1, 4, 4), -5.0)
        masks = torch.full((1, 4, 4, 4), -6.0)
        classes[0, 0, 0], classes[0, 1, 1] = 6.0, 6.0
        classes[0, 2:, -1] = 6.0
        first_rows = slice(0, 2) if first_above else slice(2, 4)
        second_rows = slice(2, 4) if first_above else slice(0, 2)
        masks[0, 0, first_rows, :2] = 6.0
        masks[0, 1, second_rows, 2:] = 6.0
        return classes, masks

    def test_query_permutation_invariance(self):
        classes, masks = self.logits()
        original = class_conditioned_outputs(classes, masks, [0, 1])
        permutation = torch.tensor([2, 0, 3, 1])
        permuted = class_conditioned_outputs(
            classes[:, permutation], masks[:, permutation], [0, 1]
        )
        self.assertTrue(torch.allclose(original["class_masks"], permuted["class_masks"], atol=1e-6))
        self.assertTrue(torch.allclose(original["presence"], permuted["presence"], atol=1e-6))
        self.assertTrue(
            torch.allclose(
                original["detector_scores"].amax(dim=1),
                permuted["detector_scores"].amax(dim=1),
                atol=1e-6,
            )
        )

    def test_student_receives_gradient_teacher_does_not(self):
        classes, masks = self.logits()
        student_classes = classes.clone().requires_grad_(True)
        student_masks = masks.clone().requires_grad_(True)
        teacher = class_conditioned_outputs(classes, masks, [0, 1])
        student = class_conditioned_outputs(student_classes, student_masks, [0, 1])
        losses, details = alignment_losses(student, teacher, ["left"])
        total, _ = compose_loss(losses, {name: 1.0 for name in LOSS_NAMES})
        total.backward()
        self.assertIsNotNone(student_classes.grad)
        self.assertIsNotNone(student_masks.grad)
        self.assertGreater(float(student_classes.grad.norm() + student_masks.grad.norm()), 0.0)
        self.assertGreater(details["student_relation_score"][0], 0.5)

    def test_swapped_masks_raise_relation_loss(self):
        classes, masks = self.logits()
        teacher = class_conditioned_outputs(classes, masks, [0, 1])
        correct = class_conditioned_outputs(classes, masks, [0, 1])
        swapped = class_conditioned_outputs(classes, masks.flip(1), [0, 1])
        correct_losses, _ = alignment_losses(correct, teacher, ["left"])
        swapped_losses, _ = alignment_losses(swapped, teacher, ["left"])
        self.assertLess(float(correct_losses["relation_pass"]), float(swapped_losses["relation_pass"]))
        self.assertLess(float(correct_losses["mask_dice"]), float(swapped_losses["mask_dice"]))

    def test_orthogonal_relations_are_supervised_together(self):
        classes, teacher_masks = self.diagonal_logits(first_above=True)
        _, wrong_vertical_masks = self.diagonal_logits(first_above=False)
        teacher = class_conditioned_outputs(classes, teacher_masks, [0, 1])
        correct = class_conditioned_outputs(classes, teacher_masks, [0, 1])
        wrong_vertical = class_conditioned_outputs(
            classes, wrong_vertical_masks, [0, 1]
        )
        correct_losses, details = alignment_losses(
            correct, teacher, [("left", "above")]
        )
        wrong_losses, _ = alignment_losses(
            wrong_vertical, teacher, [("left", "above")]
        )
        self.assertEqual(details["relation_targets"], [["left", "above"]])
        self.assertEqual(len(details["student_relation_scores"][0]), 2)
        self.assertTrue(all(score > 0.5 for score in details["student_relation_scores"][0]))
        self.assertLess(
            float(correct_losses["relation_pass"]),
            float(wrong_losses["relation_pass"]),
        )

    def test_same_axis_relation_targets_are_rejected(self):
        classes, masks = self.logits()
        outputs = class_conditioned_outputs(classes, masks, [0, 1])
        with self.assertRaisesRegex(ValueError, "conflicting same-axis"):
            alignment_losses(outputs, outputs, [("left", "right")])

    def test_object_detect_uses_class_probability_times_mask_quality(self):
        classes, masks = self.logits()
        teacher = class_conditioned_outputs(classes, masks, [0, 1])
        good = class_conditioned_outputs(classes, masks, [0, 1])

        weak_classes = classes.clone()
        weak_classes[0, 0, 0] = -5.0
        weak_classes[0, 1, 1] = -5.0
        weak_classes[0, 0, -1] = 6.0
        weak_classes[0, 1, -1] = 6.0
        weak_classes.requires_grad_(True)
        weak_masks = masks.clone().requires_grad_(True)
        weak = class_conditioned_outputs(weak_classes, weak_masks, [0, 1])

        good_losses, _ = alignment_losses(good, teacher, ["left"])
        weak_losses, weak_details = alignment_losses(weak, teacher, ["left"])
        self.assertEqual(float(good_losses["object_detect"]), 0.0)
        self.assertGreater(float(weak_losses["object_detect"].detach()), 0.0)
        self.assertTrue(
            all(score < 0.35 for score in weak_details["student_best_detector_scores"][0])
        )

        weak_losses["object_detect"].backward()
        self.assertGreater(float(weak_classes.grad.norm()), 0.0)
        self.assertGreater(float(weak_masks.grad.norm()), 0.0)


    def test_detector_score_forward_matches_official_hard_mask_quality(self):
        classes, masks = self.logits()
        outputs = class_conditioned_outputs(classes, masks, [0, 1])
        probabilities = classes.float().softmax(dim=-1)[..., :-1]
        target_probabilities = probabilities.index_select(
            -1, torch.tensor([0, 1])
        )
        mask_probabilities = masks.float().sigmoid()
        foreground = (masks > 0).float()
        expected_quality = (
            (mask_probabilities * foreground).flatten(2).sum(-1)
            / foreground.flatten(2).sum(-1).clamp_min(1e-6)
        )
        expected_scores = target_probabilities * expected_quality.unsqueeze(-1)
        self.assertTrue(
            torch.allclose(outputs["mask_quality"], expected_quality, atol=1e-6)
        )
        self.assertTrue(
            torch.allclose(outputs["detector_scores"], expected_scores, atol=1e-6)
        )

    def test_invalid_object_teacher_masks_only_teacher_dependent_losses(self):
        classes, masks = self.logits()
        invalid_classes = torch.full_like(classes, -5.0)
        invalid_classes[..., -1] = 6.0
        teacher = class_conditioned_outputs(invalid_classes, masks, [0, 1])

        student_classes = invalid_classes.clone().requires_grad_(True)
        student_masks = masks.flip(1).clone().requires_grad_(True)
        student = class_conditioned_outputs(
            student_classes, student_masks, [0, 1]
        )
        losses, details = alignment_losses(student, teacher, ["left"])

        self.assertFalse(details["teacher_object_valid"][0])
        self.assertFalse(details["teacher_valid"][0])
        for name in ("mask_bce", "mask_dice", "presence", "relation_distill"):
            self.assertEqual(float(losses[name].detach()), 0.0)
        self.assertGreater(float(losses["object_detect"].detach()), 0.0)

        (losses["object_detect"] + losses["relation_pass"]).backward()
        self.assertGreater(float(student_classes.grad.norm()), 0.0)
        self.assertGreater(float(student_masks.grad.norm()), 0.0)

    def test_invalid_relation_teacher_keeps_object_and_mask_supervision(self):
        classes, masks = self.logits()
        teacher = class_conditioned_outputs(classes, masks, [0, 1])
        student = class_conditioned_outputs(classes, masks.flip(1), [0, 1])
        losses, details = alignment_losses(student, teacher, ["right"])

        self.assertTrue(details["teacher_object_valid"][0])
        self.assertFalse(details["teacher_relation_valid"][0])
        self.assertFalse(details["teacher_valid"][0])
        self.assertGreater(float(losses["mask_dice"].detach()), 0.0)
        self.assertEqual(float(losses["relation_distill"].detach()), 0.0)


    def test_training_loss_adds_flow_and_teacher_retention(self):
        parameter = torch.tensor(2.0, requires_grad=True)
        losses = {
            name: parameter * 0.0 + float(index + 1)
            for index, name in enumerate(TRAINING_LOSS_NAMES)
        }
        weights = {name: 0.0 for name in TRAINING_LOSS_NAMES}
        weights["flow"] = 0.25
        weights["teacher_retention"] = 0.20

        total, weighted = compose_training_loss(losses, weights)
        expected = 0.25 * losses["flow"] + 0.20 * losses["teacher_retention"]
        self.assertTrue(torch.allclose(total, expected))
        self.assertEqual(set(weighted), set(TRAINING_LOSS_NAMES))

    def test_training_loss_auxiliary_terms_backpropagate(self):
        parameter = torch.tensor(2.0, requires_grad=True)
        losses = {
            name: parameter * 0.0 for name in TRAINING_LOSS_NAMES
        }
        losses["flow"] = parameter.square()
        losses["teacher_retention"] = (parameter - 1.0).square()
        weights = {name: 0.0 for name in TRAINING_LOSS_NAMES}
        weights["flow"] = 0.25
        weights["teacher_retention"] = 0.20

        total, _ = compose_training_loss(losses, weights)
        total.backward()
        self.assertAlmostEqual(float(parameter.grad), 1.4, places=6)


if __name__ == "__main__":
    unittest.main()
