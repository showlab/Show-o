"""Differentiable class-conditioned Mask2Former alignment losses."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F


LOSS_NAMES = (
    "mask_bce",
    "mask_dice",
    "presence",
    "relation_distill",
    "relation_pass",
    "object_detect",
)

AUXILIARY_LOSS_NAMES = (
    "flow",
    "teacher_retention",
)

TRAINING_LOSS_NAMES = LOSS_NAMES + AUXILIARY_LOSS_NAMES

RELATION_AXES = {
    "left": "horizontal",
    "right": "horizontal",
    "above": "vertical",
    "below": "vertical",
}


def class_conditioned_outputs(
    class_logits: torch.Tensor,
    mask_logits: torch.Tensor,
    class_ids: Sequence[int] | torch.Tensor,
    *,
    query_temperature: float = 0.25,
    foreground_temperature: float = 0.25,
    quality_mode: str = "official",
    eps: float = 1e-6,
) -> dict[str, torch.Tensor]:
    """Aggregate unordered Mask2Former queries into one soft mask per class."""
    if class_logits.ndim != 3 or mask_logits.ndim != 4:
        raise ValueError("expected class [B,Q,C+1] and mask [B,Q,H,W] logits")
    if class_logits.shape[:2] != mask_logits.shape[:2]:
        raise ValueError("class and mask query dimensions must match")
    if query_temperature <= 0.0 or foreground_temperature <= 0.0:
        raise ValueError("temperatures must be positive")
    if quality_mode not in ("official", "smooth"):
        raise ValueError("quality_mode must be official or smooth")
    ids = torch.as_tensor(class_ids, device=class_logits.device, dtype=torch.long)
    if ids.ndim != 1 or len(ids) != 2:
        raise ValueError("position alignment requires exactly two class ids")
    if int(ids.min()) < 0 or int(ids.max()) >= class_logits.shape[-1] - 1:
        raise ValueError("class id is outside the non-background logits")

    class_probabilities = class_logits.float().softmax(dim=-1)[..., :-1]
    target_class_probabilities = class_probabilities.index_select(-1, ids)
    mask_probabilities = mask_logits.float().sigmoid()

    # The official path uses hard mask-logit-positive pixels in the forward
    # detector score. The legacy ablation keeps the prior fully smooth proxy.
    hard_foreground = (mask_logits.float() > 0).to(mask_probabilities.dtype)
    soft_foreground = (mask_logits.float() / foreground_temperature).sigmoid()
    hard_quality = (
        (mask_probabilities * hard_foreground).flatten(2).sum(-1)
        / hard_foreground.flatten(2).sum(-1).clamp_min(eps)
    )
    soft_quality = (
        (mask_probabilities * soft_foreground).flatten(2).sum(-1)
        / soft_foreground.flatten(2).sum(-1).clamp_min(eps)
    )
    if quality_mode == "official":
        mask_quality = (
            hard_quality.detach()
            + soft_quality
            - soft_quality.detach()
        )
    else:
        mask_quality = soft_quality
    detector_scores = target_class_probabilities * mask_quality.unsqueeze(-1)
    query_weights = F.softmax(
        detector_scores.clamp_min(eps).log() / query_temperature,
        dim=1,
    )
    class_masks = torch.einsum("bqo,bqhw->bohw", query_weights, mask_probabilities)
    presence = (query_weights * detector_scores).sum(dim=1)
    entropy = -(query_weights * query_weights.clamp_min(eps).log()).sum(dim=1)
    return {
        "class_masks": class_masks,
        "presence": presence,
        "detector_scores": detector_scores,
        "mask_quality": mask_quality,
        "query_weights": query_weights,
        "query_entropy": entropy,
    }


def _soft_boxes(class_masks: torch.Tensor) -> torch.Tensor:
    if class_masks.ndim != 4 or class_masks.shape[1] != 2:
        raise ValueError("class_masks must have shape [B,2,H,W]")
    batch, objects, height, width = class_masks.shape
    mass = class_masks.flatten(2).sum(-1).clamp_min(1e-6)
    probability = class_masks / mass[:, :, None, None]
    ys = (torch.arange(height, device=class_masks.device, dtype=torch.float32) + 0.5) / height
    xs = (torch.arange(width, device=class_masks.device, dtype=torch.float32) + 0.5) / width
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    center_x = (probability * xx).sum(dim=(-2, -1))
    center_y = (probability * yy).sum(dim=(-2, -1))
    variance_x = (probability * (xx - center_x[:, :, None, None]).square()).sum(dim=(-2, -1))
    variance_y = (probability * (yy - center_y[:, :, None, None]).square()).sum(dim=(-2, -1))
    box_width = (12.0 * variance_x + 1e-12).sqrt().clamp(1.0 / width, 1.0)
    box_height = (12.0 * variance_y + 1e-12).sqrt().clamp(1.0 / height, 1.0)
    return torch.stack((center_x, center_y, box_width, box_height), dim=-1).reshape(batch, objects, 4)


def _straight_through_relu(value: torch.Tensor, softness: float) -> torch.Tensor:
    hard = F.relu(value)
    soft = F.softplus(value / softness) * softness
    return hard.detach() + soft - soft.detach()


def relation_scores(
    boxes: torch.Tensor,
    *,
    position_threshold: float = 0.1,
    boundary_softness: float = 0.05,
    norm_floor: float = 1e-3,
) -> dict[str, torch.Tensor]:
    """Apply GenEval's geometry to differentiable cx/cy/width/height boxes."""
    if boxes.ndim != 3 or boxes.shape[1:] != (2, 4):
        raise ValueError("boxes must have shape [B,2,4]")
    first, second = boxes[:, 0], boxes[:, 1]
    dx, dy = first[:, 0] - second[:, 0], first[:, 1] - second[:, 1]
    tolerance_x = position_threshold * (first[:, 2] + second[:, 2])
    tolerance_y = position_threshold * (first[:, 3] + second[:, 3])
    norm = torch.sqrt(dx.square() + dy.square() + norm_floor**2)
    return {
        "left": _straight_through_relu(-dx - tolerance_x, boundary_softness) / norm,
        "right": _straight_through_relu(dx - tolerance_x, boundary_softness) / norm,
        "above": _straight_through_relu(-dy - tolerance_y, boundary_softness) / norm,
        "below": _straight_through_relu(dy - tolerance_y, boundary_softness) / norm,
    }


def _masked_mean(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    if values.ndim != 1 or valid.shape != values.shape:
        raise ValueError("masked losses require matching one-dimensional tensors")
    weights = valid.to(dtype=values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def _normalize_relation_targets(
    relations: Sequence[str | Sequence[str]], batch_size: int
) -> tuple[tuple[str, ...], ...]:
    if len(relations) != batch_size:
        raise ValueError("one relation set is required per batch item")
    normalized = []
    for index, value in enumerate(relations):
        targets = (value,) if isinstance(value, str) else tuple(value)
        if not 1 <= len(targets) <= 2:
            raise ValueError(
                f"batch item {index} must have one or two relation targets"
            )
        if len(set(targets)) != len(targets):
            raise ValueError(f"batch item {index} has duplicate relation targets")
        try:
            axes = [RELATION_AXES[target] for target in targets]
        except KeyError as error:
            raise ValueError(
                f"batch item {index} has unsupported relation {error.args[0]!r}"
            ) from error
        if len(set(axes)) != len(axes):
            raise ValueError(
                f"batch item {index} has conflicting same-axis relations: {targets}"
            )
        normalized.append(tuple(str(target) for target in targets))
    return tuple(normalized)


def alignment_losses(
    student: Mapping[str, torch.Tensor],
    teacher: Mapping[str, torch.Tensor],
    relations: Sequence[str | Sequence[str]],
    *,
    presence_floor: float = 0.3,
    relation_threshold: float = 0.5,
    relation_margin: float = 0.1,
    object_threshold: float = 0.3,
    object_margin: float = 0.05,
    object_teacher_ratio: float = 0.8,
    teacher_validity_mask: bool = True,
    object_loss_mode: str = "squared",
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    student_masks = student["class_masks"].float()
    teacher_masks = teacher["class_masks"].detach().float()
    if not torch.isfinite(student_masks).all() or not torch.isfinite(teacher_masks).all():
        raise FloatingPointError("non-finite class-conditioned detector mask")
    student_masks = student_masks.clamp(0.0, 1.0)
    teacher_masks = teacher_masks.clamp(0.0, 1.0)
    student_presence = student["presence"].float()
    teacher_presence = teacher["presence"].detach().float()
    student_detector_scores = student["detector_scores"].float()
    teacher_detector_scores = teacher["detector_scores"].detach().float()
    if student_masks.shape != teacher_masks.shape or student_presence.shape != teacher_presence.shape:
        raise ValueError("student and teacher detector outputs must share shapes")
    if student_detector_scores.shape != teacher_detector_scores.shape:
        raise ValueError("student and teacher detector scores must share shapes")
    if object_threshold < 0.0 or object_margin < 0.0:
        raise ValueError("object threshold and margin must be non-negative")
    if not 0.0 <= object_teacher_ratio <= 1.0:
        raise ValueError("object teacher ratio must be between zero and one")
    if object_loss_mode not in ("linear", "squared"):
        raise ValueError("object loss mode must be linear or squared")
    relation_targets = _normalize_relation_targets(
        relations, student_masks.shape[0]
    )

    student_best_detector_scores = student_detector_scores.amax(dim=1)
    teacher_best_detector_scores = teacher_detector_scores.amax(dim=1)
    teacher_class_valid_actual = teacher_best_detector_scores >= object_threshold
    teacher_object_valid_actual = teacher_class_valid_actual.all(dim=1)
    if teacher_validity_mask:
        teacher_class_valid = teacher_class_valid_actual
        teacher_object_valid = teacher_object_valid_actual
    else:
        teacher_class_valid = torch.ones_like(
            teacher_class_valid_actual, dtype=torch.bool
        )
        teacher_object_valid = torch.ones_like(
            teacher_object_valid_actual, dtype=torch.bool
        )

    mask_bce_per_sample = F.binary_cross_entropy(
        student_masks.clamp(1e-6, 1.0 - 1e-6),
        teacher_masks,
        reduction="none",
    ).flatten(1).mean(-1)
    intersection = (student_masks * teacher_masks).sum(dim=(-2, -1))
    denominator = student_masks.sum(dim=(-2, -1)) + teacher_masks.sum(dim=(-2, -1))
    mask_dice_per_sample = (
        1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)
    ).mean(dim=1)
    mask_bce = _masked_mean(mask_bce_per_sample, teacher_object_valid)
    mask_dice = _masked_mean(mask_dice_per_sample, teacher_object_valid)

    target_presence = torch.maximum(
        teacher_presence, teacher_presence.new_full(teacher_presence.shape, presence_floor)
    )
    presence_per_sample = F.smooth_l1_loss(
        student_presence, teacher_presence, reduction="none"
    ).mean(dim=1)
    presence_per_sample = presence_per_sample + F.relu(
        target_presence - student_presence
    ).square().mean(dim=1)
    presence = _masked_mean(presence_per_sample, teacher_object_valid)

    height, width = student_masks.shape[-2:]
    norm_floor = 1.0 / max(height, width)
    student_boxes, teacher_boxes = _soft_boxes(student_masks), _soft_boxes(teacher_masks)
    student_scores = relation_scores(student_boxes, norm_floor=norm_floor)
    teacher_scores = relation_scores(teacher_boxes, norm_floor=norm_floor)
    student_expected_by_sample = []
    teacher_expected_by_sample = []
    teacher_relation_valid_by_sample = []
    relation_distill_per_sample = []
    relation_distill_sample_valid = []
    relation_pass_per_sample = []
    relation_target = relation_threshold + relation_margin
    for index, targets in enumerate(relation_targets):
        student_expected = torch.stack([
            student_scores[relation][index] for relation in targets
        ])
        teacher_expected = torch.stack([
            teacher_scores[relation][index] for relation in targets
        ]).detach()
        relation_valid_actual = teacher_expected > relation_threshold
        joint_valid_actual = relation_valid_actual & teacher_object_valid_actual[index]
        relation_valid = (
            joint_valid_actual
            if teacher_validity_mask
            else torch.ones_like(joint_valid_actual, dtype=torch.bool)
        )
        distill_targets = F.smooth_l1_loss(
            student_expected, teacher_expected, reduction="none"
        )
        relation_distill_per_sample.append(
            _masked_mean(distill_targets, relation_valid)
        )
        relation_distill_sample_valid.append(relation_valid.any())
        relation_pass_per_sample.append(
            F.relu(relation_target - student_expected).square().mean()
        )
        student_expected_by_sample.append(student_expected)
        teacher_expected_by_sample.append(teacher_expected)
        teacher_relation_valid_by_sample.append(relation_valid_actual)

    relation_distill = _masked_mean(
        torch.stack(relation_distill_per_sample),
        torch.stack(relation_distill_sample_valid),
    )
    relation_pass = torch.stack(relation_pass_per_sample).mean()
    teacher_relation_valid_actual = torch.stack([
        values.all() for values in teacher_relation_valid_by_sample
    ])
    teacher_valid_actual = (
        teacher_object_valid_actual & teacher_relation_valid_actual
    )
    if teacher_validity_mask:
        teacher_relation_valid = teacher_relation_valid_actual
        teacher_valid = teacher_valid_actual
    else:
        teacher_relation_valid = torch.ones_like(
            teacher_relation_valid_actual, dtype=torch.bool
        )
        teacher_valid = torch.ones_like(
            teacher_valid_actual, dtype=torch.bool
        )

    # GenEval accepts an object when at least one query class probability
    # times official mask quality clears its detector threshold. Valid teachers
    # provide a stronger one-sided retention target; invalid teacher classes
    # fall back to the metadata-driven absolute threshold.
    absolute_object_target = object_threshold + object_margin
    absolute_targets = student_best_detector_scores.new_full(
        student_best_detector_scores.shape, absolute_object_target
    )
    teacher_targets = teacher_best_detector_scores * object_teacher_ratio
    object_targets = torch.maximum(absolute_targets, teacher_targets)
    object_targets = torch.where(
        teacher_class_valid_actual, object_targets, absolute_targets
    ).detach()
    object_deficit = F.relu(object_targets - student_best_detector_scores)
    if object_loss_mode == "linear":
        object_detect = object_deficit.mean()
    else:
        object_detect = object_deficit.square().mean()

    losses = {
        "mask_bce": mask_bce,
        "mask_dice": mask_dice,
        "presence": presence,
        "relation_distill": relation_distill,
        "relation_pass": relation_pass,
        "object_detect": object_detect,
    }
    if any(not torch.isfinite(value) for value in losses.values()):
        raise FloatingPointError("non-finite online detector alignment loss")
    details = {
        "student_presence": student_presence.detach().cpu().tolist(),
        "teacher_presence": teacher_presence.cpu().tolist(),
        "student_best_detector_scores": student_best_detector_scores.detach().cpu().tolist(),
        "teacher_best_detector_scores": teacher_best_detector_scores.cpu().tolist(),
        "object_detection_target": absolute_object_target,
        "object_detection_targets": object_targets.cpu().tolist(),
        "teacher_class_valid": teacher_class_valid_actual.cpu().tolist(),
        "teacher_object_valid": teacher_object_valid_actual.cpu().tolist(),
        "teacher_relation_valid": teacher_relation_valid_actual.cpu().tolist(),
        "teacher_valid": teacher_valid_actual.cpu().tolist(),
        "teacher_mask_class_valid": teacher_class_valid.cpu().tolist(),
        "teacher_mask_object_valid": teacher_object_valid.cpu().tolist(),
        "teacher_mask_relation_valid": teacher_relation_valid.cpu().tolist(),
        "teacher_mask_joint_valid": teacher_valid.cpu().tolist(),
        "teacher_validity_mask": teacher_validity_mask,
        "student_boxes_cxcywh": student_boxes.detach().cpu().tolist(),
        "teacher_boxes_cxcywh": teacher_boxes.detach().cpu().tolist(),
        "relation_targets": [list(targets) for targets in relation_targets],
        "student_relation_scores": [
            values.detach().cpu().tolist() for values in student_expected_by_sample
        ],
        "teacher_relation_scores": [
            values.cpu().tolist() for values in teacher_expected_by_sample
        ],
        "teacher_relation_valid_by_target": [
            values.cpu().tolist() for values in teacher_relation_valid_by_sample
        ],
        "student_relation_score": [
            float(values.detach().mean().cpu())
            for values in student_expected_by_sample
        ],
        "teacher_relation_score": [
            float(values.mean().cpu())
            for values in teacher_expected_by_sample
        ],
        "student_query_entropy": student["query_entropy"].detach().cpu().tolist(),
        "teacher_query_entropy": teacher["query_entropy"].detach().cpu().tolist(),
    }
    return losses, details


def compose_loss(
    losses: Mapping[str, torch.Tensor],
    weights: Mapping[str, float],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if set(losses) != set(LOSS_NAMES) or set(weights) != set(LOSS_NAMES):
        raise KeyError(f"losses and weights must exactly match {LOSS_NAMES}")
    weighted = {name: losses[name].float() * float(weights[name]) for name in LOSS_NAMES}
    total = torch.stack(tuple(weighted.values())).sum()
    if not torch.isfinite(total):
        raise FloatingPointError("non-finite composed online detector loss")
    return total, weighted


def compose_training_loss(
    losses: Mapping[str, torch.Tensor],
    weights: Mapping[str, float],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Combine detector alignment with flow and frozen-teacher retention."""
    if set(losses) != set(TRAINING_LOSS_NAMES):
        raise KeyError(
            f"training losses must exactly match {TRAINING_LOSS_NAMES}, "
            f"got {tuple(losses)}"
        )
    if set(weights) != set(TRAINING_LOSS_NAMES):
        raise KeyError(
            f"training weights must exactly match {TRAINING_LOSS_NAMES}, "
            f"got {tuple(weights)}"
        )
    weighted = {
        name: losses[name].float() * float(weights[name])
        for name in TRAINING_LOSS_NAMES
    }
    total = torch.stack(tuple(weighted.values())).sum()
    if not torch.isfinite(total):
        raise FloatingPointError("non-finite composed online training loss")
    return total, weighted
