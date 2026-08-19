"""Per-question frozen-U gating for multi-dimensional TIFA cycle loss."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def per_question_tifa_cycle_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    targets: torch.Tensor,
    candidate_mask: torch.Tensor,
    *,
    gate_margin: float = 0.0,
    min_valid_questions: int = 1,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Average student CE only over real-image teacher-valid questions."""
    if student_logits.shape != teacher_logits.shape or student_logits.ndim != 2:
        raise ValueError("student and teacher logits must have matching [Q,C] shapes")
    if targets.shape != student_logits.shape[:-1]:
        raise ValueError("one target is required per question")
    if candidate_mask.shape != student_logits.shape:
        raise ValueError("candidate_mask must match logits")
    if gate_margin < 0.0:
        raise ValueError("gate_margin must be non-negative")
    if min_valid_questions < 1:
        raise ValueError("min_valid_questions must be positive")
    mask = candidate_mask.to(device=student_logits.device, dtype=torch.bool)
    targets = targets.to(device=student_logits.device, dtype=torch.long)
    indices = torch.arange(targets.numel(), device=student_logits.device)
    if not bool(mask[indices, targets].all()):
        raise ValueError("every target must be enabled by candidate_mask")
    student = student_logits.float().masked_fill(~mask, float("-inf"))
    teacher = teacher_logits.detach().float().masked_fill(~mask, float("-inf"))
    target_logits = teacher[indices, targets]
    competitors = teacher.masked_fill(
        F.one_hot(targets, num_classes=teacher.shape[-1]).bool(),
        float("-inf"),
    ).amax(dim=-1)
    margins = target_logits - competitors
    predictions = teacher.argmax(dim=-1)
    teacher_valid = predictions.eq(targets) & (margins > gate_margin)
    per_question_ce = F.cross_entropy(student, targets, reduction="none")
    valid_count = int(teacher_valid.sum().detach().cpu())
    step_valid = valid_count >= min_valid_questions
    if step_valid:
        loss = per_question_ce[teacher_valid].mean()
    else:
        loss = per_question_ce.sum() * 0.0
    details = {
        "gate_policy": "strict_per_question",
        "gate_margin_threshold": float(gate_margin),
        "min_valid_questions": min_valid_questions,
        "step_valid": step_valid,
        "valid_question_count": valid_count,
        "question_count": targets.numel(),
        "valid_question_fraction": valid_count / targets.numel(),
        "teacher_valid": teacher_valid.detach().cpu().tolist(),
        "teacher_predictions": predictions.detach().cpu().tolist(),
        "teacher_target_margins": margins.detach().cpu().tolist(),
        "student_predictions": student.argmax(dim=-1).detach().cpu().tolist(),
        "per_question_ce": per_question_ce.detach().cpu().tolist(),
    }
    return loss, details
