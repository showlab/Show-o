"""GenEval-aligned position-only objective for B-prime generation training."""

from __future__ import annotations

import math
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from guidance.layout import LayoutSpec

from evaluation.c5_boxdiff_adapter.geneval_geometry import (
    POSITION_THRESHOLD,
    RELATION_THRESHOLD,
    geneval_relation_score,
)


LOSS_NAMES = (
    "relation_g",
    "presence_g",
    "extent_retention",
    "separation",
    "residual_norm",
)


def _soft_box(probability, xx, yy, *, min_width, min_height):
    """Estimate center and extent from a normalized object response map.

    For a uniform box, ``sqrt(12 * variance)`` recovers its width. This makes
    GenEval's object-size tolerance differentiable on a hidden-space map.
    """
    center_x = (probability * xx).sum()
    center_y = (probability * yy).sum()
    variance_x = (probability * (xx - center_x).square()).sum()
    variance_y = (probability * (yy - center_y).square()).sum()
    box_width = (12.0 * variance_x + 1e-12).sqrt().clamp(min=min_width, max=1.0)
    box_height = (12.0 * variance_y + 1e-12).sqrt().clamp(min=min_height, max=1.0)
    return center_x, center_y, box_width, box_height


def _straight_through_relu(value, softness):
    """Use GenEval's exact ReLU value and a softplus boundary gradient."""
    hard = F.relu(value)
    soft = F.softplus(value / softness) * softness
    return hard.detach() + soft - soft.detach()


def _geneval_scores_from_soft_boxes(
    first, second, *, position_threshold, softness, norm_floor
):
    first_x, first_y, first_width, first_height = first
    second_x, second_y, second_width, second_height = second
    dx, dy = first_x - second_x, first_y - second_y
    raw_offset_norm = torch.sqrt(dx.square() + dy.square()).clamp_min(norm_floor)
    tolerance_x = position_threshold * (first_width + second_width)
    tolerance_y = position_threshold * (first_height + second_height)
    return {
        "left": _straight_through_relu(-dx - tolerance_x, softness) / raw_offset_norm,
        "right": _straight_through_relu(dx - tolerance_x, softness) / raw_offset_norm,
        "above": _straight_through_relu(-dy - tolerance_y, softness) / raw_offset_norm,
        "below": _straight_through_relu(dy - tolerance_y, softness) / raw_offset_norm,
    }


def position_only_losses(
    spatial_hidden: torch.Tensor,
    base_hidden: torch.Tensor,
    layout: LayoutSpec,
    phrase_vectors: torch.Tensor,
    *,
    height: int = 27,
    width: int = 27,
    temperature: float = 10.0,
    position_threshold: float = POSITION_THRESHOLD,
    relation_threshold: float = RELATION_THRESHOLD,
    relation_margin: float = 0.1,
    boundary_softness: float = 0.05,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Compute benchmark-aligned relative-position losses without flow loss.

    COCO boxes are used only to audit the relation label. They are not absolute
    regression targets because their coordinates are absent from model input.
    """
    if spatial_hidden.shape != base_hidden.shape:
        raise ValueError("spatial_hidden and base_hidden must share a shape")
    if spatial_hidden.ndim != 3 or spatial_hidden.shape[:2] != (1, height * width):
        raise ValueError(f"expected hidden [1,{height * width},D], got {tuple(spatial_hidden.shape)}")
    if len(layout.objects) != 2 or phrase_vectors.shape[0] != 2:
        raise ValueError("B-prime position loss requires exactly two objects")
    if temperature <= 0.0 or boundary_softness <= 0.0:
        raise ValueError("temperature and boundary_softness must be positive")
    if not 0.0 <= position_threshold < 1.0:
        raise ValueError("position_threshold must be in [0, 1)")
    if relation_threshold <= 0.0 or relation_margin < 0.0:
        raise ValueError("relation_threshold and relation_margin must be valid")

    phrases = F.normalize(phrase_vectors.detach().float(), dim=-1)
    spatial_features = F.normalize(spatial_hidden.float(), dim=-1)
    base_features = F.normalize(base_hidden.detach().float(), dim=-1)
    spatial_logits = torch.einsum("bld,od->blo", spatial_features, phrases) * temperature
    base_logits = torch.einsum("bld,od->blo", base_features, phrases) * temperature
    maps = spatial_logits.softmax(dim=1)[0].transpose(0, 1).reshape(2, height, width)
    base_maps = base_logits.softmax(dim=1)[0].transpose(0, 1).reshape(2, height, width)
    ys = (torch.arange(height, device=spatial_hidden.device, dtype=torch.float32) + 0.5) / height
    xs = (torch.arange(width, device=spatial_hidden.device, dtype=torch.float32) + 0.5) / width
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")

    soft_boxes, base_soft_boxes, object_details = [], [], []
    for index, item in enumerate(layout.objects):
        soft_box = _soft_box(
            maps[index], xx, yy, min_width=1.0 / width, min_height=1.0 / height
        )
        base_soft_box = _soft_box(
            base_maps[index], xx, yy,
            min_width=1.0 / width,
            min_height=1.0 / height,
        )
        soft_boxes.append(soft_box)
        base_soft_boxes.append(base_soft_box)
        object_details.append({
            "phrase": item.phrase,
            "source_box_for_label_audit_only": list(item.box),
            "predicted_soft_box_cxcywh": [
                float(value.detach().cpu()) for value in soft_box
            ],
            "base_soft_box_cxcywh": [
                float(value.detach().cpu()) for value in base_soft_box
            ],
        })

    predicted_scores = _geneval_scores_from_soft_boxes(
        soft_boxes[0],
        soft_boxes[1],
        position_threshold=position_threshold,
        softness=boundary_softness,
        norm_floor=1.0 / max(height, width),
    )
    expected_score = predicted_scores[layout.relation]
    pass_target = expected_score.new_tensor(relation_threshold + relation_margin)
    relation_g = F.relu(pass_target - expected_score).square()

    predicted_extents = torch.stack([
        torch.stack((box[2], box[3])) for box in soft_boxes
    ])
    base_extents = torch.stack([
        torch.stack((box[2], box[3])) for box in base_soft_boxes
    ]).detach()
    extent_retention = F.smooth_l1_loss(predicted_extents, base_extents)

    normalization = math.log(float(spatial_logits.shape[1]))
    spatial_presence = torch.logsumexp(spatial_logits[0], dim=0) - normalization
    base_presence = torch.logsumexp(base_logits[0], dim=0) - normalization
    presence_g = F.relu(base_presence.detach() - spatial_presence).square().mean()
    separation = torch.sqrt(maps[0].clamp_min(1e-12) * maps[1].clamp_min(1e-12)).sum()
    residual_norm = (spatial_hidden.float() - base_hidden.detach().float()).square().mean()
    losses = {
        "relation_g": relation_g,
        "presence_g": presence_g,
        "extent_retention": extent_retention,
        "separation": separation,
        "residual_norm": residual_norm,
    }
    if any(not torch.isfinite(value) for value in losses.values()):
        raise FloatingPointError("non-finite B-prime GenEval loss")

    source_score = geneval_relation_score(
        layout.objects[0].box,
        layout.objects[1].box,
        layout.relation,
        position_threshold=position_threshold,
    )
    details = {
        "objects": object_details,
        "relation": layout.relation,
        "benchmark_formula": {
            "position_threshold": position_threshold,
            "relation_threshold": relation_threshold,
            "training_margin": relation_margin,
            "training_target": relation_threshold + relation_margin,
            "surrogate_norm_floor": 1.0 / max(height, width),
        },
        "predicted_relation_scores": {
            name: float(value.detach().cpu()) for name, value in predicted_scores.items()
        },
        "expected_relation_score": float(expected_score.detach().cpu()),
        "source_box_relation_score": source_score,
        "source_box_relation_passes": source_score > relation_threshold,
        "presence_base": [float(value) for value in base_presence.detach().cpu()],
        "presence_spatial": [float(value) for value in spatial_presence.detach().cpu()],
        "map_overlap": float(separation.detach().cpu()),
        "residual_rms": float(residual_norm.detach().sqrt().cpu()),
    }
    return losses, details


def module_grad_norm(loss, module, *, retain_graph):
    parameters = [parameter for parameter in module.parameters() if parameter.requires_grad]
    gradients = torch.autograd.grad(
        loss, parameters, retain_graph=retain_graph, allow_unused=True
    )
    squares = [
        gradient.detach().float().square().sum()
        for gradient in gradients
        if gradient is not None
    ]
    return float(torch.stack(squares).sum().sqrt().cpu()) if squares else 0.0


def calibrate_weights(
    losses: Mapping[str, torch.Tensor],
    module: torch.nn.Module,
    *,
    targets: Mapping[str, float] | None = None,
    minimum: float = 1e-4,
    maximum: float = 100.0,
):
    target_values = {
        "relation_g": 1.0,
        "presence_g": 0.5,
        "extent_retention": 0.25,
        "separation": 0.25,
        "residual_norm": 0.01,
    }
    if targets is not None:
        target_values.update({name: float(value) for name, value in targets.items()})
    unknown = set(losses) - set(LOSS_NAMES)
    if unknown:
        raise KeyError(f"unsupported B-prime GenEval losses: {sorted(unknown)}")
    names, norms = list(losses), {}
    for index, name in enumerate(names):
        norms[name] = module_grad_norm(
            losses[name], module, retain_graph=index + 1 < len(names)
        )
    primary = [
        norms[name]
        for name in ("relation_g", "separation")
        if norms.get(name, 0.0) > 0.0
    ]
    if not primary:
        raise RuntimeError(f"all primary B-prime GenEval gradient norms are zero: {norms}")
    reference = sorted(primary)[len(primary) // 2]
    weights = {}
    for name in LOSS_NAMES:
        norm = norms.get(name, 0.0)
        value = target_values[name] * reference / norm if norm > 0.0 else target_values[name]
        weights[name] = max(minimum, min(maximum, float(value)))
    return weights, norms


def compose_position_loss(losses: Mapping[str, torch.Tensor], weights: Mapping[str, float]):
    if set(losses) != set(LOSS_NAMES):
        raise KeyError(f"losses must be exactly {LOSS_NAMES}, got {tuple(losses)}")
    if set(weights) != set(LOSS_NAMES):
        raise KeyError(f"weights must be exactly {LOSS_NAMES}, got {tuple(weights)}")
    weighted = {name: losses[name].float() * float(weights[name]) for name in LOSS_NAMES}
    total = torch.stack(tuple(weighted.values())).sum()
    if not torch.isfinite(total):
        raise FloatingPointError("non-finite B-prime GenEval total loss")
    return total, weighted
