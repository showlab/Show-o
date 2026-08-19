"""Position-only G losses for the B-prime split-objective experiment."""

from __future__ import annotations

import math
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from guidance.layout import LayoutSpec


LOSS_NAMES = ("box_g", "relation_g", "presence_g", "separation", "residual_norm")


def _mask(box, height, width, device):
    x0, y0, x1, y1 = box
    # Use token-cell overlap so thin valid boxes still provide supervision.
    x0, x1 = sorted((max(0.0, min(1.0, float(x0))), max(0.0, min(1.0, float(x1)))))
    y0, y1 = sorted((max(0.0, min(1.0, float(y0))), max(0.0, min(1.0, float(y1)))))
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"invalid normalized box {box}")
    cell_x0 = torch.arange(width, device=device, dtype=torch.float32) / width
    cell_x1 = (torch.arange(width, device=device, dtype=torch.float32) + 1.0) / width
    cell_y0 = torch.arange(height, device=device, dtype=torch.float32) / height
    cell_y1 = (torch.arange(height, device=device, dtype=torch.float32) + 1.0) / height
    overlap_x = (torch.minimum(cell_x1, cell_x1.new_tensor(x1)) - torch.maximum(cell_x0, cell_x0.new_tensor(x0))).clamp_min(0.0) * width
    overlap_y = (torch.minimum(cell_y1, cell_y1.new_tensor(y1)) - torch.maximum(cell_y0, cell_y0.new_tensor(y0))).clamp_min(0.0) * height
    value = overlap_y[:, None] * overlap_x[None, :]
    if float(value.sum()) == 0.0:
        raise ValueError(f"box {box} has no overlap with {height}x{width} token cells")
    return value


def _box_center(box):
    x0, y0, x1, y1 = box
    return 0.5 * (x0 + x1), 0.5 * (y0 + y1)


def position_only_losses(
    spatial_hidden: torch.Tensor,
    base_hidden: torch.Tensor,
    layout: LayoutSpec,
    phrase_vectors: torch.Tensor,
    *,
    height: int = 27,
    width: int = 27,
    temperature: float = 10.0,
    centroid_weight: float = 2.0,
    relation_margin_floor: float = 0.05,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Compute losses that never depend on velocity or a diffusion target."""
    if spatial_hidden.shape != base_hidden.shape:
        raise ValueError("spatial_hidden and base_hidden must share a shape")
    if spatial_hidden.ndim != 3 or spatial_hidden.shape[:2] != (1, height * width):
        raise ValueError(f"expected hidden [1,{height * width},D], got {tuple(spatial_hidden.shape)}")
    if len(layout.objects) != 2 or phrase_vectors.shape[0] != 2:
        raise ValueError("B-prime position loss requires exactly two objects")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")

    phrases = F.normalize(phrase_vectors.detach().float(), dim=-1)
    spatial_features = F.normalize(spatial_hidden.float(), dim=-1)
    base_features = F.normalize(base_hidden.detach().float(), dim=-1)
    spatial_logits = torch.einsum("bld,od->blo", spatial_features, phrases) * temperature
    base_logits = torch.einsum("bld,od->blo", base_features, phrases) * temperature
    maps = spatial_logits.softmax(dim=1)[0].transpose(0, 1).reshape(2, height, width)
    ys = (torch.arange(height, device=spatial_hidden.device, dtype=torch.float32) + 0.5) / height
    xs = (torch.arange(width, device=spatial_hidden.device, dtype=torch.float32) + 0.5) / width
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    box_terms, centroids, object_details = [], [], []
    for index, item in enumerate(layout.objects):
        probability = maps[index]
        box_mask = _mask(item.box, height, width, spatial_hidden.device)
        inside_mass = (probability * box_mask).sum().clamp_min(1e-8)
        centroid_x = (probability * xx).sum()
        centroid_y = (probability * yy).sum()
        target_x, target_y = _box_center(item.box)
        centroid_error = (centroid_x - target_x).square() + (centroid_y - target_y).square()
        box_terms.append(-inside_mass.log() + centroid_weight * centroid_error)
        centroids.append((centroid_x, centroid_y))
        object_details.append({
            "phrase": item.phrase,
            "box": list(item.box),
            "inside_mass": float(inside_mass.detach().cpu()),
            "centroid_x": float(centroid_x.detach().cpu()),
            "centroid_y": float(centroid_y.detach().cpu()),
            "target_x": target_x,
            "target_y": target_y,
            "centroid_error": float(centroid_error.detach().cpu()),
        })
    box_g = torch.stack(box_terms).mean()
    (first_x, first_y), (second_x, second_y) = centroids
    first_target, second_target = _box_center(layout.objects[0].box), _box_center(layout.objects[1].box)
    predicted_signed = {
        "left": second_x - first_x,
        "right": first_x - second_x,
        "above": second_y - first_y,
        "below": first_y - second_y,
    }[layout.relation]
    target_signed = {
        "left": second_target[0] - first_target[0],
        "right": first_target[0] - second_target[0],
        "above": second_target[1] - first_target[1],
        "below": first_target[1] - second_target[1],
    }[layout.relation]
    target_tensor = predicted_signed.new_tensor(target_signed)
    margin = predicted_signed.new_tensor(max(relation_margin_floor, min(0.20, 0.5 * target_signed)))
    relation_g = F.smooth_l1_loss(predicted_signed, target_tensor)
    relation_g = relation_g + F.relu(margin - predicted_signed).square()

    normalization = math.log(float(spatial_logits.shape[1]))
    spatial_presence = torch.logsumexp(spatial_logits[0], dim=0) - normalization
    base_presence = torch.logsumexp(base_logits[0], dim=0) - normalization
    presence_g = F.relu(base_presence.detach() - spatial_presence).square().mean()
    separation = torch.sqrt(maps[0].clamp_min(1e-12) * maps[1].clamp_min(1e-12)).sum()
    residual_norm = (spatial_hidden.float() - base_hidden.detach().float()).square().mean()
    losses = {
        "box_g": box_g,
        "relation_g": relation_g,
        "presence_g": presence_g,
        "separation": separation,
        "residual_norm": residual_norm,
    }
    if any(not torch.isfinite(value) for value in losses.values()):
        raise FloatingPointError("non-finite B-prime position loss")
    details = {
        "objects": object_details,
        "relation": layout.relation,
        "predicted_signed_delta": float(predicted_signed.detach().cpu()),
        "target_signed_delta": target_signed,
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
    squares = [gradient.detach().float().square().sum() for gradient in gradients if gradient is not None]
    return float(torch.stack(squares).sum().sqrt().cpu()) if squares else 0.0


def calibrate_weights(
    losses: Mapping[str, torch.Tensor],
    module: torch.nn.Module,
    *,
    targets: Mapping[str, float] | None = None,
    minimum: float = 1e-4,
    maximum: float = 100.0,
):
    """Equalize nonzero first-batch gradient scales, then apply priority targets."""
    target_values = {
        "box_g": 1.0,
        "relation_g": 0.5,
        "presence_g": 0.25,
        "separation": 0.25,
        "residual_norm": 0.01,
    }
    if targets is not None:
        target_values.update({name: float(value) for name, value in targets.items()})
    unknown = set(losses) - set(LOSS_NAMES)
    if unknown:
        raise KeyError(f"unsupported B-prime losses: {sorted(unknown)}")
    names, norms = list(losses), {}
    for index, name in enumerate(names):
        norms[name] = module_grad_norm(
            losses[name], module, retain_graph=index + 1 < len(names)
        )
    primary = [norms[name] for name in ("box_g", "relation_g", "separation") if norms.get(name, 0.0) > 0.0]
    if not primary:
        raise RuntimeError(f"all primary B-prime gradient norms are zero: {norms}")
    reference = sorted(primary)[len(primary) // 2]
    weights = {}
    for name in LOSS_NAMES:
        norm = norms.get(name, 0.0)
        value = target_values[name] * reference / norm if norm > 0.0 else target_values[name]
        weights[name] = max(minimum, min(maximum, float(value)))
    return weights, norms


def compose_position_loss(losses: Mapping[str, torch.Tensor], weights: Mapping[str, float]):
    if set(losses) != set(LOSS_NAMES):
        raise KeyError(f"B-prime losses must be exactly {LOSS_NAMES}, got {tuple(losses)}")
    if set(weights) != set(LOSS_NAMES):
        raise KeyError(f"B-prime weights must be exactly {LOSS_NAMES}, got {tuple(weights)}")
    weighted = {name: losses[name].float() * float(weights[name]) for name in LOSS_NAMES}
    total = torch.stack(tuple(weighted.values())).sum()
    if not torch.isfinite(total):
        raise FloatingPointError("non-finite B-prime total loss")
    return total, weighted
