"""Differentiable layout energy over Qwen image-to-text attention."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from .layout import LayoutSpec


@dataclass(frozen=True)
class EnergyConfig:
    layers: tuple[int, ...]
    inside_weight: float = 1.0
    cross_weight: float = 0.0
    relation_weight: float = 0.0
    relation_margin: float = 0.20
    relation_dominance_weight: float = 0.0
    relation_dominance_ratio: float = 0.60
    entropy_weight: float = 0.0
    entropy_target: float = 1.0
    presence_weight: float = 0.0
    presence_min: float = 0.0
    boxdiff_inner_weight: float = 0.0
    boxdiff_outer_weight: float = 0.0
    boxdiff_corner_weight: float = 0.0
    boxdiff_p: float = 0.20
    boxdiff_corner_radius: int = 1
    boxdiff_text_scale: float = 100.0
    boxdiff_token_mode: str = "phrase_mean"
    attention_normalization: str = "legacy_boxdiff"
    text_gate_power: float = 0.0
    eps: float = 1e-8


def _box_mask(
    box: tuple[float, float, float, float],
    height: int,
    width: int,
    device: torch.device,
) -> torch.Tensor:
    x0, y0, x1, y1 = box
    ys = (torch.arange(height, device=device, dtype=torch.float32) + 0.5) / height
    xs = (torch.arange(width, device=device, dtype=torch.float32) + 0.5) / width
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    return ((grid_x >= x0) & (grid_x < x1) & (grid_y >= y0) & (grid_y < y1)).float()


def _boxdiff_mask_and_bounds(
    box: tuple[float, float, float, float],
    height: int,
    width: int,
    device: torch.device,
) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
    x0, y0, x1, y1 = box
    left = min(max(round(x0 * width), 0), width - 1)
    top = min(max(round(y0 * height), 0), height - 1)
    right = min(max(round(x1 * width), left + 1), width)
    bottom = min(max(round(y1 * height), top + 1), height)
    mask = torch.zeros((height, width), device=device, dtype=torch.float32)
    mask[top:bottom, left:right] = 1.0
    return mask, (left, top, right, bottom)


def _corner_mask(
    length: int,
    first: int,
    second: int,
    radius: int,
    device: torch.device,
) -> torch.Tensor:
    mask = torch.zeros(length, device=device, dtype=torch.float32)
    for coordinate in (first, second):
        start = max(coordinate - radius, 0)
        end = min(coordinate + radius + 1, length)
        mask[start:end] = 1.0
    return mask


class LayoutEnergy:
    def __init__(self, config: EnergyConfig):
        if not config.layers:
            raise ValueError("At least one attention layer is required")
        if (
            config.inside_weight < 0.0
            or config.cross_weight < 0.0
            or config.relation_weight < 0.0
            or config.relation_dominance_weight < 0.0
            or config.entropy_weight < 0.0
            or config.presence_weight < 0.0
            or config.boxdiff_inner_weight < 0.0
            or config.boxdiff_outer_weight < 0.0
            or config.boxdiff_corner_weight < 0.0
        ):
            raise ValueError("Energy weights must be non-negative")
        if not 0.0 <= config.relation_margin <= 1.0:
            raise ValueError("relation_margin must be in [0, 1]")
        if config.relation_dominance_ratio < 0.0:
            raise ValueError("relation_dominance_ratio must be non-negative")
        if (
            config.relation_dominance_weight > 0.0
            and config.relation_weight == 0.0
        ):
            raise ValueError(
                "relation_dominance_weight requires a positive relation_weight"
            )
        if not 0.0 <= config.entropy_target <= 1.0:
            raise ValueError("entropy_target must be in [0, 1]")
        if config.presence_min < 0.0:
            raise ValueError("presence_min must be non-negative")
        if config.presence_weight > 0.0 and config.presence_min <= 0.0:
            raise ValueError("presence_weight requires a positive presence_min")
        if not 0.0 < config.boxdiff_p <= 1.0:
            raise ValueError("boxdiff_p must be in (0, 1]")
        if config.boxdiff_corner_radius < 0:
            raise ValueError("boxdiff_corner_radius must be non-negative")
        if config.boxdiff_text_scale <= 0.0:
            raise ValueError("boxdiff_text_scale must be positive")
        if config.boxdiff_token_mode not in {"phrase_mean", "per_token"}:
            raise ValueError(
                "boxdiff_token_mode must be 'phrase_mean' or 'per_token'"
            )
        if config.attention_normalization not in {
            "text_conditional",
            "legacy_boxdiff",
        }:
            raise ValueError(
                "attention_normalization must be 'text_conditional' or "
                "'legacy_boxdiff'"
            )
        if config.text_gate_power < 0.0:
            raise ValueError("text_gate_power must be non-negative")
        if not any(
            weight > 0.0
            for weight in (
                config.inside_weight,
                config.cross_weight,
                config.relation_weight,
                config.entropy_weight,
                config.presence_weight,
                config.boxdiff_inner_weight,
                config.boxdiff_outer_weight,
                config.boxdiff_corner_weight,
            )
        ):
            raise ValueError("At least one energy weight must be positive")
        self.config = config
        self.boxdiff_enabled = any(
            weight > 0.0
            for weight in (
                config.boxdiff_inner_weight,
                config.boxdiff_outer_weight,
                config.boxdiff_corner_weight,
            )
        )

    @staticmethod
    def _relation_delta(
        relation: str,
        centroids: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        if len(centroids) != 2:
            raise ValueError("Relative layout energy requires exactly two objects")
        (first_x, first_y), (second_x, second_y) = centroids
        return {
            "left": second_x - first_x,
            "right": first_x - second_x,
            "above": second_y - first_y,
            "below": first_y - second_y,
        }[relation]

    @staticmethod
    def _orthogonal_delta(
        relation: str,
        centroids: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        if len(centroids) != 2:
            raise ValueError("Relative layout energy requires exactly two objects")
        (first_x, first_y), (second_x, second_y) = centroids
        if relation in {"left", "right"}:
            return (first_y - second_y).abs()
        return (first_x - second_x).abs()

    def _selected_layers(
        self,
        attentions: Sequence[torch.Tensor | None],
    ) -> list[tuple[int, torch.Tensor]]:
        selected = []
        layer_count = len(attentions)
        for raw_index in self.config.layers:
            index = raw_index if raw_index >= 0 else layer_count + raw_index
            if not 0 <= index < layer_count:
                raise IndexError(
                    f"Attention layer {raw_index} is outside a model with {layer_count} layers"
                )
            value = attentions[index]
            if value is None:
                raise RuntimeError(
                    f"Attention layer {index} is unavailable; enable eager output_attentions"
                )
            selected.append((index, value))
        return selected

    def _text_attention_maps(
        self,
        selected: Sequence[tuple[int, torch.Tensor]],
        batch_index: int,
        image_start: int,
        image_end: int,
        text_end: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        text_start = 1  # Skip BOS, matching BoxDiff's removal of the first token.
        if text_end <= text_start:
            raise ValueError("Layout energy requires at least one prompt token")
        layout_maps = []
        boxdiff_maps = []
        raw_maps = []
        gates = []
        for _, layer_attention in selected:
            text_attention = layer_attention[
                batch_index, :, image_start:image_end, text_start:text_end
            ].float()
            text_gate = text_attention.sum(dim=-1)
            if self.config.attention_normalization == "text_conditional":
                # Qwen self-attention is normalized over text and image keys together.
                # Renormalizing the text slice recovers p(text token | prompt text,
                # image query) instead of applying BoxDiff's second softmax to tiny
                # post-softmax probabilities.
                conditional = text_attention / text_gate.unsqueeze(-1).clamp_min(
                    self.config.eps
                )
                layout_map = conditional
                boxdiff_map = conditional
            else:
                # Preserve the original layout-energy behavior while retaining
                # BoxDiff's post-softmax text competition for legacy runs.
                layout_map = text_attention
                boxdiff_map = torch.softmax(
                    text_attention * self.config.boxdiff_text_scale,
                    dim=-1,
                )
            layout_maps.append(layout_map.mean(dim=0))
            boxdiff_maps.append(boxdiff_map.mean(dim=0))
            raw_maps.append(text_attention.mean(dim=0))
            gates.append(text_gate.mean(dim=0))
        layout_attention_for_text = torch.stack(layout_maps).mean(dim=0)
        boxdiff_attention_for_text = torch.stack(boxdiff_maps).mean(dim=0)
        raw_attention_for_text = torch.stack(raw_maps).mean(dim=0)
        text_gate = torch.stack(gates).mean(dim=0)
        if self.config.text_gate_power > 0.0:
            relative_gate = text_gate / text_gate.mean().clamp_min(self.config.eps)
            gate = relative_gate.clamp(max=5.0).pow(
                self.config.text_gate_power
            ).unsqueeze(-1)
            layout_attention_for_text = layout_attention_for_text * gate
            boxdiff_attention_for_text = boxdiff_attention_for_text * gate
        return (
            layout_attention_for_text,
            boxdiff_attention_for_text,
            raw_attention_for_text,
            text_gate,
        )

    def _entity_attention_maps(
        self,
        attention_for_text: torch.Tensor,
        token_indices: torch.Tensor,
        text_end: int,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        text_start = 1
        if bool(((token_indices < text_start) | (token_indices >= text_end)).any()):
            raise ValueError(
                f"Entity token indices must be within [{text_start}, {text_end})"
            )
        relative_indices = token_indices - text_start
        per_token = [
            attention_for_text[:, index] for index in relative_indices.tolist()
        ]
        # Summing token probabilities gives a phrase probability. Its later spatial
        # normalization makes this invariant to phrase token count for centroids.
        return torch.stack(per_token).sum(dim=0), per_token

    @staticmethod
    def _aggregate_boxdiff_terms(
        terms: Sequence[dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        if not terms:
            raise ValueError("At least one entity token is required")
        activation_keys = ("inner_activation", "outer_activation")
        loss_keys = (
            "inner_loss",
            "outer_loss",
            "corner_x_loss",
            "corner_y_loss",
            "corner_loss",
        )
        return {
            **{
                key: torch.stack([item[key] for item in terms]).mean()
                for key in activation_keys
            },
            **{
                # Official BoxDiff constrains each selected entity token.
                key: torch.stack([item[key] for item in terms]).sum()
                for key in loss_keys
            },
        }

    def _boxdiff_terms(
        self,
        attention_map: torch.Tensor,
        box: tuple[float, float, float, float],
        height: int,
        width: int,
    ) -> dict[str, torch.Tensor]:
        image = attention_map.reshape(height, width)
        box_mask, (left, top, right, bottom) = _boxdiff_mask_and_bounds(
            box,
            height,
            width,
            image.device,
        )
        outside_mask = 1.0 - box_mask
        inside_k = max(1, round(float(box_mask.sum().item()) * self.config.boxdiff_p))
        outside_k = max(
            1,
            round(float(outside_mask.sum().item()) * self.config.boxdiff_p),
        )
        inner_activation = (image * box_mask).reshape(-1).topk(inside_k).values.mean()
        outer_activation = (
            (image * outside_mask).reshape(-1).topk(outside_k).values.mean()
        )
        inner_loss = torch.relu(inner_activation.new_tensor(1.0) - inner_activation)
        outer_loss = torch.relu(outer_activation)

        target_x = box_mask.amax(dim=0)
        target_y = box_mask.amax(dim=1)
        predicted_x = image.amax(dim=0)
        predicted_y = image.amax(dim=1)
        corner_mask_x = _corner_mask(
            width,
            left,
            right,
            self.config.boxdiff_corner_radius,
            image.device,
        )
        corner_mask_y = _corner_mask(
            height,
            top,
            bottom,
            self.config.boxdiff_corner_radius,
            image.device,
        )
        corner_x_loss = ((predicted_x - target_x).abs() * corner_mask_x).mean()
        corner_y_loss = ((predicted_y - target_y).abs() * corner_mask_y).mean()
        return {
            "inner_activation": inner_activation,
            "outer_activation": outer_activation,
            "inner_loss": inner_loss,
            "outer_loss": outer_loss,
            "corner_x_loss": corner_x_loss,
            "corner_y_loss": corner_y_loss,
            "corner_loss": corner_x_loss + corner_y_loss,
        }

    def evaluate(
        self,
        attentions: Sequence[torch.Tensor | None],
        layouts: Sequence[LayoutSpec],
        modality_positions: torch.Tensor,
        spatial_shape: tuple[int, int],
        add_time_embeds: bool,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        list[dict[str, torch.Tensor | str | int]],
    ]:
        height, width = spatial_shape
        expected_tokens = height * width
        selected = self._selected_layers(attentions)
        sample_energies = []
        details = []

        if len(layouts) > modality_positions.shape[0]:
            raise ValueError("There are more layouts than conditional batch entries")

        for batch_index, layout in enumerate(layouts):
            offset, length = (
                int(value.item()) for value in modality_positions[batch_index, 0]
            )
            image_start = offset + (1 if add_time_embeds else 0)
            image_count = length - (1 if add_time_embeds else 0)
            if image_count != expected_tokens:
                raise ValueError(
                    f"Expected {expected_tokens} spatial image tokens, got {image_count}"
                )
            image_end = image_start + image_count
            text_end = offset - 1  # BOI is immediately before the time/image block.
            (
                layout_attention_for_text,
                boxdiff_attention_for_text,
                raw_attention_for_text,
                text_gate,
            ) = self._text_attention_maps(
                selected,
                batch_index,
                image_start,
                image_end,
                text_end,
            )
            masks = [
                _box_mask(item.box, height, width, modality_positions.device).reshape(
                    -1
                )
                for item in layout.objects
            ]

            xs = (
                (
                    torch.arange(
                        width, device=modality_positions.device, dtype=torch.float32
                    )
                    + 0.5
                )
                / width
            ).repeat(height)
            ys = (
                (
                    torch.arange(
                        height, device=modality_positions.device, dtype=torch.float32
                    )
                    + 0.5
                )
                / height
            ).repeat_interleave(width)
            centroids = []
            layout_details = []
            layout_object_energies = []
            layout_boxdiff_terms = []
            for object_index, item in enumerate(layout.objects):
                token_indices = torch.tensor(
                    item.token_indices,
                    device=modality_positions.device,
                    dtype=torch.long,
                )
                for _, layer_attention in selected:
                    if layer_attention.ndim != 4:
                        raise ValueError(
                            "Attention tensors must have shape [batch, heads, query, key]"
                        )
                attention_map, per_token_maps = self._entity_attention_maps(
                    layout_attention_for_text,
                    token_indices,
                    text_end,
                )
                _, boxdiff_per_token_maps = self._entity_attention_maps(
                    boxdiff_attention_for_text,
                    token_indices,
                    text_end,
                )
                _raw_attention_map, raw_per_token_maps = self._entity_attention_maps(
                    raw_attention_for_text,
                    token_indices,
                    text_end,
                )
                total_mass = attention_map.sum().clamp_min(self.config.eps)
                normalized_map = attention_map / total_mass
                normalized_entropy = -(
                    normalized_map
                    * normalized_map.clamp_min(self.config.eps).log()
                ).sum() / normalized_map.new_tensor(float(expected_tokens)).log()
                entropy_violation = torch.relu(
                    normalized_entropy
                    - normalized_entropy.new_tensor(self.config.entropy_target)
                )
                entropy_energy = entropy_violation.square()
                entity_presence = torch.stack(raw_per_token_maps).mean()
                if self.config.presence_min > 0.0:
                    presence_ratio = entity_presence / entity_presence.new_tensor(
                        self.config.presence_min
                    )
                    presence_violation = torch.relu(
                        presence_ratio.new_tensor(1.0) - presence_ratio
                    )
                else:
                    presence_ratio = entity_presence.new_zeros(())
                    presence_violation = entity_presence.new_zeros(())
                presence_energy = presence_violation.square()
                centroid_x = (normalized_map * xs).sum()
                centroid_y = (normalized_map * ys).sum()
                centroids.append((centroid_x, centroid_y))
                inside_ratio = (normalized_map * masks[object_index]).sum()
                inside_energy = (1.0 - inside_ratio).square()

                cross_ratio = attention_map.new_zeros(())
                if len(masks) > 1:
                    other_mask = torch.stack(
                        [
                            mask
                            for index, mask in enumerate(masks)
                            if index != object_index
                        ]
                    ).amax(dim=0)
                    cross_ratio = (normalized_map * other_mask).sum()

                object_energy = (
                    self.config.inside_weight * inside_energy
                    + self.config.cross_weight * cross_ratio
                    + self.config.entropy_weight * entropy_energy
                    + self.config.presence_weight * presence_energy
                )
                if self.boxdiff_enabled:
                    if self.config.boxdiff_token_mode == "per_token":
                        boxdiff_maps = boxdiff_per_token_maps
                    else:
                        boxdiff_maps = [
                            torch.stack(boxdiff_per_token_maps).mean(dim=0)
                        ]
                    boxdiff = self._aggregate_boxdiff_terms(
                        [
                            self._boxdiff_terms(
                                token_map,
                                item.box,
                                height,
                                width,
                            )
                            for token_map in boxdiff_maps
                        ]
                    )
                else:
                    zero = attention_map.new_zeros(())
                    boxdiff = {
                        "inner_activation": zero,
                        "outer_activation": zero,
                        "inner_loss": zero,
                        "outer_loss": zero,
                        "corner_x_loss": zero,
                        "corner_y_loss": zero,
                        "corner_loss": zero,
                    }
                weighted_boxdiff = (
                    self.config.boxdiff_inner_weight * boxdiff["inner_loss"]
                    + self.config.boxdiff_outer_weight * boxdiff["outer_loss"]
                    + self.config.boxdiff_corner_weight * boxdiff["corner_loss"]
                )
                layout_boxdiff_terms.append(weighted_boxdiff)
                layout_object_energies.append(object_energy)
                detail = {
                    "batch_index": batch_index,
                    "object_index": object_index,
                    "phrase": item.phrase,
                    "inside_ratio": inside_ratio,
                    "cross_ratio": cross_ratio,
                    "centroid_x": centroid_x,
                    "centroid_y": centroid_y,
                    "text_gate_mean": text_gate.mean(),
                    "spatial_entropy": normalized_entropy,
                    "entropy_violation": entropy_violation,
                    "entropy_energy": entropy_energy,
                    "entity_presence": entity_presence,
                    "presence_ratio": presence_ratio,
                    "presence_violation": presence_violation,
                    "presence_energy": presence_energy,
                    **boxdiff,
                    "legacy_energy": object_energy,
                    "boxdiff_energy": weighted_boxdiff,
                    "energy": object_energy + weighted_boxdiff,
                }
                details.append(detail)
                layout_details.append(detail)

            if self.config.relation_weight > 0.0:
                if layout.relation not in {"left", "right", "above", "below"}:
                    raise ValueError(
                        f"Unsupported relative layout relation: {layout.relation!r}"
                    )
                signed_delta = self._relation_delta(layout.relation, centroids)
                orthogonal_delta = self._orthogonal_delta(layout.relation, centroids)
                violation = torch.relu(
                    signed_delta.new_tensor(self.config.relation_margin) - signed_delta
                )
                dominance_violation = torch.relu(
                    self.config.relation_dominance_ratio * orthogonal_delta
                    - signed_delta
                )
                relation_energy = (
                    violation.square()
                    + self.config.relation_dominance_weight
                    * dominance_violation.square()
                )
                for detail in layout_details:
                    detail["relation"] = layout.relation
                    detail["relation_signed_delta"] = signed_delta
                    detail["relation_orthogonal_delta"] = orthogonal_delta
                    detail["relation_violation"] = violation
                    detail["relation_dominance_violation"] = dominance_violation
                    detail["relation_energy"] = relation_energy

            sample_energy = torch.stack(layout_object_energies).mean()
            sample_energy = sample_energy + torch.stack(layout_boxdiff_terms).sum()
            if self.config.relation_weight > 0.0:
                sample_energy = (
                    sample_energy + self.config.relation_weight * relation_energy
                )
            sample_energies.append(sample_energy)

        if not sample_energies:
            raise ValueError("No layout objects were provided")
        per_sample_energy = torch.stack(sample_energies)
        return per_sample_energy.mean(), per_sample_energy, details

    def __call__(
        self,
        attentions: Sequence[torch.Tensor | None],
        layouts: Sequence[LayoutSpec],
        modality_positions: torch.Tensor,
        spatial_shape: tuple[int, int],
        add_time_embeds: bool,
    ) -> tuple[torch.Tensor, list[dict[str, torch.Tensor | str | int]]]:
        energy, _per_sample_energy, details = self.evaluate(
            attentions,
            layouts,
            modality_positions,
            spatial_shape,
            add_time_embeds,
        )
        return energy, details
