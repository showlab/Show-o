"""Staged objective composition for the frozen-Show-o2 DRSA adapter.

The schedule keeps the native velocity/MMU objectives active while spatial
grounding is introduced gradually. All terms are scalar tensors so gradients
remain connected to the route that produced each term.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch


_LOSS_NAMES = (
    "native_flow",
    "mmu",
    "guided_flow",
    "retention",
    "box_u",
    "box_g",
    "relation_u",
    "relation_g",
    "relation_code",
)


@dataclass(frozen=True)
class StageWeights:
    """Weights applied to one optimizer step."""

    stage: str
    native_flow: float
    mmu: float
    guided_flow: float
    retention: float
    box_u: float
    box_g: float
    relation_u: float
    relation_g: float
    relation_code: float
    progress: float = 1.0

    def as_dict(self) -> dict[str, float | str]:
        return {
            "stage": self.stage,
            "progress": self.progress,
            **{
                name: float(getattr(self, name))
                for name in _LOSS_NAMES
            },
        }


@dataclass(frozen=True)
class StagedAdapterLossConfig:
    """Three-phase loss schedule.

    Phase A (native_anchor) preserves the original model objectives.
    Phase B (spatial_ramp) linearly introduces the guided flow and BoxDiff
    terms. Phase C (joint_stabilize) keeps native flow/MMU and teacher
    retention active while using a conservative spatial weight.
    """

    native_steps: int = 500
    ramp_steps: int = 500
    native_flow_weight: float = 1.0
    mmu_weight: float = 0.01
    guided_flow_weight: float = 0.25
    retention_weight: float = 0.20
    box_u_weight: float = 0.05
    box_g_weight: float = 0.01
    relation_u_weight: float = 0.02
    relation_g_weight: float = 0.005
    relation_code_weight: float = 0.0

    def __post_init__(self) -> None:
        if self.native_steps < 0 or self.ramp_steps < 0:
            raise ValueError("native_steps and ramp_steps must be non-negative")
        values = (
            self.native_flow_weight,
            self.mmu_weight,
            self.guided_flow_weight,
            self.retention_weight,
            self.box_u_weight,
            self.box_g_weight,
            self.relation_u_weight,
            self.relation_g_weight,
            self.relation_code_weight,
        )
        if any(value < 0.0 for value in values):
            raise ValueError("loss weights must be non-negative")
        if self.native_steps == 0 and self.ramp_steps == 0:
            raise ValueError("at least one scheduled step is required")

    def weights_at(self, step: int) -> StageWeights:
        if step < 1:
            raise ValueError("step is one-based and must be positive")
        if step <= self.native_steps:
            return StageWeights(
                stage="native_anchor",
                native_flow=self.native_flow_weight,
                mmu=self.mmu_weight,
                guided_flow=0.0,
                retention=0.0,
                box_u=0.0,
                box_g=0.0,
                relation_u=0.0,
                relation_g=0.0,
                relation_code=0.0,
                progress=0.0,
            )
        if self.ramp_steps > 0 and step <= self.native_steps + self.ramp_steps:
            progress = float(step - self.native_steps) / float(self.ramp_steps)
            return StageWeights(
                stage="spatial_ramp",
                native_flow=self.native_flow_weight,
                mmu=self.mmu_weight,
                guided_flow=self.guided_flow_weight * progress,
                retention=self.retention_weight * progress,
                box_u=self.box_u_weight * progress,
                box_g=self.box_g_weight * progress,
                relation_u=self.relation_u_weight * progress,
                relation_g=self.relation_g_weight * progress,
                relation_code=self.relation_code_weight * progress,
                progress=progress,
            )
        return StageWeights(
            stage="joint_stabilize",
            native_flow=self.native_flow_weight,
            mmu=self.mmu_weight,
            guided_flow=self.guided_flow_weight,
            retention=self.retention_weight,
            box_u=self.box_u_weight,
            box_g=self.box_g_weight,
            relation_u=self.relation_u_weight,
            relation_g=self.relation_g_weight,
            relation_code=self.relation_code_weight,
            progress=1.0,
        )


def _zero_like_reference(losses: Mapping[str, torch.Tensor | None]) -> torch.Tensor:
    for value in losses.values():
        if value is not None:
            if value.ndim != 0:
                raise ValueError("every loss term must be a scalar tensor")
            return value.new_zeros((), dtype=torch.float32)
    return torch.zeros((), dtype=torch.float32)


def compose_staged_loss(
    losses: Mapping[str, torch.Tensor | None],
    weights: StageWeights,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compose weighted loss terms without detaching their computation graphs.

    Missing terms are treated as zero. This lets native-only steps omit
    grounding tensors and still use one stable training interface.
    """

    reference = _zero_like_reference(losses)
    components: dict[str, torch.Tensor] = {}
    total = reference
    for name in _LOSS_NAMES:
        value = losses.get(name)
        if value is None:
            value = reference
        elif value.ndim != 0:
            raise ValueError(f"{name} must be a scalar tensor")
        value = value.float()
        weighted = value * float(getattr(weights, name))
        components[name] = weighted
        total = total + weighted
    if not torch.isfinite(total):
        raise FloatingPointError(
            f"non-finite staged loss in {weights.stage}: {float(total.detach())}"
        )
    components["total"] = total
    return total, components
