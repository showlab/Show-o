"""Unified generation-only adaptation routes and checkpoint state."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, Mapping

import torch
import torch.nn as nn

from .generation_lora import GenerationLoRAController, matched_lora_budget
from .spatial_alignment_adapter import GenerationSpatialAdapter


ADAPTATION_ROUTES = ("g_adapter", "lora")
DEFAULT_PARAMETER_MATCH_TOLERANCE = 0.01


def generation_adapter_parameter_count(
    hidden_size: int,
    spatial_dim: int = 604,
    num_heads: int = 8,
    bias_hidden_dim: int = 128,
) -> int:
    feature_dim = 12
    return (
        4 * hidden_size * hidden_size
        + 3 * hidden_size
        + 3 * spatial_dim
        + spatial_dim * bias_hidden_dim
        + bias_hidden_dim
        + bias_hidden_dim * num_heads * feature_dim
        + num_heads * feature_dim
    )


@dataclass
class GenerationAdaptation:
    model: nn.Module
    route: str
    target_parameters: int
    g_adapter: nn.Module | None = None
    lora: GenerationLoRAController | None = None

    def parameters(self) -> Iterator[nn.Parameter]:
        if self.route == "g_adapter":
            if self.g_adapter is None:
                raise RuntimeError("g_adapter route has no module")
            yield from self.g_adapter.parameters()
        else:
            if self.lora is None:
                raise RuntimeError("LoRA route has no controller")
            yield from self.lora.parameters()

    def named_parameters(self) -> Iterator[tuple[str, nn.Parameter]]:
        if self.route == "g_adapter":
            if self.g_adapter is None:
                raise RuntimeError("g_adapter route has no module")
            for name, parameter in self.g_adapter.named_parameters():
                yield f"generation_adapter.{name}", parameter
        else:
            if self.lora is None:
                raise RuntimeError("LoRA route has no controller")
            yield from self.lora.named_parameters()

    def requires_grad_(self, requires_grad: bool = True) -> None:
        for parameter in self.parameters():
            parameter.requires_grad_(requires_grad)

    def train(self, mode: bool = True) -> None:
        if self.route == "g_adapter":
            assert self.g_adapter is not None
            self.g_adapter.train(mode)
        else:
            assert self.lora is not None
            self.lora.train(mode)

    def set_enabled(self, enabled: bool) -> None:
        if self.route == "g_adapter":
            self.model.generation_adapter_active = bool(enabled)
        else:
            assert self.lora is not None
            self.lora.set_enabled(enabled)

    @contextmanager
    def enabled(self, enabled: bool):
        if self.route == "g_adapter":
            previous = bool(self.model.generation_adapter_active)
            self.model.generation_adapter_active = bool(enabled)
            try:
                yield
            finally:
                self.model.generation_adapter_active = previous
        else:
            assert self.lora is not None
            with self.lora.enabled(enabled):
                yield

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def relative_parameter_difference(self) -> float:
        return abs(self.parameter_count - self.target_parameters) / self.target_parameters

    def state_dict(self) -> dict[str, torch.Tensor]:
        if self.route == "g_adapter":
            assert self.g_adapter is not None
            return {
                name: value.detach().cpu()
                for name, value in self.g_adapter.state_dict().items()
            }
        assert self.lora is not None
        return self.lora.state_dict()

    def load_state_dict(self, state: Mapping[str, torch.Tensor], strict: bool = True) -> None:
        if self.route == "g_adapter":
            assert self.g_adapter is not None
            self.g_adapter.load_state_dict(state, strict=strict)
        else:
            assert self.lora is not None
            self.lora.load_state_dict(state, strict=strict)

    def config_dict(self) -> dict[str, Any]:
        config: dict[str, Any] = {
            "route": self.route,
            "trainable_parameters": self.parameter_count,
            "matched_g_adapter_parameters": self.target_parameters,
            "relative_parameter_difference": self.relative_parameter_difference,
        }
        if self.route == "lora":
            assert self.lora is not None
            config.update({
                "rank": self.lora.rank,
                "alpha": self.lora.alpha,
                "targets": ["q_proj", "k_proj", "v_proj", "o_proj"],
            })
        else:
            assert self.g_adapter is not None
            config["spatial_dim"] = self.g_adapter.spatial.config.relation_dim
        return config


def configure_generation_adaptation(
    model: nn.Module,
    route: str,
    *,
    lora_rank: int | None = None,
    lora_alpha: float | None = None,
    parameter_match_tolerance: float = DEFAULT_PARAMETER_MATCH_TOLERANCE,
) -> GenerationAdaptation:
    if route not in ADAPTATION_ROUTES:
        raise ValueError(f"unknown generation adaptation route: {route!r}")
    if parameter_match_tolerance < 0.0:
        raise ValueError("parameter match tolerance must be non-negative")
    target_parameters = generation_adapter_parameter_count(int(model.config.hidden_size))
    if route == "g_adapter":
        if model.generation_adapter is None:
            raise RuntimeError(
                "g_adapter route requires generation_adapter_enabled=True at model construction"
            )
        adaptation = GenerationAdaptation(
            model=model,
            route=route,
            target_parameters=target_parameters,
            g_adapter=model.generation_adapter,
        )
    else:
        if model.generation_adapter is not None:
            raise RuntimeError("LoRA route must not construct a generation adapter")
        budget = matched_lora_budget(model.showo, target_parameters)
        rank = int(lora_rank if lora_rank is not None else budget.rank)
        controller = GenerationLoRAController(model.showo, rank, lora_alpha)
        adaptation = GenerationAdaptation(
            model=model,
            route=route,
            target_parameters=target_parameters,
            lora=controller,
        )
    if adaptation.relative_parameter_difference > parameter_match_tolerance:
        raise ValueError(
            f"{route} has {adaptation.parameter_count} trainable parameters versus "
            f"{target_parameters} for G adapter "
            f"({adaptation.relative_parameter_difference:.3%} difference > "
            f"{parameter_match_tolerance:.3%})"
        )
    return adaptation


def checkpoint_route_config(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    config = payload.get("adaptation_config")
    if not isinstance(config, Mapping) or config.get("route") not in ADAPTATION_ROUTES:
        raise ValueError("checkpoint has no valid generation adaptation config")
    return config


def checkpoint_model_kwargs(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return model-construction flags for legacy DRSA or new route states."""
    if "adaptation_config" in payload:
        config = checkpoint_route_config(payload)
        route = str(config["route"])
        return {
            "drsa_enabled": False,
            "generation_adapter_enabled": route == "g_adapter",
            "generation_adapter_spatial_dim": int(config.get("spatial_dim", 604)),
        }
    if "drsa_adapter" in payload:
        return {
            "drsa_enabled": True,
            "generation_adapter_enabled": False,
        }
    raise ValueError("checkpoint contains neither adaptation_state nor drsa_adapter")


def configure_from_checkpoint(
    model: nn.Module,
    payload: Mapping[str, Any],
    *,
    parameter_match_tolerance: float = DEFAULT_PARAMETER_MATCH_TOLERANCE,
) -> GenerationAdaptation:
    config = checkpoint_route_config(payload)
    route = str(config["route"])
    adaptation = configure_generation_adaptation(
        model,
        route,
        lora_rank=int(config["rank"]) if route == "lora" else None,
        lora_alpha=float(config["alpha"]) if route == "lora" else None,
        parameter_match_tolerance=parameter_match_tolerance,
    )
    state = payload.get("adaptation_state")
    if not isinstance(state, Mapping):
        raise ValueError("checkpoint has no generation adaptation state")
    adaptation.load_state_dict(state, strict=True)
    return adaptation
