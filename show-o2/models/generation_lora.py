"""Generation-scoped LoRA injection for the local Qwen backbone."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F


LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")


class LoRALinear(nn.Module):
    """Frozen linear layer with a switchable zero-initialized LoRA branch."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        if rank < 1:
            raise ValueError("LoRA rank must be positive")
        if alpha <= 0.0:
            raise ValueError("LoRA alpha must be positive")
        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.lora_a = nn.Linear(base.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, base.out_features, bias=False)
        self.lora_a.to(device=base.weight.device, dtype=base.weight.dtype)
        self.lora_b.to(device=base.weight.device, dtype=base.weight.dtype)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=5 ** 0.5)
        nn.init.zeros_(self.lora_b.weight)
        self.enabled = True
        self.base.requires_grad_(False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = self.base(inputs)
        if not self.enabled:
            return output
        update = F.linear(F.linear(inputs, self.lora_a.weight), self.lora_b.weight)
        return output + update * self.scaling


@dataclass(frozen=True)
class LoRAParameterBudget:
    rank: int
    trainable_parameters: int
    target_parameters: int

    @property
    def relative_difference(self) -> float:
        return abs(self.trainable_parameters - self.target_parameters) / self.target_parameters


def lora_parameters_per_rank(qwen_model: nn.Module) -> int:
    total = 0
    for layer in qwen_model.model.layers:
        attention = layer.self_attn
        for target in LORA_TARGETS:
            linear = getattr(attention, target)
            if not isinstance(linear, nn.Linear):
                raise TypeError(f"expected nn.Linear at self_attn.{target}, got {type(linear)}")
            total += linear.in_features + linear.out_features
    return total


def matched_lora_budget(qwen_model: nn.Module, target_parameters: int) -> LoRAParameterBudget:
    if target_parameters < 1:
        raise ValueError("target parameter count must be positive")
    per_rank = lora_parameters_per_rank(qwen_model)
    rank = max(1, round(target_parameters / per_rank))
    return LoRAParameterBudget(
        rank=rank,
        trainable_parameters=rank * per_rank,
        target_parameters=target_parameters,
    )


class GenerationLoRAController:
    """Inject and route LoRA only for generation forward passes."""

    def __init__(self, qwen_model: nn.Module, rank: int, alpha: float | None = None):
        self.qwen_model = qwen_model
        self.rank = int(rank)
        self.alpha = float(alpha if alpha is not None else rank)
        self._wrappers: dict[str, LoRALinear] = {}
        for layer_index, layer in enumerate(qwen_model.model.layers):
            attention = layer.self_attn
            for target in LORA_TARGETS:
                base = getattr(attention, target)
                if isinstance(base, LoRALinear):
                    raise RuntimeError(f"LoRA is already installed at layer {layer_index} {target}")
                wrapper = LoRALinear(base, self.rank, self.alpha)
                setattr(attention, target, wrapper)
                self._wrappers[f"layers.{layer_index}.self_attn.{target}"] = wrapper

    def parameters(self) -> Iterator[nn.Parameter]:
        for wrapper in self._wrappers.values():
            yield from wrapper.lora_a.parameters()
            yield from wrapper.lora_b.parameters()

    def named_parameters(self) -> Iterator[tuple[str, nn.Parameter]]:
        for name, wrapper in self._wrappers.items():
            yield f"{name}.lora_a.weight", wrapper.lora_a.weight
            yield f"{name}.lora_b.weight", wrapper.lora_b.weight

    def train(self, mode: bool = True) -> None:
        for wrapper in self._wrappers.values():
            wrapper.lora_a.train(mode)
            wrapper.lora_b.train(mode)

    def requires_grad_(self, requires_grad: bool = True) -> None:
        for parameter in self.parameters():
            parameter.requires_grad_(requires_grad)

    def set_enabled(self, enabled: bool) -> None:
        for wrapper in self._wrappers.values():
            wrapper.enabled = bool(enabled)

    @contextmanager
    def enabled(self, enabled: bool):
        previous = [wrapper.enabled for wrapper in self._wrappers.values()]
        self.set_enabled(enabled)
        try:
            yield
        finally:
            for wrapper, value in zip(self._wrappers.values(), previous):
                wrapper.enabled = value

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {name: parameter.detach().cpu() for name, parameter in self.named_parameters()}

    def load_state_dict(self, state: Mapping[str, torch.Tensor], strict: bool = True) -> None:
        parameters = dict(self.named_parameters())
        missing = set(parameters) - set(state)
        unexpected = set(state) - set(parameters)
        if strict and (missing or unexpected):
            raise RuntimeError(
                f"LoRA state mismatch missing={sorted(missing)[:5]} "
                f"unexpected={sorted(unexpected)[:5]}"
            )
        with torch.no_grad():
            for name in set(parameters) & set(state):
                source, target = state[name], parameters[name]
                if source.shape != target.shape:
                    raise RuntimeError(
                        f"LoRA tensor shape mismatch for {name}: "
                        f"{tuple(source.shape)} != {tuple(target.shape)}"
                    )
                target.copy_(source.to(device=target.device, dtype=target.dtype))

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
