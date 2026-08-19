"""Differentiable truncated Euler rollouts for flow-alignment training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


VelocityFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class EulerRollout:
    """States and predictions from consecutive velocity evaluations."""

    times: torch.Tensor
    states: tuple[torch.Tensor, ...]
    velocities: tuple[torch.Tensor, ...]
    predicted_cleans: tuple[torch.Tensor, ...]

    @property
    def terminal_clean(self) -> torch.Tensor:
        return self.predicted_cleans[-1]

    @property
    def transition_count(self) -> int:
        return max(0, len(self.states) - 1)


def select_rollout_times(
    time_grid: torch.Tensor,
    rollout_steps: int,
    t_min: float,
    t_max: float,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Select a consecutive window whose first time is in the train range."""
    if time_grid.ndim != 1:
        raise ValueError(f"time_grid must be one-dimensional, got {tuple(time_grid.shape)}")
    if time_grid.numel() < 2:
        raise ValueError("time_grid must contain at least two points")
    if rollout_steps < 1:
        raise ValueError("rollout_steps must be positive")
    if rollout_steps > time_grid.numel():
        raise ValueError("rollout_steps exceeds the available time grid")
    if not 0.0 <= t_min < t_max <= 1.0:
        raise ValueError("expected 0 <= t_min < t_max <= 1")
    if not bool(torch.all(time_grid[1:] > time_grid[:-1])):
        raise ValueError("time_grid must be strictly increasing")

    latest_start = time_grid.numel() - rollout_steps
    start_times = time_grid[: latest_start + 1]
    candidates = torch.nonzero(
        (start_times >= t_min) & (start_times <= t_max), as_tuple=False
    ).flatten()
    if candidates.numel() == 0:
        raise ValueError(
            f"no rollout window starts in [{t_min}, {t_max}] for "
            f"K={rollout_steps} and {time_grid.numel()} schedule points"
        )
    selected = torch.randint(
        candidates.numel(),
        (1,),
        generator=generator,
        device="cpu",
    )
    start = int(candidates.cpu()[selected].item())
    return time_grid[start : start + rollout_steps]


def differentiable_euler_rollout(
    initial_state: torch.Tensor,
    times: torch.Tensor,
    velocity_fn: VelocityFn,
) -> EulerRollout:
    """Run K velocity evaluations with K-1 recurrent Euler transitions.

    The Show-o2 linear-flow convention runs from noise at t=0 toward data at
    t=1. At every visited state, the clean prediction is
    ``x_t + (1 - t) * v_theta(x_t, t)``.
    """
    if initial_state.ndim < 2:
        raise ValueError("initial_state must include batch and feature dimensions")
    if times.ndim != 1 or times.numel() < 1:
        raise ValueError("times must be a non-empty one-dimensional tensor")
    if times.numel() > 1 and not bool(torch.all(times[1:] > times[:-1])):
        raise ValueError("rollout times must be strictly increasing")

    state = initial_state
    states: list[torch.Tensor] = []
    velocities: list[torch.Tensor] = []
    predicted_cleans: list[torch.Tensor] = []
    for index, scalar_time in enumerate(times):
        states.append(state)
        timestep = torch.full(
            (state.shape[0],),
            float(scalar_time.detach().cpu()),
            device=state.device,
            dtype=torch.float32,
        )
        velocity = velocity_fn(state, timestep)
        if velocity.shape != state.shape:
            raise ValueError(
                f"velocity shape {tuple(velocity.shape)} does not match "
                f"state shape {tuple(state.shape)}"
            )
        velocity = velocity.float()
        clean = state.float() + (1.0 - timestep).view(
            -1, *([1] * (state.ndim - 1))
        ) * velocity
        velocities.append(velocity)
        predicted_cleans.append(clean)
        if index + 1 < times.numel():
            dt = (times[index + 1] - scalar_time).float()
            state = state.float() + dt * velocity

    return EulerRollout(
        times=times,
        states=tuple(states),
        velocities=tuple(velocities),
        predicted_cleans=tuple(predicted_cleans),
    )

