from __future__ import annotations

import torch

from evaluation.c6_mask2former_alignment.trajectory_rollout import (
    differentiable_euler_rollout,
    select_rollout_times,
)


def test_select_rollout_times_returns_consecutive_deterministic_window() -> None:
    grid = torch.linspace(0.0, 1.0, 50)
    generator_a = torch.Generator().manual_seed(17)
    generator_b = torch.Generator().manual_seed(17)
    selected_a = select_rollout_times(
        grid, 4, 0.1, 0.8, generator=generator_a
    )
    selected_b = select_rollout_times(
        grid, 4, 0.1, 0.8, generator=generator_b
    )

    assert torch.equal(selected_a, selected_b)
    assert selected_a.shape == (4,)
    assert 0.1 <= float(selected_a[0]) <= 0.8
    assert torch.allclose(selected_a[1:] - selected_a[:-1], grid[1] - grid[0])


def test_constant_linear_velocity_tracks_teacher_path_and_clean() -> None:
    clean = torch.tensor([[[[3.0]]]])
    noise = torch.tensor([[[[-1.0]]]])
    velocity = clean - noise
    times = torch.tensor([0.2, 0.3, 0.5, 0.7])
    initial = times[0] * clean + (1.0 - times[0]) * noise

    rollout = differentiable_euler_rollout(
        initial,
        times,
        lambda state, timestep: velocity.expand_as(state),
    )

    assert rollout.transition_count == 3
    for state, timestep in zip(rollout.states, times):
        expected_state = timestep * clean + (1.0 - timestep) * noise
        assert torch.allclose(state, expected_state)
    for predicted_clean in rollout.predicted_cleans:
        assert torch.allclose(predicted_clean, clean)


def test_terminal_loss_backpropagates_through_recurrent_states() -> None:
    scale = torch.nn.Parameter(torch.tensor(0.5))
    initial = torch.zeros((1, 1, 1, 1))
    times = torch.tensor([0.1, 0.2, 0.3, 0.4])

    rollout = differentiable_euler_rollout(
        initial,
        times,
        lambda state, timestep: scale * (state + 1.0),
    )
    rollout.terminal_clean.square().mean().backward()

    assert scale.grad is not None
    assert torch.isfinite(scale.grad)
    assert float(scale.grad.abs()) > 0.0


def test_k1_matches_existing_predicted_clean_formula() -> None:
    state = torch.randn((1, 2, 3, 3))
    velocity = torch.randn_like(state)
    timestep = torch.tensor([0.37])

    rollout = differentiable_euler_rollout(
        state,
        timestep,
        lambda current, current_timestep: velocity,
    )
    expected = state.float() + (1.0 - timestep).view(-1, 1, 1, 1) * velocity.float()

    assert rollout.transition_count == 0
    assert torch.allclose(rollout.terminal_clean, expected)

