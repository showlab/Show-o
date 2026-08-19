"""Inference-time latent guidance and fixed-step flow sampling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import torch

from .energy import LayoutEnergy
from .layout import LayoutSpec


@dataclass(frozen=True)
class GuiderConfig:
    inner_steps: int = 5
    step_size: float = 30.0
    max_grad_rms: float = 0.0
    max_flow_step_ratio: float = 0.0
    timestep_decay_power: float = 0.0
    attention_source: str = "qwen"
    backtracking_steps: int = 0
    backtracking_factor: float = 0.5
    min_relative_drop: float = 0.0
    lookahead_weight: float = 0.0
    presence_preservation_ratio: float = 0.0

    def __post_init__(self) -> None:
        if self.inner_steps < 1:
            raise ValueError("inner_steps must be positive")
        if self.step_size <= 0.0:
            raise ValueError("step_size must be positive")
        if self.max_grad_rms < 0.0:
            raise ValueError("max_grad_rms must be non-negative")
        if self.max_flow_step_ratio < 0.0:
            raise ValueError("max_flow_step_ratio must be non-negative")
        if self.timestep_decay_power < 0.0:
            raise ValueError("timestep_decay_power must be non-negative")
        if self.attention_source not in {"qwen", "diffusion"}:
            raise ValueError("attention_source must be 'qwen' or 'diffusion'")
        if self.backtracking_steps < 0:
            raise ValueError("backtracking_steps must be non-negative")
        if not 0.0 < self.backtracking_factor < 1.0:
            raise ValueError("backtracking_factor must be in (0, 1)")
        if self.min_relative_drop < 0.0:
            raise ValueError("min_relative_drop must be non-negative")
        if self.lookahead_weight < 0.0:
            raise ValueError("lookahead_weight must be non-negative")
        if not 0.0 <= self.presence_preservation_ratio <= 1.0:
            raise ValueError("presence_preservation_ratio must be in [0, 1]")
        if self.lookahead_weight > 0.0 and self.backtracking_steps == 0:
            raise ValueError("lookahead_weight requires backtracking_steps > 0")


def make_time_grid(
    transport: Any,
    num_steps: int,
    device: torch.device,
    time_shifting_factor: float | None,
    reverse: bool = False,
) -> torch.Tensor:
    if num_steps < 2:
        raise ValueError("num_steps must be at least 2")
    t0, t1 = transport.check_interval(
        transport.train_eps,
        transport.sample_eps,
        sde=False,
        eval=True,
        reverse=reverse,
        last_step_size=0.0,
    )
    times = torch.linspace(t0, t1, num_steps, device=device, dtype=torch.float32)
    if time_shifting_factor is not None:
        if time_shifting_factor <= 0.0:
            raise ValueError("time_shifting_factor must be positive")
        times = times / (times + time_shifting_factor - time_shifting_factor * times)
    return times


def _rms_per_sample(value: torch.Tensor) -> torch.Tensor:
    return value.float().flatten(1).square().mean(dim=1).sqrt()


def _mean_cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    left_flat = left.float().flatten(1)
    right_flat = right.float().flatten(1)
    denominator = (
        torch.linalg.vector_norm(left_flat, dim=1)
        * torch.linalg.vector_norm(right_flat, dim=1)
    ).clamp_min(1e-12)
    cosine = (left_flat * right_flat).sum(dim=1) / denominator
    return float(cosine.mean().detach().cpu())


def _batch_invariant_objective(energy: torch.Tensor, batch_size: int) -> torch.Tensor:
    if energy.ndim != 0:
        raise ValueError("Layout energy must be a scalar")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    return energy * batch_size


def _accept_candidate(
    score_before: torch.Tensor,
    score_after: torch.Tensor,
    min_relative_drop: float,
    presence_before: torch.Tensor,
    presence_after: torch.Tensor,
    presence_preservation_ratio: float,
    update_rms: torch.Tensor,
) -> torch.Tensor:
    required_drop = min_relative_drop * score_before.abs().clamp_min(1e-8)
    energy_ok = (score_before - score_after) >= required_drop
    presence_ok = presence_after >= (
        presence_preservation_ratio * presence_before
    )
    return energy_ok & presence_ok & (update_rms > 1e-12)


class LatentLayoutGuider:
    def __init__(
        self,
        model: torch.nn.Module,
        energy: LayoutEnergy,
        layouts: Sequence[LayoutSpec],
        modality_positions: torch.Tensor,
        spatial_shape: tuple[int, int],
        add_time_embeds: bool,
        config: GuiderConfig,
    ):
        if not layouts:
            raise ValueError("At least one layout is required")
        self.model = model
        self.energy = energy
        self.layouts = tuple(layouts)
        self.modality_positions = modality_positions
        self.spatial_shape = spatial_shape
        self.add_time_embeds = add_time_embeds
        self.config = config

    def _cfg_symmetric_gradient(
        self,
        conditional_gradient: torch.Tensor,
        latent_batch_size: int,
    ) -> torch.Tensor:
        conditional_batch_size = conditional_gradient.shape[0]
        if latent_batch_size == conditional_batch_size:
            return conditional_gradient
        if latent_batch_size != conditional_batch_size * 2:
            raise ValueError(
                f"Expected latent batch {conditional_batch_size} or "
                f"{conditional_batch_size * 2}, got {latent_batch_size}"
            )
        return torch.cat([conditional_gradient, conditional_gradient], dim=0)

    @staticmethod
    def _minimum_presence(
        details: Sequence[dict[str, Any]],
        batch_size: int,
    ) -> torch.Tensor:
        values = []
        for batch_index in range(batch_size):
            object_values = [
                item["entity_presence"]
                for item in details
                if int(item["batch_index"]) == batch_index
            ]
            if not object_values:
                raise ValueError(
                    f"No object presence values for batch index {batch_index}"
                )
            values.append(torch.stack(object_values).amin())
        return torch.stack(values)

    def _evaluate_latents(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        model_kwargs: dict[str, Any],
    ) -> tuple[torch.Tensor, list[dict[str, Any]]]:
        with torch.no_grad():
            _velocity, attentions = self.model.t2i_generate_with_attentions(
                image_latents=latents,
                t=timestep,
                attention_source=self.config.attention_source,
                **model_kwargs,
            )
            _energy, per_sample_energy, details = self.energy.evaluate(
                attentions=attentions,
                layouts=self.layouts,
                modality_positions=self.modality_positions,
                spatial_shape=self.spatial_shape,
                add_time_embeds=self.add_time_embeds,
            )
        return per_sample_energy.detach(), details

    def guide(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        dt: torch.Tensor,
        model_kwargs: dict[str, Any],
        reference_velocity: torch.Tensor | None = None,
        reference_velocity_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, list[dict[str, Any]]]:
        current = latents.detach()
        batch_size = len(self.layouts)
        records = []
        if reference_velocity is not None and reference_velocity.shape != latents.shape:
            raise ValueError(
                "reference_velocity must have the same shape as guided latents"
            )

        for inner_step in range(self.config.inner_steps):
            current.requires_grad_(True)
            with torch.enable_grad():
                velocity, attentions = self.model.t2i_generate_with_attentions(
                    image_latents=current,
                    t=timestep,
                    attention_source=self.config.attention_source,
                    **model_kwargs,
                )
                energy, per_sample_energy, object_details = self.energy.evaluate(
                    attentions=attentions,
                    layouts=self.layouts,
                    modality_positions=self.modality_positions,
                    spatial_shape=self.spatial_shape,
                    add_time_embeds=self.add_time_embeds,
                )
                gradient = torch.autograd.grad(
                    per_sample_energy.sum(),
                    current,
                    retain_graph=False,
                    create_graph=False,
                )[0]

            conditional_gradient = gradient[:batch_size]
            raw_grad_rms = _rms_per_sample(conditional_gradient)
            if self.config.max_grad_rms > 0.0:
                scale = (
                    self.config.max_grad_rms / raw_grad_rms.clamp_min(1e-12)
                ).clamp(max=1.0)
                conditional_gradient = conditional_gradient * scale.view(
                    -1, *([1] * (conditional_gradient.ndim - 1))
                )
            effective_gradient = self._cfg_symmetric_gradient(
                conditional_gradient,
                current.shape[0],
            )
            decay = (1.0 - timestep[:batch_size].float()).clamp(0.0, 1.0).pow(
                self.config.timestep_decay_power
            )
            decay_expanded = decay.repeat(
                current.shape[0] // batch_size
            )
            raw_update = (
                self.config.step_size
                * effective_gradient
                * decay_expanded.view(
                    -1, *([1] * (effective_gradient.ndim - 1))
                )
            )
            update = raw_update
            if reference_velocity_fn is not None:
                flow_velocity = reference_velocity_fn(current)
            else:
                flow_velocity = (
                    reference_velocity if reference_velocity is not None else velocity
                )
            flow_step_rms = (
                abs(float(dt.detach().cpu()))
                * _rms_per_sample(flow_velocity[:batch_size])
            )
            trust_scale = torch.ones_like(raw_grad_rms)
            if self.config.max_flow_step_ratio > 0.0:
                # Divide the step budget across inner iterations so the total
                # latent correction stays comparable to one Euler displacement.
                target_update_rms = (
                    self.config.max_flow_step_ratio
                    * flow_step_rms
                    / self.config.inner_steps
                )
                conditional_update_rms = _rms_per_sample(raw_update[:batch_size])
                trust_scale = (
                    target_update_rms
                    / conditional_update_rms.clamp_min(1e-12)
                ).clamp(max=1.0)
                conditional_update = raw_update[:batch_size] * trust_scale.view(
                    -1, *([1] * (raw_update.ndim - 1))
                )
                update = self._cfg_symmetric_gradient(
                    conditional_update,
                    current.shape[0],
                )
            presence_before = self._minimum_presence(object_details, batch_size)
            accepted = torch.ones(batch_size, dtype=torch.bool, device=current.device)
            selected_factor = torch.ones(batch_size, device=current.device)
            energy_before = per_sample_energy.detach()
            energy_after = energy_before
            lookahead_before = torch.zeros_like(energy_before)
            lookahead_after = torch.zeros_like(energy_before)
            score_before = energy_before
            score_after = energy_before
            presence_after = presence_before.detach()
            backtracking_attempts = 0

            if self.config.backtracking_steps > 0:
                if reference_velocity_fn is None:
                    raise RuntimeError(
                        "Backtracking guidance requires the official sampling "
                        "velocity function"
                    )
                if self.config.lookahead_weight > 0.0:
                    lookahead_latents = current + dt * flow_velocity
                    lookahead_before, _lookahead_details = self._evaluate_latents(
                        lookahead_latents,
                        timestep + dt,
                        model_kwargs,
                    )
                score_before = energy_before + (
                    self.config.lookahead_weight * lookahead_before
                )
                accepted = torch.zeros(
                    batch_size, dtype=torch.bool, device=current.device
                )
                selected_factor = torch.zeros(batch_size, device=current.device)
                next_current = current.detach()
                energy_after = energy_before.clone()
                lookahead_after = lookahead_before.clone()
                score_after = score_before.clone()
                presence_after = presence_before.detach().clone()

                for attempt in range(self.config.backtracking_steps):
                    factor = self.config.backtracking_factor**attempt
                    candidate_update = update * factor
                    candidate = (current - candidate_update).detach()
                    candidate_energy, candidate_details = self._evaluate_latents(
                        candidate,
                        timestep,
                        model_kwargs,
                    )
                    candidate_presence = self._minimum_presence(
                        candidate_details, batch_size
                    ).detach()
                    candidate_lookahead = torch.zeros_like(candidate_energy)
                    if self.config.lookahead_weight > 0.0:
                        candidate_velocity = reference_velocity_fn(candidate)
                        candidate_next = candidate + dt * candidate_velocity
                        candidate_lookahead, _candidate_lookahead_details = (
                            self._evaluate_latents(
                                candidate_next,
                                timestep + dt,
                                model_kwargs,
                            )
                        )
                    candidate_score = candidate_energy + (
                        self.config.lookahead_weight * candidate_lookahead
                    )
                    candidate_update_rms = _rms_per_sample(
                        candidate_update[:batch_size]
                    )
                    newly_accepted = (~accepted) & _accept_candidate(
                        score_before,
                        candidate_score,
                        self.config.min_relative_drop,
                        presence_before.detach(),
                        candidate_presence,
                        self.config.presence_preservation_ratio,
                        candidate_update_rms,
                    )
                    expanded_acceptance = newly_accepted.repeat(
                        current.shape[0] // batch_size
                    ).view(-1, *([1] * (current.ndim - 1)))
                    next_current = torch.where(
                        expanded_acceptance,
                        candidate,
                        next_current,
                    )
                    energy_after = torch.where(
                        newly_accepted, candidate_energy, energy_after
                    )
                    lookahead_after = torch.where(
                        newly_accepted, candidate_lookahead, lookahead_after
                    )
                    score_after = torch.where(
                        newly_accepted, candidate_score, score_after
                    )
                    presence_after = torch.where(
                        newly_accepted, candidate_presence, presence_after
                    )
                    selected_factor = torch.where(
                        newly_accepted,
                        selected_factor.new_full((), factor),
                        selected_factor,
                    )
                    accepted = accepted | newly_accepted
                    backtracking_attempts = attempt + 1
                    if bool(accepted.all()):
                        break
                actual_update = current.detach() - next_current
                current_after = next_current
            else:
                actual_update = update.detach()
                current_after = (current - update).detach()
            records.append(
                {
                    "inner_step": inner_step,
                    "energy": float(energy.detach().cpu()),
                    "grad_rms": float(raw_grad_rms.mean().detach().cpu()),
                    "raw_update_rms": float(
                        _rms_per_sample(raw_update).mean().detach().cpu()
                    ),
                    "update_rms": float(
                        _rms_per_sample(actual_update).mean().detach().cpu()
                    ),
                    "flow_step_rms": float(flow_step_rms.mean().detach().cpu()),
                    "update_to_flow_ratio": float(
                        (
                            _rms_per_sample(actual_update[:batch_size])
                            / flow_step_rms.clamp_min(1e-12)
                        )
                        .mean()
                        .detach()
                        .cpu()
                    ),
                    "trust_scale": float(trust_scale.mean().detach().cpu()),
                    "backtracking_enabled": self.config.backtracking_steps > 0,
                    "backtracking_attempts": backtracking_attempts,
                    "accepted_fraction": float(accepted.float().mean().cpu()),
                    "accepted": [bool(value) for value in accepted.cpu().tolist()],
                    "selected_factor": [
                        float(value) for value in selected_factor.cpu().tolist()
                    ],
                    "energy_before": [
                        float(value) for value in energy_before.cpu().tolist()
                    ],
                    "energy_after": [
                        float(value) for value in energy_after.cpu().tolist()
                    ],
                    "lookahead_energy_before": [
                        float(value) for value in lookahead_before.cpu().tolist()
                    ],
                    "lookahead_energy_after": [
                        float(value) for value in lookahead_after.cpu().tolist()
                    ],
                    "acceptance_score_before": [
                        float(value) for value in score_before.cpu().tolist()
                    ],
                    "acceptance_score_after": [
                        float(value) for value in score_after.cpu().tolist()
                    ],
                    "presence_before": [
                        float(value) for value in presence_before.detach().cpu().tolist()
                    ],
                    "presence_after": [
                        float(value) for value in presence_after.cpu().tolist()
                    ],
                    "timestep_decay": float(decay.mean().detach().cpu()),
                    "velocity_rms": float(
                        _rms_per_sample(flow_velocity[:batch_size])
                        .mean()
                        .detach()
                        .cpu()
                    ),
                    "flow_velocity_source": (
                        "sampling_path"
                        if reference_velocity is not None
                        or reference_velocity_fn is not None
                        else "attention_proxy"
                    ),
                    "energy_velocity_cosine": _mean_cosine(
                        -conditional_gradient,
                        flow_velocity[:batch_size],
                    ),
                    "objects": [
                        {
                            "batch_index": int(item["batch_index"]),
                            "object_index": int(item["object_index"]),
                            "phrase": str(item["phrase"]),
                            "inside_ratio": float(item["inside_ratio"].detach().cpu()),
                            "cross_ratio": float(item["cross_ratio"].detach().cpu()),
                            "centroid_x": float(item["centroid_x"].detach().cpu()),
                            "centroid_y": float(item["centroid_y"].detach().cpu()),
                            "relation": (
                                str(item["relation"]) if "relation" in item else None
                            ),
                            "relation_signed_delta": (
                                float(item["relation_signed_delta"].detach().cpu())
                                if "relation_signed_delta" in item
                                else None
                            ),
                            "relation_violation": (
                                float(item["relation_violation"].detach().cpu())
                                if "relation_violation" in item
                                else None
                            ),
                            "relation_orthogonal_delta": (
                                float(item["relation_orthogonal_delta"].detach().cpu())
                                if "relation_orthogonal_delta" in item
                                else None
                            ),
                            "relation_dominance_violation": (
                                float(
                                    item["relation_dominance_violation"]
                                    .detach()
                                    .cpu()
                                )
                                if "relation_dominance_violation" in item
                                else None
                            ),
                            "relation_energy": (
                                float(item["relation_energy"].detach().cpu())
                                if "relation_energy" in item
                                else None
                            ),
                            "boxdiff_inner_activation": float(
                                item["inner_activation"].detach().cpu()
                            ),
                            "boxdiff_outer_activation": float(
                                item["outer_activation"].detach().cpu()
                            ),
                            "boxdiff_inner_loss": float(
                                item["inner_loss"].detach().cpu()
                            ),
                            "boxdiff_outer_loss": float(
                                item["outer_loss"].detach().cpu()
                            ),
                            "boxdiff_corner_x_loss": float(
                                item["corner_x_loss"].detach().cpu()
                            ),
                            "boxdiff_corner_y_loss": float(
                                item["corner_y_loss"].detach().cpu()
                            ),
                            "boxdiff_corner_loss": float(
                                item["corner_loss"].detach().cpu()
                            ),
                            "boxdiff_energy": float(
                                item["boxdiff_energy"].detach().cpu()
                            ),
                            "text_gate_mean": float(
                                item["text_gate_mean"].detach().cpu()
                            ),
                            "spatial_entropy": float(
                                item["spatial_entropy"].detach().cpu()
                            ),
                            "entropy_violation": float(
                                item["entropy_violation"].detach().cpu()
                            ),
                            "entity_presence": float(
                                item["entity_presence"].detach().cpu()
                            ),
                            "presence_ratio": float(
                                item["presence_ratio"].detach().cpu()
                            ),
                            "presence_violation": float(
                                item["presence_violation"].detach().cpu()
                            ),
                            "energy": float(item["energy"].detach().cpu()),
                        }
                        for item in object_details
                    ],
                }
            )
            current = current_after
            del attentions, energy, gradient, velocity
            if self.config.backtracking_steps > 0 and not bool(accepted.any()):
                break

        return current, records


class GuidedEulerSampler:
    def __init__(
        self,
        transport: Any,
        num_steps: int,
        guidance_steps: int,
        time_shifting_factor: float | None,
        reverse: bool = False,
    ):
        if guidance_steps < 0:
            raise ValueError("guidance_steps must be non-negative")
        self.transport = transport
        self.drift = transport.get_drift()
        self.num_steps = num_steps
        self.guidance_steps = guidance_steps
        self.time_shifting_factor = time_shifting_factor
        self.reverse = reverse

    def sample(
        self,
        initial_latents: torch.Tensor,
        model: torch.nn.Module,
        model_kwargs: dict[str, Any],
        guider: LatentLayoutGuider,
        velocity_model_kwargs: dict[str, Any] | None = None,
        progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> tuple[torch.Tensor, list[dict[str, Any]]]:
        if velocity_model_kwargs is None:
            velocity_model_kwargs = model_kwargs
        separate_velocity_path = velocity_model_kwargs is not model_kwargs
        latents = initial_latents.float()
        times = make_time_grid(
            self.transport,
            self.num_steps,
            latents.device,
            self.time_shifting_factor,
            self.reverse,
        )
        diagnostics = []

        for step in range(times.numel() - 1):
            t = times[step]
            t_next = times[step + 1]
            timestep = torch.full(
                (latents.shape[0],),
                float(t.item()),
                device=latents.device,
                dtype=torch.float32,
            )
            guidance_records = []
            if step < self.guidance_steps:
                reference_velocity = None
                reference_velocity_fn = None
                if separate_velocity_path:
                    def reference_velocity_fn(value: torch.Tensor) -> torch.Tensor:
                        with torch.no_grad():
                            return self.drift(
                                value,
                                timestep,
                                model.t2i_generate,
                                **velocity_model_kwargs,
                            ).float()

                    with torch.no_grad():
                        reference_velocity = reference_velocity_fn(latents)
                latents, guidance_records = guider.guide(
                    latents,
                    timestep,
                    t_next - t,
                    model_kwargs,
                    reference_velocity=reference_velocity,
                    reference_velocity_fn=reference_velocity_fn,
                )

            with torch.no_grad():
                velocity = self.drift(
                    latents,
                    timestep,
                    model.t2i_generate,
                    **velocity_model_kwargs,
                ).float()
                latents = latents + (t_next - t) * velocity

            record = {
                "step": step,
                "time": float(t.detach().cpu()),
                "next_time": float(t_next.detach().cpu()),
                "dt": float((t_next - t).detach().cpu()),
                "guided": bool(guidance_records),
                "guidance": guidance_records,
                "latent_rms": float(_rms_per_sample(latents).mean().detach().cpu()),
                "velocity_rms": float(_rms_per_sample(velocity).mean().detach().cpu()),
            }
            diagnostics.append(record)
            if progress is not None:
                progress(record)

        return latents.detach(), diagnostics
