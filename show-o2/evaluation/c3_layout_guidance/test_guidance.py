#!/usr/bin/env python3
"""CPU smoke tests for Show-o2 layout guidance primitives."""

from __future__ import annotations

import re
import unittest

import torch

from guidance.energy import EnergyConfig, LayoutEnergy
from guidance.layout import (
    LayoutSpec,
    ObjectConstraint,
    map_layout_tokens,
    parse_position_layout,
)
from guidance.sampler import _accept_candidate, _batch_invariant_objective
from guidance.sampler import _rms_per_sample
from guidance.sampler import GuiderConfig, LatentLayoutGuider, make_time_grid
from evaluation.c3_layout_guidance.layout_layer_probe import attention_layer_metrics


class OffsetTokenizer:
    def __call__(
        self,
        text: str,
        add_special_tokens: bool = False,
        return_offsets_mapping: bool = False,
    ):
        del add_special_tokens
        matches = list(re.finditer(r"\S+", text))
        payload = {"input_ids": list(range(len(matches)))}
        if return_offsets_mapping:
            payload["offset_mapping"] = [
                (match.start(), match.end()) for match in matches
            ]
        return payload


class FakeTransport:
    train_eps = None
    sample_eps = None

    @staticmethod
    def check_interval(*args, **kwargs):
        del args, kwargs
        return 0.0, 1.0


class ConstantDriftTransport(FakeTransport):
    @staticmethod
    def get_drift():
        def drift(latents, timestep, model, **model_kwargs):
            return model(latents, timestep, **model_kwargs)

        return drift


class NoOpGuider:
    @staticmethod
    def guide(*args, **kwargs):
        raise AssertionError("guidance_steps=0 must not invoke the guider")


class CapturingGuider:
    def __init__(self):
        self.reference_velocity = None

    def guide(
        self,
        latents,
        timestep,
        dt,
        model_kwargs,
        reference_velocity=None,
        reference_velocity_fn=None,
    ):
        del timestep, dt, model_kwargs, reference_velocity_fn
        self.reference_velocity = reference_velocity
        return latents, []


class ToyAttentionModel:
    @staticmethod
    def t2i_generate_with_attentions(
        image_latents,
        t,
        attention_source,
        **kwargs,
    ):
        del t, attention_source, kwargs
        value = image_latents[:, 0, 0, 0]
        left = torch.exp(-((value - 1.0) ** 2))
        right = left.new_full(left.shape, 0.2)
        attention = left.new_full((value.shape[0], 1, 9, 9), 0.01)
        attention[:, 0, 5, 1] = left
        attention[:, 0, 7, 1] = left
        attention[:, 0, 6, 1] = right
        attention[:, 0, 8, 1] = right
        return torch.zeros_like(image_latents), (attention,)


class LayoutGuidanceTest(unittest.TestCase):
    @staticmethod
    def _boxdiff_fixture() -> tuple[LayoutSpec, torch.Tensor]:
        layout = LayoutSpec(
            prompt="cube background",
            relation="explicit",
            objects=(
                ObjectConstraint(
                    phrase="cube",
                    box=(0.0, 0.0, 0.5, 1.0),
                    char_span=(0, 4),
                    token_indices=(1,),
                ),
            ),
        )
        return layout, torch.tensor([[[4, 5]]])

    @staticmethod
    def _spatial_text_attention(
        entity_values: tuple[float, float, float, float],
    ) -> torch.Tensor:
        attention = torch.full((1, 1, 9, 9), 0.01)
        for query, value in zip((5, 6, 7, 8), entity_values):
            attention[0, 0, query, 1] = value
            attention[0, 0, query, 2] = 1.0 - value
        return attention.requires_grad_(True)

    def test_position_parser_and_token_mapping(self):
        prompt = "a photo of a red cube to the left of a blue ball"
        layout = parse_position_layout(prompt)
        self.assertEqual(layout.relation, "left")
        self.assertEqual(
            [item.phrase for item in layout.objects], ["red cube", "blue ball"]
        )
        mapped = map_layout_tokens(layout, OffsetTokenizer())
        self.assertTrue(all(item.token_indices for item in mapped.objects))
        self.assertLess(mapped.objects[0].box[2], mapped.objects[1].box[0])

    def test_layout_energy_prefers_attention_inside_box(self):
        layout = LayoutSpec(
            prompt="cube",
            relation="explicit",
            objects=(
                ObjectConstraint(
                    phrase="cube",
                    box=(0.0, 0.0, 0.5, 1.0),
                    char_span=(0, 4),
                    token_indices=(1,),
                ),
            ),
        )
        positions = torch.tensor([[[3, 5]]])
        energy_fn = LayoutEnergy(EnergyConfig(layers=(0,)))

        inside = torch.full((1, 1, 8, 8), 0.01)
        inside[0, 0, 4, 1] = 1.0
        inside[0, 0, 6, 1] = 1.0
        inside.requires_grad_(True)
        low, _ = energy_fn((inside,), (layout,), positions, (2, 2), True)

        outside = torch.full((1, 1, 8, 8), 0.01)
        outside[0, 0, 5, 1] = 1.0
        outside[0, 0, 7, 1] = 1.0
        outside.requires_grad_(True)
        high, _ = energy_fn((outside,), (layout,), positions, (2, 2), True)

        self.assertLess(float(low.detach()), float(high.detach()))
        low.backward()
        self.assertIsNotNone(inside.grad)
        self.assertGreater(float(inside.grad.abs().sum()), 0.0)

    def test_relative_energy_uses_ordered_centroid_margin(self):
        layout = LayoutSpec(
            prompt="cube left of ball",
            relation="left",
            objects=(
                ObjectConstraint(
                    phrase="cube",
                    box=(0.0, 0.0, 0.5, 1.0),
                    char_span=(0, 4),
                    token_indices=(1,),
                ),
                ObjectConstraint(
                    phrase="ball",
                    box=(0.5, 0.0, 1.0, 1.0),
                    char_span=(13, 17),
                    token_indices=(2,),
                ),
            ),
        )
        positions = torch.tensor([[[4, 5]]])
        energy_fn = LayoutEnergy(
            EnergyConfig(
                layers=(0,),
                inside_weight=0.0,
                relation_weight=1.0,
                relation_margin=0.2,
            )
        )

        ordered = torch.full((1, 1, 9, 9), 0.01)
        ordered[0, 0, 5, 1] = 1.0
        ordered[0, 0, 7, 1] = 1.0
        ordered[0, 0, 6, 2] = 1.0
        ordered[0, 0, 8, 2] = 1.0
        ordered.requires_grad_(True)
        low, low_details = energy_fn((ordered,), (layout,), positions, (2, 2), True)

        reversed_attention = torch.full((1, 1, 9, 9), 0.01)
        reversed_attention[0, 0, 6, 1] = 1.0
        reversed_attention[0, 0, 8, 1] = 1.0
        reversed_attention[0, 0, 5, 2] = 1.0
        reversed_attention[0, 0, 7, 2] = 1.0
        reversed_attention.requires_grad_(True)
        high, high_details = energy_fn(
            (reversed_attention,), (layout,), positions, (2, 2), True
        )

        self.assertAlmostEqual(float(low.detach()), 0.0, places=7)
        self.assertGreater(float(high.detach()), 0.0)
        self.assertGreater(float(low_details[0]["relation_signed_delta"]), 0.2)
        self.assertGreater(float(high_details[0]["relation_violation"]), 0.0)
        high.backward()
        self.assertGreater(float(reversed_attention.grad.abs().sum()), 0.0)

    def test_boxdiff_inner_prefers_peaks_inside_box(self):
        layout, positions = self._boxdiff_fixture()
        energy_fn = LayoutEnergy(
            EnergyConfig(
                layers=(0,),
                inside_weight=0.0,
                boxdiff_inner_weight=1.0,
                boxdiff_text_scale=5.0,
            )
        )
        aligned = self._spatial_text_attention((0.9, 0.1, 0.9, 0.1))
        misplaced = self._spatial_text_attention((0.1, 0.9, 0.1, 0.9))
        low, low_details = energy_fn(
            (aligned,), (layout,), positions, (2, 2), True
        )
        high, _ = energy_fn((misplaced,), (layout,), positions, (2, 2), True)
        self.assertLess(float(low.detach()), float(high.detach()))
        self.assertGreater(float(low_details[0]["inner_activation"]), 0.5)
        low.backward()
        self.assertGreater(float(aligned.grad.abs().sum()), 0.0)

    def test_text_conditional_attention_is_invariant_to_full_softmax_mass(self):
        layout, positions = self._boxdiff_fixture()
        energy_fn = LayoutEnergy(
            EnergyConfig(
                layers=(0,),
                inside_weight=0.0,
                boxdiff_inner_weight=1.0,
                attention_normalization="text_conditional",
            )
        )
        base = self._spatial_text_attention((0.9, 0.1, 0.9, 0.1))
        scaled = base.detach().clone()
        scaled[:, :, 5:9, 1:3] *= 0.01
        scaled.requires_grad_(True)
        base_energy, _ = energy_fn((base,), (layout,), positions, (2, 2), True)
        scaled_energy, _ = energy_fn(
            (scaled,), (layout,), positions, (2, 2), True
        )
        torch.testing.assert_close(base_energy, scaled_energy, rtol=1e-5, atol=1e-6)

    def test_entropy_penalizes_diffuse_entity_attention(self):
        layout, positions = self._boxdiff_fixture()
        energy_fn = LayoutEnergy(
            EnergyConfig(
                layers=(0,),
                inside_weight=0.0,
                entropy_weight=1.0,
                entropy_target=0.5,
                attention_normalization="text_conditional",
            )
        )
        concentrated = self._spatial_text_attention((0.99, 0.01, 0.01, 0.01))
        diffuse = self._spatial_text_attention((0.5, 0.5, 0.5, 0.5))
        low, low_details = energy_fn(
            (concentrated,), (layout,), positions, (2, 2), True
        )
        high, high_details = energy_fn(
            (diffuse,), (layout,), positions, (2, 2), True
        )
        self.assertLess(float(low.detach()), float(high.detach()))
        self.assertLess(
            float(low_details[0]["spatial_entropy"]),
            float(high_details[0]["spatial_entropy"]),
        )

    def test_presence_uses_raw_attention_mass(self):
        layout, positions = self._boxdiff_fixture()
        energy_fn = LayoutEnergy(
            EnergyConfig(
                layers=(0,),
                inside_weight=0.0,
                presence_weight=1.0,
                presence_min=0.1,
                attention_normalization="text_conditional",
            )
        )
        present = self._spatial_text_attention((0.9, 0.1, 0.9, 0.1))
        weak = present.detach().clone()
        weak[:, :, 5:9, 1:3] *= 0.01
        weak.requires_grad_(True)
        low, low_details = energy_fn((present,), (layout,), positions, (2, 2), True)
        high, high_details = energy_fn((weak,), (layout,), positions, (2, 2), True)
        self.assertLess(float(low.detach()), float(high.detach()))
        self.assertGreater(
            float(low_details[0]["entity_presence"]),
            float(high_details[0]["entity_presence"]),
        )

    def test_scalar_energy_is_mean_of_per_sample_energy(self):
        layout, positions = self._boxdiff_fixture()
        energy_fn = LayoutEnergy(EnergyConfig(layers=(0,)))
        first = self._spatial_text_attention((0.9, 0.1, 0.9, 0.1))
        second = self._spatial_text_attention((0.1, 0.9, 0.1, 0.9))
        attention = torch.cat([first, second], dim=0)
        scalar, per_sample, _details = energy_fn.evaluate(
            (attention,),
            (layout, layout),
            positions.repeat(2, 1, 1),
            (2, 2),
            True,
        )
        self.assertEqual(tuple(per_sample.shape), (2,))
        torch.testing.assert_close(scalar, per_sample.mean())

    def test_relation_dominance_penalizes_wrong_axis(self):
        layout = LayoutSpec(
            prompt="cube left of ball",
            relation="left",
            objects=(
                ObjectConstraint("cube", (0.0, 0.0, 0.5, 1.0), (0, 4), (1,)),
                ObjectConstraint("ball", (0.5, 0.0, 1.0, 1.0), (13, 17), (2,)),
            ),
        )
        positions = torch.tensor([[[4, 5]]])
        energy_fn = LayoutEnergy(
            EnergyConfig(
                layers=(0,),
                inside_weight=0.0,
                relation_weight=1.0,
                relation_margin=0.0,
                relation_dominance_weight=1.0,
                relation_dominance_ratio=1.5,
                attention_normalization="text_conditional",
            )
        )

        aligned = torch.full((1, 1, 9, 9), 0.01)
        aligned[0, 0, 5, 1] = 0.9
        aligned[0, 0, 6, 2] = 0.9
        wrong_axis = torch.full((1, 1, 9, 9), 0.01)
        wrong_axis[0, 0, 5, 1] = 0.9
        wrong_axis[0, 0, 8, 2] = 0.9
        low, _ = energy_fn((aligned,), (layout,), positions, (2, 2), True)
        high, details = energy_fn(
            (wrong_axis.requires_grad_(True),),
            (layout,),
            positions,
            (2, 2),
            True,
        )
        self.assertLess(float(low), float(high))
        self.assertGreater(float(details[0]["relation_dominance_violation"]), 0.0)

    def test_boxdiff_outer_penalizes_response_outside_box(self):
        layout, positions = self._boxdiff_fixture()
        energy_fn = LayoutEnergy(
            EnergyConfig(
                layers=(0,),
                inside_weight=0.0,
                boxdiff_outer_weight=1.0,
                boxdiff_text_scale=5.0,
            )
        )
        contained = self._spatial_text_attention((0.9, 0.1, 0.9, 0.1))
        leaking = self._spatial_text_attention((0.9, 0.9, 0.9, 0.9))
        low, _ = energy_fn((contained,), (layout,), positions, (2, 2), True)
        high, high_details = energy_fn(
            (leaking,), (layout,), positions, (2, 2), True
        )
        self.assertLess(float(low.detach()), float(high.detach()))
        self.assertGreater(float(high_details[0]["outer_activation"]), 0.5)
        high.backward()
        self.assertGreater(float(leaking.grad.abs().sum()), 0.0)

    def test_boxdiff_corner_prefers_aligned_box_projection(self):
        layout, positions = self._boxdiff_fixture()
        energy_fn = LayoutEnergy(
            EnergyConfig(
                layers=(0,),
                inside_weight=0.0,
                boxdiff_corner_weight=1.0,
                boxdiff_corner_radius=0,
                boxdiff_text_scale=5.0,
            )
        )
        aligned = self._spatial_text_attention((0.9, 0.1, 0.9, 0.1))
        shifted = self._spatial_text_attention((0.1, 0.9, 0.1, 0.9))
        low, low_details = energy_fn(
            (aligned,), (layout,), positions, (2, 2), True
        )
        high, _ = energy_fn((shifted,), (layout,), positions, (2, 2), True)
        self.assertLess(float(low.detach()), float(high.detach()))
        self.assertLess(float(low_details[0]["corner_loss"]), 0.1)
        high.backward()
        self.assertGreater(float(shifted.grad.abs().sum()), 0.0)

    def test_boxdiff_config_validation(self):
        with self.assertRaises(ValueError):
            LayoutEnergy(
                EnergyConfig(
                    layers=(0,),
                    inside_weight=0.0,
                    boxdiff_inner_weight=1.0,
                    boxdiff_p=0.0,
                )
            )

    def test_layer_probe_requires_correct_matched_and_inverse_directions(self):
        layouts = (
            LayoutSpec(
                prompt="cube left ball",
                relation="left",
                objects=(
                    ObjectConstraint("cube", (0.0, 0.0, 0.5, 1.0), (0, 4), (1,)),
                    ObjectConstraint("ball", (0.5, 0.0, 1.0, 1.0), (10, 14), (2,)),
                ),
            ),
            LayoutSpec(
                prompt="cube right ball",
                relation="right",
                objects=(
                    ObjectConstraint("cube", (0.5, 0.0, 1.0, 1.0), (0, 4), (1,)),
                    ObjectConstraint("ball", (0.0, 0.0, 0.5, 1.0), (11, 15), (2,)),
                ),
            ),
        )
        positions = torch.tensor([[[4, 5]], [[4, 5]]])
        attention = torch.full((2, 1, 9, 9), 0.01)
        for query in (5, 7):
            attention[0, 0, query, 1] = 0.9
            attention[0, 0, query, 2] = 0.1
            attention[1, 0, query, 1] = 0.1
            attention[1, 0, query, 2] = 0.9
        for query in (6, 8):
            attention[0, 0, query, 1] = 0.1
            attention[0, 0, query, 2] = 0.9
            attention[1, 0, query, 1] = 0.9
            attention[1, 0, query, 2] = 0.1
        metrics = attention_layer_metrics(
            attention,
            layouts,
            positions,
            (2, 2),
            True,
            {
                "relation_margin": 0.2,
                "p": 0.2,
                "corner_radius": 1,
                "text_scale": 5.0,
            },
        )
        self.assertTrue(metrics["pair_direction_correct"])
        self.assertTrue(metrics["pair_margin_correct"])
        self.assertTrue(metrics["relation_flip"])
        with self.assertRaises(ValueError):
            LayoutEnergy(
                EnergyConfig(
                    layers=(0,),
                    inside_weight=0.0,
                    boxdiff_corner_weight=1.0,
                    boxdiff_corner_radius=-1,
                )
            )

    def test_batch_invariant_objective_restores_mean_gradient_scale(self):
        energy = torch.tensor(2.0, requires_grad=True)
        _batch_invariant_objective(energy, batch_size=4).backward()
        self.assertEqual(float(energy.grad), 4.0)

    def test_time_shift_grid_matches_expected_endpoints(self):
        grid = make_time_grid(
            FakeTransport(),
            num_steps=5,
            device=torch.device("cpu"),
            time_shifting_factor=3.0,
        )
        self.assertEqual(tuple(grid.shape), (5,))
        self.assertAlmostEqual(float(grid[0]), 0.0)
        self.assertAlmostEqual(float(grid[-1]), 1.0)
        self.assertTrue(bool(torch.all(grid[1:] > grid[:-1])))

    def test_sampler_uses_separate_velocity_kwargs(self):
        from guidance.sampler import GuidedEulerSampler

        sampler = GuidedEulerSampler(
            ConstantDriftTransport(),
            num_steps=3,
            guidance_steps=0,
            time_shifting_factor=None,
        )

        def model(latents, timestep, value):
            del timestep
            return torch.full_like(latents, value)

        output, _ = sampler.sample(
            initial_latents=torch.zeros(1, 1, 1, 1),
            model=type("Model", (), {"t2i_generate": staticmethod(model)})(),
            model_kwargs={"value": 1.0},
            velocity_model_kwargs={"value": 2.0},
            guider=NoOpGuider(),
        )
        torch.testing.assert_close(output, torch.full_like(output, 2.0))

    def test_separate_sampling_path_sets_guidance_reference_velocity(self):
        from guidance.sampler import GuidedEulerSampler

        sampler = GuidedEulerSampler(
            ConstantDriftTransport(),
            num_steps=2,
            guidance_steps=1,
            time_shifting_factor=None,
        )

        def model(latents, timestep, value):
            del timestep
            return torch.full_like(latents, value)

        guider = CapturingGuider()
        sampler.sample(
            initial_latents=torch.zeros(1, 1, 1, 1),
            model=type("Model", (), {"t2i_generate": staticmethod(model)})(),
            model_kwargs={"value": 1.0},
            velocity_model_kwargs={"value": 2.0},
            guider=guider,
        )
        torch.testing.assert_close(
            guider.reference_velocity,
            torch.full_like(guider.reference_velocity, 2.0),
        )

    def test_cfg_gradient_is_applied_symmetrically(self):
        guider = object.__new__(LatentLayoutGuider)
        conditional = torch.randn(1, 2, 3, 3)
        expanded = guider._cfg_symmetric_gradient(conditional, 2)
        self.assertEqual(tuple(expanded.shape), (2, 2, 3, 3))
        torch.testing.assert_close(expanded[0], expanded[1])

    def test_flow_relative_budget_scales_per_inner_step(self):
        raw_update = torch.full((2, 1, 2, 2), 2.0)
        flow_step_rms = torch.tensor([0.5, 1.0])
        ratio = 0.10
        inner_steps = 2
        target = ratio * flow_step_rms / inner_steps
        scale = (target / _rms_per_sample(raw_update)).clamp(max=1.0)
        update = raw_update * scale.view(-1, 1, 1, 1)
        torch.testing.assert_close(
            _rms_per_sample(update),
            torch.tensor([0.025, 0.05]),
        )

    def test_candidate_acceptance_is_per_sample_and_presence_safe(self):
        accepted = _accept_candidate(
            score_before=torch.tensor([1.0, 1.0, 1.0, 1.0]),
            score_after=torch.tensor([0.8, 0.9995, 0.8, 0.8]),
            min_relative_drop=0.01,
            presence_before=torch.ones(4),
            presence_after=torch.tensor([1.0, 1.0, 0.8, 1.0]),
            presence_preservation_ratio=0.9,
            update_rms=torch.tensor([0.1, 0.1, 0.1, 0.0]),
        )
        torch.testing.assert_close(
            accepted,
            torch.tensor([True, False, False, False]),
        )

    def test_backtracking_accepts_smaller_improving_step(self):
        layout, positions = self._boxdiff_fixture()
        guider = LatentLayoutGuider(
            model=ToyAttentionModel(),
            energy=LayoutEnergy(EnergyConfig(layers=(0,))),
            layouts=(layout,),
            modality_positions=positions,
            spatial_shape=(2, 2),
            add_time_embeds=True,
            config=GuiderConfig(
                inner_steps=1,
                step_size=10.0,
                backtracking_steps=4,
                backtracking_factor=0.5,
                min_relative_drop=0.001,
            ),
        )
        initial = torch.zeros(1, 1, 1, 1)
        updated, records = guider.guide(
            initial,
            torch.zeros(1),
            torch.tensor(0.1),
            {},
            reference_velocity_fn=torch.zeros_like,
        )
        self.assertTrue(records[0]["accepted"][0])
        self.assertLess(records[0]["selected_factor"][0], 1.0)
        self.assertLess(
            records[0]["acceptance_score_after"][0],
            records[0]["acceptance_score_before"][0],
        )
        self.assertGreater(float(updated.item()), float(initial.item()))

    def test_backtracking_rejects_all_nonqualifying_steps(self):
        layout, positions = self._boxdiff_fixture()
        guider = LatentLayoutGuider(
            model=ToyAttentionModel(),
            energy=LayoutEnergy(EnergyConfig(layers=(0,))),
            layouts=(layout,),
            modality_positions=positions,
            spatial_shape=(2, 2),
            add_time_embeds=True,
            config=GuiderConfig(
                inner_steps=1,
                step_size=10.0,
                backtracking_steps=2,
                min_relative_drop=100.0,
            ),
        )
        initial = torch.zeros(1, 1, 1, 1)
        updated, records = guider.guide(
            initial,
            torch.zeros(1),
            torch.tensor(0.1),
            {},
            reference_velocity_fn=torch.zeros_like,
        )
        self.assertFalse(records[0]["accepted"][0])
        self.assertEqual(records[0]["selected_factor"][0], 0.0)
        torch.testing.assert_close(updated, initial)

    def test_guider_config_validation(self):
        with self.assertRaises(ValueError):
            GuiderConfig(inner_steps=0)
        with self.assertRaises(ValueError):
            GuiderConfig(attention_source="unknown")
        with self.assertRaises(ValueError):
            GuiderConfig(max_flow_step_ratio=-0.1)
        with self.assertRaises(ValueError):
            GuiderConfig(timestep_decay_power=-1.0)
        with self.assertRaises(ValueError):
            GuiderConfig(backtracking_steps=-1)
        with self.assertRaises(ValueError):
            GuiderConfig(backtracking_factor=1.0)
        with self.assertRaises(ValueError):
            GuiderConfig(min_relative_drop=-0.1)
        with self.assertRaises(ValueError):
            GuiderConfig(lookahead_weight=0.5, backtracking_steps=0)
        with self.assertRaises(ValueError):
            GuiderConfig(presence_preservation_ratio=1.1)


if __name__ == "__main__":
    unittest.main()
