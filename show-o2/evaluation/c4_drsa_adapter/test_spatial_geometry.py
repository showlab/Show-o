import unittest

import torch

from models.spatial_alignment_adapter import (
    AxialRoPE2D,
    DualRouteSpatialAlignmentAdapter,
    RelationConditionedRelativeBias,
    RelationConditionedSpatialAttention,
    SpatialAttentionConfig,
)


class AxialRoPE2DTest(unittest.TestCase):
    def test_equal_2d_displacements_have_equal_scores(self):
        rope = AxialRoPE2D(head_dim=8)
        query = torch.ones(1, 1, 6, 8)
        key = torch.ones_like(query)
        query, key = rope(query, key, grid_height=2, grid_width=3)
        scores = torch.matmul(query, key.transpose(-2, -1))[0, 0]
        self.assertTrue(torch.allclose(scores[0, 1], scores[3, 4], atol=1e-6))

    def test_row_wrap_is_not_treated_as_horizontal_neighbor(self):
        rope = AxialRoPE2D(head_dim=8)
        query = torch.ones(1, 1, 6, 8)
        key = torch.ones_like(query)
        query, key = rope(query, key, grid_height=2, grid_width=3)
        scores = torch.matmul(query, key.transpose(-2, -1))[0, 0]
        self.assertFalse(torch.allclose(scores[1, 2], scores[2, 3], atol=1e-4))


class RelativeBiasTest(unittest.TestCase):
    def test_bias_is_zero_initialized(self):
        module = RelationConditionedRelativeBias(16, 2, 8)
        bias = module(torch.randn(3, 16), grid_height=2, grid_width=3, dtype=torch.float32)
        self.assertEqual(tuple(bias.shape), (3, 2, 6, 6))
        self.assertTrue(torch.equal(bias, torch.zeros_like(bias)))

    def test_signed_features_are_antisymmetric(self):
        features = RelationConditionedRelativeBias.relative_features(
            2,
            3,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        self.assertTrue(torch.allclose(features[..., 0], -features[..., 0].T))
        self.assertTrue(torch.allclose(features[..., 1], -features[..., 1].T))
        self.assertTrue(torch.allclose(features[..., 2], features[..., 2].T))
        self.assertTrue(torch.allclose(features[..., 3], features[..., 3].T))


class SpatialAttentionTest(unittest.TestCase):
    def test_initial_adapter_is_exact_noop(self):
        config = SpatialAttentionConfig(
            hidden_size=32,
            relation_dim=16,
            num_heads=4,
            bias_hidden_dim=8,
        )
        module = RelationConditionedSpatialAttention(config)
        tokens = torch.randn(2, 12, 32)
        relation = torch.randn(2, 16)
        output, attention = module(
            tokens,
            relation,
            grid_height=3,
            grid_width=4,
            return_attention=True,
        )
        self.assertTrue(torch.equal(output, tokens))
        self.assertEqual(tuple(attention.shape), (2, 4, 12, 12))

    def test_dynamic_rectangular_grid(self):
        config = SpatialAttentionConfig(
            hidden_size=32,
            relation_dim=16,
            num_heads=4,
            bias_hidden_dim=8,
        )
        module = RelationConditionedSpatialAttention(config)
        tokens = torch.randn(1, 15, 32)
        relation = torch.randn(1, 16)
        output = module(tokens, relation, grid_height=3, grid_width=5)
        self.assertEqual(tuple(output.shape), (1, 15, 32))




class DualRouteAdapterTest(unittest.TestCase):
    def test_default_adapter_has_exactly_one_generation_route(self):
        module = DualRouteSpatialAlignmentAdapter(
            hidden_size=32,
            relation_dim=16,
            num_heads=4,
            bias_hidden_dim=8,
        )
        self.assertIsNone(module.generation_spatial)
        state_keys = tuple(module.state_dict())
        self.assertTrue(any(key.startswith("generation.") for key in state_keys))
        self.assertFalse(any(key.startswith("generation_spatial.") for key in state_keys))

    def test_legacy_second_generation_route_is_opt_in(self):
        module = DualRouteSpatialAlignmentAdapter(
            hidden_size=32,
            relation_dim=16,
            num_heads=4,
            bias_hidden_dim=8,
        )
        module.enable_generation_spatial()
        self.assertTrue(
            any(key.startswith("generation_spatial.") for key in module.state_dict())
        )

    def test_multimodal_codes_and_generation_backward(self):
        module = DualRouteSpatialAlignmentAdapter(
            hidden_size=32,
            relation_dim=16,
            num_heads=4,
            bias_hidden_dim=8,
        )
        understanding_tokens = torch.randn(4, 6, 32)
        relation_codes = torch.randn(2, 2, 16)
        understanding = module.apply_understanding(
            understanding_tokens,
            relation_codes,
            grid_height=2,
            grid_width=3,
        )
        self.assertTrue(torch.equal(understanding, understanding_tokens))
        hidden_states = torch.randn(1, 7, 32, requires_grad=True)
        modality_positions = torch.tensor([[[0, 7]]])
        generation = module.apply_generation(
            hidden_states,
            modality_positions,
            relation_codes[:1, :1],
            grid_height=2,
            grid_width=3,
            add_time_embeds=True,
        )
        generation.square().mean().backward()
        self.assertIsNotNone(module.generation.output.weight.grad)
        self.assertGreater(module.generation.output.weight.grad.abs().sum().item(), 0.0)
if __name__ == "__main__":
    unittest.main()
