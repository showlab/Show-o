import unittest

import torch
import torch.nn as nn

from models.generation_adaptation import (
    checkpoint_model_kwargs,
    configure_generation_adaptation,
    configure_from_checkpoint,
    generation_adapter_parameter_count,
)
from models.generation_lora import GenerationLoRAController, matched_lora_budget
from models.spatial_alignment_adapter import GenerationSpatialAdapter


class FakeAttention(nn.Module):
    def __init__(self, hidden_size: int, kv_size: int, device=None):
        super().__init__()
        self.q_proj = nn.Linear(hidden_size, hidden_size, device=device)
        self.k_proj = nn.Linear(hidden_size, kv_size, device=device)
        self.v_proj = nn.Linear(hidden_size, kv_size, device=device)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False, device=device)


class FakeLayer(nn.Module):
    def __init__(self, hidden_size: int, kv_size: int, device=None):
        super().__init__()
        self.self_attn = FakeAttention(hidden_size, kv_size, device=device)


class FakeQwen(nn.Module):
    def __init__(self, layers: int, hidden_size: int, kv_size: int, device=None):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(
            [FakeLayer(hidden_size, kv_size, device=device) for _ in range(layers)]
        )


class FakeModel(nn.Module):
    def __init__(self, *, g_adapter: bool):
        super().__init__()
        self.config = type("Config", (), {"hidden_size": 32})()
        self.showo = FakeQwen(layers=2, hidden_size=32, kv_size=8)
        self.generation_adapter = (
            GenerationSpatialAdapter(
                hidden_size=32, spatial_dim=604, num_heads=8, bias_hidden_dim=128
            )
            if g_adapter else None
        )
        self.generation_adapter_active = True


class GenerationAdaptationTest(unittest.TestCase):
    def test_g_adapter_is_exact_noop_at_initialization(self):
        adapter = GenerationSpatialAdapter(
            hidden_size=32, spatial_dim=16, num_heads=4, bias_hidden_dim=8
        )
        hidden = torch.randn(1, 9, 32)
        positions = torch.tensor([[[2, 7]]])
        output = adapter(
            hidden,
            positions,
            grid_height=2,
            grid_width=3,
            add_time_embeds=True,
        )
        self.assertTrue(torch.equal(output, hidden))

    def test_qwen_1_5b_parameter_match_is_below_one_percent(self):
        qwen = FakeQwen(layers=28, hidden_size=1536, kv_size=256, device="meta")
        target = generation_adapter_parameter_count(1536)
        budget = matched_lora_budget(qwen, target)
        self.assertEqual(budget.rank, 35)
        self.assertEqual(target, 9_533_428)
        self.assertEqual(budget.trainable_parameters, 9_533_440)
        self.assertEqual(abs(target - budget.trainable_parameters), 12)
        self.assertLess(budget.relative_difference, 0.01)

    def test_lora_enable_disable_and_state_round_trip(self):
        torch.manual_seed(7)
        qwen = FakeQwen(layers=2, hidden_size=16, kv_size=4)
        controller = GenerationLoRAController(qwen, rank=3, alpha=3)
        attention = qwen.model.layers[0].self_attn
        inputs = torch.randn(2, 5, 16)
        with controller.enabled(False):
            base = attention.q_proj(inputs)
        self.assertTrue(torch.equal(attention.q_proj(inputs), base))
        with torch.no_grad():
            attention.q_proj.lora_b.weight.fill_(0.25)
        active = attention.q_proj(inputs)
        self.assertFalse(torch.equal(active, base))
        state = controller.state_dict()
        clone_qwen = FakeQwen(layers=2, hidden_size=16, kv_size=4)
        clone = GenerationLoRAController(clone_qwen, rank=3, alpha=3)
        clone.load_state_dict(state)
        for name, parameter in clone.named_parameters():
            self.assertTrue(torch.equal(parameter, state[name]))

    def test_route_controller_disables_only_selected_route(self):
        model = FakeModel(g_adapter=True)
        route = configure_generation_adaptation(model, "g_adapter")
        with route.enabled(False):
            self.assertFalse(model.generation_adapter_active)
        self.assertTrue(model.generation_adapter_active)

    def test_new_checkpoint_model_kwargs_preserve_route_and_dimension(self):
        g_payload = {
            "adaptation_config": {"route": "g_adapter", "spatial_dim": 604}
        }
        lora_payload = {
            "adaptation_config": {"route": "lora", "rank": 3, "alpha": 3.0}
        }
        self.assertEqual(checkpoint_model_kwargs(g_payload), {
            "drsa_enabled": False,
            "generation_adapter_enabled": True,
            "generation_adapter_spatial_dim": 604,
        })
        self.assertEqual(checkpoint_model_kwargs(lora_payload), {
            "drsa_enabled": False,
            "generation_adapter_enabled": False,
            "generation_adapter_spatial_dim": 604,
        })

    def test_g_adapter_checkpoint_round_trip(self):
        source_model = FakeModel(g_adapter=True)
        source = configure_generation_adaptation(source_model, "g_adapter")
        with torch.no_grad():
            source_model.generation_adapter.spatial.output.weight.fill_(0.125)
        payload = {
            "adaptation_config": source.config_dict(),
            "adaptation_state": source.state_dict(),
        }
        target_model = FakeModel(g_adapter=True)
        target = configure_from_checkpoint(target_model, payload)
        for name, value in source.state_dict().items():
            self.assertTrue(torch.equal(value, target.state_dict()[name]))

    def test_lora_checkpoint_round_trip(self):
        source_model = FakeModel(g_adapter=False)
        source = configure_generation_adaptation(
            source_model, "lora", lora_rank=230, lora_alpha=230
        )
        with torch.no_grad():
            next(source.parameters()).fill_(0.25)
        payload = {
            "adaptation_config": source.config_dict(),
            "adaptation_state": source.state_dict(),
        }
        target_model = FakeModel(g_adapter=False)
        target = configure_from_checkpoint(target_model, payload)
        for name, value in source.state_dict().items():
            self.assertTrue(torch.equal(value, target.state_dict()[name]))


if __name__ == "__main__":
    unittest.main()
