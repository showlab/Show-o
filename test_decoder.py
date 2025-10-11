import pytest
import torch
import torch.nn as nn
from models.modeling import create_decoder_layer, create_ffn


def create_test_layer(config_name: str, layer_idx: int = 0):
    config_paths = {
        "moe": "configs/showo_demo_w_clip_vit_512x512_moe.yaml",
        "default": "configs/showo_demo_w_clip_vit_512x512_default.yaml",
    }
    return create_decoder_layer(config_paths[config_name], layer_idx=layer_idx)


def create_test_ffn(config_name: str):
    config_paths = {
        "moe": "configs/showo_demo_w_clip_vit_512x512_moe.yaml",
        "default": "configs/showo_demo_w_clip_vit_512x512_default.yaml",
    }
    return create_ffn(config_paths[config_name])


class TestFFNDefault:
    def test_ffn_default_initialization(self):
        ffn = create_test_ffn("default")

        assert isinstance(ffn.mlp, nn.Module)
        assert isinstance(ffn.input_layernorm, nn.LayerNorm)
        assert isinstance(ffn.dropout, nn.Dropout)
        assert ffn.input_layernorm.normalized_shape == (2048,)

    def test_ffn_default_forward(self):
        ffn = create_test_ffn("default")
        batch_size, seq_len, hidden_size = 2, 10, 2048
        x = torch.randn(batch_size, seq_len, hidden_size)
        output = ffn(x)

        assert output.shape == x.shape
        assert output.dtype == x.dtype
        assert output.device == x.device
        assert not torch.allclose(output, x, atol=1e-6)

    def test_ffn_default_gradient_flow(self):
        ffn = create_test_ffn("default")
        x = torch.randn(2, 10, 2048, requires_grad=True)
        output = ffn(x)
        loss = output.sum()
        loss.backward()
        assert x.grad is not None
        assert not torch.allclose(x.grad, torch.zeros_like(x.grad))


class TestMoE:
    def test_moe_initialization(self):
        moe = create_test_ffn("moe")

        assert moe.num_experts == 8
        assert moe.top_k == 2
        assert moe.hidden_size == 2048
        assert len(moe.experts) == 8
        assert hasattr(moe, "gate")
        assert moe.alpha.shape == (8,)

    def test_moe_forward(self):
        moe = create_test_ffn("moe")
        batch_size, seq_len, hidden_size = 2, 10, 2048
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        output = moe(hidden_states)
        assert output.shape == hidden_states.shape
        assert output.dtype == hidden_states.dtype
        assert output.device == hidden_states.device

    def test_moe_gradient_flow(self):
        moe = create_test_ffn("moe")
        hidden_states = torch.randn(2, 10, 2048, requires_grad=True)

        output = moe(hidden_states)
        loss = output.sum()
        loss.backward()

        assert hidden_states.grad is not None
        assert moe.alpha.grad is not None


class TestPhiDecoderLayer:
    """Тесты для PhiDecoderLayer класса"""

    def test_decoder_layer_initialization_default(self):
        """Тест инициализации PhiDecoderLayer с FFNDefault"""
        layer = create_test_layer("default")

        assert hasattr(layer, "self_attn")
        assert hasattr(layer, "ffn")
        assert hasattr(layer, "input_layernorm")
        assert hasattr(layer, "resid_dropout")

    def test_decoder_layer_initialization_moe(self):
        """Тест инициализации PhiDecoderLayer с MoE"""
        layer = create_test_layer("moe")

        assert hasattr(layer, "self_attn")
        assert hasattr(layer, "ffn")
        assert hasattr(layer, "input_layernorm")
        assert hasattr(layer, "resid_dropout")

    def test_decoder_layer_forward_default_ffn(self):
        """Тест forward pass с FFNDefault"""
        layer = create_test_layer("default")

        batch_size, seq_len, hidden_size = 2, 10, 2048
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)

        outputs = layer(hidden_states)

        assert len(outputs) == 1
        assert outputs[0].shape == hidden_states.shape

    def test_decoder_layer_forward_moe(self):
        """Тест forward pass с MoE"""
        layer = create_test_layer("moe")

        batch_size, seq_len, hidden_size = 2, 10, 2048
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)

        outputs = layer(hidden_states)

        assert len(outputs) == 1
        assert outputs[0].shape == hidden_states.shape

    def test_decoder_layer_with_attention_output(self):
        """Тест с возвратом attention weights"""
        layer = create_test_layer("default")
        hidden_states = torch.randn(2, 10, 2048)

        outputs = layer(hidden_states, output_attentions=True)

        assert len(outputs) == 2
        assert outputs[0].shape == hidden_states.shape
        assert outputs[1] is not None

    def test_decoder_layer_gradient_flow_default(self):
        """Тест градиентного потока для PhiDecoderLayer с FFNDefault"""
        layer = create_test_layer("default")
        hidden_states = torch.randn(2, 10, 2048, requires_grad=True)

        outputs = layer(hidden_states)
        loss = outputs[0].sum()
        loss.backward()

        assert hidden_states.grad is not None

    def test_decoder_layer_gradient_flow_moe(self):
        layer = create_test_layer("moe")
        hidden_states = torch.randn(2, 10, 2048, requires_grad=True)

        outputs = layer(hidden_states)
        loss = outputs[0].sum()
        loss.backward()

        assert hidden_states.grad is not None

    def test_multiple_layers_different_indices(self):
        layers = []
        for layer_idx in range(3):
            layer = create_test_layer("moe", layer_idx=layer_idx)
            layers.append(layer)
            assert hasattr(layer, "ffn")
            assert hasattr(layer.ffn, "num_experts")

    def test_configuration_switching(self):
        layer_default = create_test_layer("default")
        layer_moe = create_test_layer("moe")

        batch_size, seq_len, hidden_size = 2, 10, 2048
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)

        outputs_default = layer_default(hidden_states)
        outputs_moe = layer_moe(hidden_states)

        assert outputs_default[0].shape == outputs_moe[0].shape
        assert hasattr(layer_default.ffn, "mlp")
        assert hasattr(layer_moe.ffn, "num_experts")

    def test_training_mode(self):
        layer = create_test_layer("moe")
        layer.train()
        assert layer.training

    def test_evaluation_mode(self):
        layer = create_test_layer("moe")
        layer.eval()
        assert not layer.training

    def test_parameter_existence(self):
        layer = create_test_layer("moe")
        parameters = list(layer.parameters())
        assert len(parameters) > 0

    def test_device_moving(self):
        """Тест перемещения на устройство"""
        layer = create_test_layer("moe")
        if torch.cuda.is_available():
            layer = layer.cuda()
            assert next(layer.parameters()).is_cuda

    def test_decoder_shapes_comparison(self):
        layer_default = create_test_layer("default")
        layer_moe = create_test_layer("moe")
        batch_size, seq_len, hidden_size = 2, 10, 2048
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        outputs_default = layer_default(hidden_states)
        outputs_moe = layer_moe(hidden_states)
        assert outputs_default[0].shape == outputs_moe[0].shape
        assert outputs_default[0].shape == hidden_states.shape
        assert not torch.allclose(outputs_default[0], outputs_moe[0], atol=1e-6)

    def test_decoder_parameter_counts(self):
        layer_default = create_test_layer("default")
        layer_moe = create_test_layer("moe")

        params_default = sum(p.numel() for p in layer_default.parameters())
        params_moe = sum(p.numel() for p in layer_moe.parameters())
        assert params_moe > params_default
        assert params_default > 0
        assert params_moe > 0
        assert hasattr(layer_moe.ffn, "experts")
        assert len(layer_moe.ffn.experts) > 0

        assert hasattr(layer_default.ffn, "mlp")

    def test_decoder_forward_consistency(self):
        layer_default = create_test_layer("default")
        layer_moe = create_test_layer("moe")

        batch_size, seq_len, hidden_size = 2, 10, 2048
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)

        for _ in range(3):
            outputs_default = layer_default(hidden_states)
            outputs_moe = layer_moe(hidden_states)

            assert outputs_default[0].shape == outputs_moe[0].shape
            assert outputs_default[0].shape == hidden_states.shape

            assert not torch.allclose(outputs_default[0], outputs_moe[0], atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
