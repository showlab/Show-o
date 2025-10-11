import pytest
import torch
from models.modeling import create_decoder_layer


def create_test_layer(config_name: str, layer_idx: int = 0):
    """Создает тестовый слой из конфигурации"""
    config_paths = {
        "moe": "configs/showo_demo_w_clip_vit_512x512_moe.yaml",
        "default": "configs/showo_demo_w_clip_vit_512x512_default.yaml"
    }
    return create_decoder_layer(config_paths[config_name], layer_idx=layer_idx)


class TestNestedConfiguration:
    def test_nested_configuration(self):
        layer_moe = create_test_layer("moe")
        batch_size, seq_len, hidden_size = 2, 10, 2048
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        outputs_moe = layer_moe(hidden_states)
        
        layer_default = create_test_layer("default")
        outputs_default = layer_default(hidden_states)
        
        assert outputs_moe[0].shape == outputs_default[0].shape

    def test_multiple_layers_with_nested_config(self):
        layers_moe = []
        for layer_idx in range(3):
            layer = create_test_layer("moe", layer_idx=layer_idx)
            layers_moe.append(layer)
        
        for i, layer in enumerate(layers_moe):
            assert hasattr(layer.ffn, "num_experts")

    def test_configuration_switching_nested(self):
        configs_to_test = [
            ("moe", "MoE"),
            ("default", "FFNDefault"),
        ]

        for config_name, expected_ffn_type in configs_to_test:
            layer = create_test_layer(config_name)
            if expected_ffn_type == "MoE":
                assert hasattr(layer.ffn, "num_experts")
            else:
                assert hasattr(layer.ffn, "mlp")

    def test_layer_output_shapes(self):
        config_names = ["moe", "default"]
        
        batch_size, seq_len, hidden_size = 2, 10, 2048
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        
        for config_name in config_names:
            layer = create_test_layer(config_name)
            outputs = layer(hidden_states)
            assert outputs[0].shape == hidden_states.shape

    def test_different_layer_indices(self):
        for layer_idx in range(4):
            layer = create_test_layer("moe", layer_idx=layer_idx)
            assert hasattr(layer, 'ffn')

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
        layer = create_test_layer("moe")
        if torch.cuda.is_available():
            layer = layer.cuda()
            assert next(layer.parameters()).is_cuda



if __name__ == "__main__":
    pytest.main([__file__, "-v"])