from typing import Optional
from training.moe import MoE
import re
import torch
from training.moe_mlflow_logger import MoEMLflowLogger
from moe_visualization import MoEVisualizer


def patch_model_with_moe(
    model,
    count_layers_to_patch=3,
    num_experts=4,
    top_k=2,
    special_tokens=None,
    noise_std: float = 1e-2,
    use_modality_bias: bool = False,
    use_domain_bias: bool = False,
    modality_init_hardness: float = 1.0,
    modality_init_steps: int = 1000,
    modality_init_hardness_min: float = 0.2,
    domain_init_hardness: float = 1.0,
    domain_init_steps: int = 1000,
    domain_init_hardness_min: float = 0.0,
    use_gumbel: bool = False,
    gate_capacity: Optional[int] = None,
    random_routing: bool = False,
    domain_to_expert_map: Optional[dict] = None,
):
    total_layers = len(model.showo.model.layers)
    print(f"🔧 Патчим модель с MoE:")
    print(f"   Всего слоев в модели: {total_layers}")
    print(f"   Количество экспертов: {num_experts}")
    print(f"   Top-K: {top_k}")
    print(f"   Слоев для патчинга: {count_layers_to_patch}")
    config_phi = model.showo.config
    patched_layers = []
    for layer_idx, layer in list(enumerate(model.showo.model.layers))[::-1]:
        if count_layers_to_patch == 0:
            break
        if hasattr(layer, 'mlp'):
            print(f"  → Слой {layer_idx}")
            patched_layers.append(layer_idx)
            original_mlp = layer.mlp
            moe_layer = MoE(
                num_experts=num_experts,
                hidden_size=config_phi.hidden_size,
                top_k=top_k,
                config=config_phi,
                template_mlp=original_mlp,
                noise_std=noise_std,
                use_modality_bias=use_modality_bias,
                use_domain_bias=use_domain_bias,
                modality_init_hardness=modality_init_hardness,
                modality_init_steps=modality_init_steps,
                modality_init_hardness_min=modality_init_hardness_min,
                domain_init_hardness=domain_init_hardness,
                domain_init_steps=domain_init_steps,
                domain_init_hardness_min=domain_init_hardness_min,
                use_gumbel=use_gumbel,
                gate_capacity=gate_capacity,
                random_routing=random_routing,
                domain_to_expert_map=domain_to_expert_map,
            )
            moe_layer.to(next(original_mlp.parameters()).device)
            moe_layer.enable_logging(log_gates=True, log_activations=True, log_frequency=50)
            moe_layer.set_layer_id(layer_idx)
            if special_tokens is not None:
                moe_layer.set_special_tokens(
                    soi_id=special_tokens.get('soi_id'),
                    eoi_id=special_tokens.get('eoi_id'),
                    sov_id=special_tokens.get('sov_id'),
                    eov_id=special_tokens.get('eov_id')
                )
            layer.mlp = moe_layer
            count_layers_to_patch -= 1
    print(f"✓ Заменено слоев: {len(patched_layers)} из {total_layers} ({100*len(patched_layers)/total_layers:.1f}%)")
    print(f"✓ Слои с MoE: {patched_layers}")
    print(f"✓ Слои с оригинальным FFN (предобученным): {[i for i in range(total_layers) if i not in patched_layers]}")
    return model


def freeze_non_moe_params(model):
    for param in model.parameters():
        param.requires_grad = False
    
    min_moe_layer = None
    for name, param in model.named_parameters():
        if 'showo.model.layers' in name and any(x in name for x in ['mlp.experts', 'mlp.gate', 'mlp.alpha']):
            match = re.search(r'layers\.(\d+)\.', name)
            if match:
                layer_idx = int(match.group(1))
                if min_moe_layer is None or layer_idx < min_moe_layer:
                    min_moe_layer = layer_idx
    
    if min_moe_layer is None:
        raise Exception("MoE слои не найдены")
    
    trainable_params = 0
    total_params = 0
    trainable_param_names = []
    
    for name, param in model.named_parameters():
        total_params += param.numel()
        if 'showo.model.layers' in name:
            match = re.search(r'layers\.(\d+)\.', name)
            if match:
                layer_idx = int(match.group(1))
                if layer_idx >= min_moe_layer:
                    param.requires_grad = True
                    trainable_params += param.numel()
                    trainable_param_names.append(name)
    return model


def patch_and_freeze_moe(
    model,
    count_layers_to_patch: int = 3, 
    num_experts: int = 8, 
    top_k: int = 2, 
    mlflow_client: Optional[object] = None,
    mlflow_run_id: Optional[str] = None,
    special_tokens: Optional[dict] = None,
    noise_std: float = 1e-2,
    use_modality_bias: bool = False,
    use_domain_bias: bool = False,
    modality_init_hardness: float = 1.0,
    modality_init_steps: int = 1000,
    modality_init_hardness_min: float = 0.2,
    domain_init_hardness: float = 1.0,
    domain_init_steps: int = 1000,
    domain_init_hardness_min: float = 0.0,
    use_gumbel: bool = False,
    gate_capacity: Optional[int] = None,
    random_routing: bool = False,
    domain_to_expert_map: Optional[dict] = None,
):
    model = patch_model_with_moe(
        model, 
        count_layers_to_patch, 
        num_experts, 
        top_k, 
        special_tokens, 
        noise_std=noise_std,
        use_modality_bias=use_modality_bias,
        use_domain_bias=use_domain_bias,
        modality_init_hardness=modality_init_hardness, 
        modality_init_steps=modality_init_steps,
        modality_init_hardness_min=modality_init_hardness_min,
        domain_init_hardness=domain_init_hardness,
        domain_init_steps=domain_init_steps,
        domain_init_hardness_min=domain_init_hardness_min,
        use_gumbel=use_gumbel,
        gate_capacity=gate_capacity,
        random_routing=random_routing,
        domain_to_expert_map=domain_to_expert_map,
    )


    if mlflow_client is not None and mlflow_run_id is not None:
        mlflow_logger = MoEMLflowLogger(mlflow_client=mlflow_client, mlflow_run_id=mlflow_run_id)
        for layer_idx, layer in enumerate(model.showo.model.layers):
            if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'experts'):
                layer.mlp.set_mlflow_logger(mlflow_logger)
                visualizer = MoEVisualizer(num_experts=num_experts, layer_id=layer_idx)
                layer.mlp.set_visualizer(visualizer)
                layer.mlp._layer_id = layer_idx
                layer.mlp.enable_logging(log_gates=True, log_activations=True, log_frequency=50)
    
    model = freeze_non_moe_params(model)
    return model


def save_moe_weights(model, path):
    moe_state = {
        k: v for k, v in model.state_dict().items() 
        if any(x in k for x in ['mlp.experts', 'mlp.gate', 'mlp.alpha'])
    }
    torch.save(moe_state, path)


def load_moe_weights(model, path):
    moe_weights = torch.load(path)
    model.load_state_dict(moe_weights, strict=False)
    return model

