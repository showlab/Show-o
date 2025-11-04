import torch
import torch.nn as nn
import copy
import json
import os
import logging
from typing import Optional
from models.phi import PhiConfig
from models.moe_gates.gshard_gate import GShardGate



class SmallPhiMLP(nn.Module):
    def __init__(self, config: PhiConfig, scale_factor: int = 1):
        super().__init__()
        self.config = config
        intermediate_size = config.intermediate_size // scale_factor
        if config.hidden_act == "gelu_new":
            self.activation_fn = torch.nn.functional.gelu
        else:
            self.activation_fn = getattr(torch.nn.functional, config.hidden_act)
        self.fc1 = nn.Linear(config.hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states



class MoE(nn.Module):
    def __init__(self, num_experts, hidden_size, top_k, config: PhiConfig, template_mlp: Optional[nn.Module] = None, noise_std: float = 1e-3,
                 modality_init_hardness: float = 1.0, modality_init_steps: int = 1000, modality_init_hardness_min: float = 0.2,
                 use_gumbel: bool = False):
        super().__init__()
        self.gate = GShardGate(hidden_size, num_experts, world_size=4, top_k=top_k, gate_bias=True, use_gumbel=use_gumbel)
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.top_k = top_k
        num_text_experts = num_experts // 2
        tot_expert = num_experts * 4  # world_size=4
        self.register_buffer('modality_bias_text', torch.zeros(tot_expert))
        self.register_buffer('modality_bias_image', torch.zeros(tot_expert))
        init_bias_val = 10.0
        for rank in range(4):  # world_size=4
            start_text = rank * num_experts
            end_text = start_text + num_text_experts
            start_image = end_text
            end_image = start_image + num_text_experts
            self.modality_bias_text[start_text:end_text] = init_bias_val
            self.modality_bias_image[start_image:end_image] = init_bias_val
        self.modality_init_hardness = float(modality_init_hardness)
        self.modality_init_steps = int(modality_init_steps)
        self.modality_init_hardness_min = float(modality_init_hardness_min)
        self.experts = nn.ModuleList()
        if template_mlp is not None:
            for _ in range(self.num_experts):
                expert = copy.deepcopy(template_mlp)
                with torch.no_grad():
                    for p in expert.parameters():
                        p.add_(torch.randn_like(p) * noise_std)
                self.experts.append(expert)
        else:
            self.experts = nn.ModuleList([SmallPhiMLP(config, scale_factor=1) for _ in range(self.num_experts)])
        self.alpha = nn.Parameter(torch.randn(self.num_experts))
        self._step_count = 0
        self._log_frequency = 100
        self._global_step = 0
        self._layer_id = None
        self._soi_id = None
        self._eoi_id = None
        self._sov_id = None
        self._eov_id = None
        self._gate_distribution_history = {}
        self._modality_gate_distribution_history = {}
        self._domain_gate_distribution_history = {}
        self._mlflow_logger = None
        self._visualizer = None

    def set_global_step(self, global_step):
        self._global_step = global_step


    def forward(self, hidden_states, input_ids=None, temperature: Optional[float] = None, domain_id: Optional[str] = None):
        device = hidden_states.device
        batch_size, seq_len, hidden_size = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_size)  # [B*L, H]
        B = hidden_states_flat.shape[0]

        # Определяем модальность и применяем bias для инициализации
        modality_bias = None
        if input_ids is not None and self.modality_init_hardness > 0:
            # Вычисляем текущую hardness (линейное уменьшение от max до min)
            # Не убираем bias полностью - оставляем минимальное значение для мягкого сигнала
            if self._global_step < self.modality_init_steps:
                # Линейное уменьшение от max до min за modality_init_steps шагов
                progress = self._global_step / max(self.modality_init_steps, 1)
                hardness = self.modality_init_hardness - (self.modality_init_hardness - self.modality_init_hardness_min) * progress
            else:
                hardness = self.modality_init_hardness_min
            
            if hardness > 0:
                modality = self._get_token_modality(input_ids.view(-1))
                if modality is not None:
                    # Для текстовых токенов (modality == 0) используем text bias, для изображений (== 1) — image bias
                    text_mask = (modality == 0)
                    image_mask = (modality == 1)
                    # Создаем bias для каждого токена: [B, E_tot]
                    modality_bias = torch.zeros(B, self.modality_bias_text.size(0), device=device, dtype=hidden_states.dtype)
                    if text_mask.any():
                        modality_bias[text_mask] = self.modality_bias_text.unsqueeze(0) * hardness
                    if image_mask.any():
                        modality_bias[image_mask] = self.modality_bias_image.unsqueeze(0) * hardness
                    # Передаем per-token bias для корректного routing

        gate_idx, gate_score = self.gate(hidden_states_flat, temperature=temperature, modality_bias=modality_bias)
        out_flat = torch.zeros(B, hidden_size, device=device, dtype=hidden_states.dtype)
        for k in range(self.top_k):
            expert_indices = gate_idx[:, k]
            weights = gate_score[:, k]
            for expert_id in range(self.num_experts):
                mask = expert_indices == expert_id
                if mask.any():
                    expert_input = hidden_states_flat[mask]
                    expert_output = self.experts[expert_id](expert_input)
                    weight = weights[mask] * self.alpha[expert_id]
                    out_flat[mask] += expert_output * weight.unsqueeze(-1)

        output = out_flat.view(batch_size, seq_len, hidden_size)
        self._step_count += 1
        should_log = (self._step_count % self._log_frequency == 0)
        if hasattr(self, '_log_gates') and self._log_gates and should_log:
            self._log_gate_distribution(gate_idx, gate_score.detach(), input_ids, domain_id=domain_id)

        return output


    def _log_gate_distribution(self, gate_idx, gate_score, input_ids=None, domain_id=None):
        logger = logging.getLogger(__name__)
        
        modality = self._get_token_modality(input_ids.view(-1) if input_ids is not None else None)
        
        expert_counts = {}
        for expert_id in range(self.num_experts):
            count = (gate_idx == expert_id).sum().item()
            expert_counts[expert_id] = count
        
        total_activations = sum(expert_counts.values())
        
        self._gate_distribution_history[self._global_step] = expert_counts.copy()
        self._save_distribution_to_json(expert_counts, "overall")
        
        text_expert_counts = None
        image_expert_counts = None
        
        if modality is not None:
            text_mask = (modality == 0)
            image_mask = (modality == 1)
            video_mask = (modality == 2)
            
            modalities = [
                ("text", text_mask),
                ("image", image_mask),
                ("video", video_mask)
            ]
            
            for modality_name, mask in modalities:
                if mask.any():
                    modality_gate_idx = gate_idx[mask]
                    modality_gate_score = gate_score[mask]
                    modality_expert_counts = {}
                    for expert_id in range(self.num_experts):
                        count = (modality_gate_idx == expert_id).sum().item()
                        modality_expert_counts[expert_id] = count
                    
                    modality_total = sum(modality_expert_counts.values())
                    
                    if modality_name not in self._modality_gate_distribution_history:
                        self._modality_gate_distribution_history[modality_name] = {}
                    self._modality_gate_distribution_history[modality_name][self._global_step] = modality_expert_counts.copy()
                    self._save_distribution_to_json(modality_expert_counts, modality_name)
                    
                    if modality_name == "text":
                        text_expert_counts = modality_expert_counts
                    elif modality_name == "image":
                        image_expert_counts = modality_expert_counts
                    
                    if modality_total > 0:
                        self._log_to_mlflow_modality_gates(
                            modality_expert_counts, modality_total, modality_gate_score, modality_name
                        )
        
        if total_activations > 0:
            self._log_to_mlflow_gates(expert_counts, total_activations, gate_score)
        
        if domain_id is not None:
            domain_expert_counts = {}
            for expert_id in range(self.num_experts):
                count = (gate_idx == expert_id).sum().item()
                domain_expert_counts[expert_id] = count
            
            if domain_id not in self._domain_gate_distribution_history:
                self._domain_gate_distribution_history[domain_id] = {}
            self._domain_gate_distribution_history[domain_id][self._global_step] = domain_expert_counts.copy()
            self._save_distribution_to_json(domain_expert_counts, f"domain_{domain_id}")
        

        self._log_all_plots_to_mlflow(
            overall_expert_counts=expert_counts,
            overall_gate_score=gate_score,
            text_expert_counts=text_expert_counts,
            image_expert_counts=image_expert_counts,
            domain_id=domain_id
        )
    
    
    def _log_all_plots_to_mlflow(self, overall_expert_counts, overall_gate_score, 
                                   text_expert_counts=None, image_expert_counts=None, domain_id=None):
        if self._mlflow_logger is None or self._visualizer is None:
            return
        
        overall_heatmap_bytes = self._visualizer.create_distribution_heatmap(
            self._gate_distribution_history, "overall", self._global_step
        )
        
        overall_histogram_bytes = self._visualizer.create_expert_activation_histogram(overall_expert_counts)
        
        text_history = self._modality_gate_distribution_history.get("text", {})
        image_history = self._modality_gate_distribution_history.get("image", {})
        combined_plot_bytes = self._visualizer.create_modality_combined_plot(
            text_history=text_history if text_history else None,
            image_history=image_history if image_history else None,
            text_expert_counts=text_expert_counts,
            image_expert_counts=image_expert_counts,
            global_step=self._global_step
        )
        
        domain_plot_bytes = None
        if domain_id is not None:
            domain_history = self._domain_gate_distribution_history.get(domain_id, {})
            current_expert_counts = domain_history.get(self._global_step, {}) if domain_history else {}
            domain_plot_bytes = self._visualizer.create_domain_plot(
                domain_id, domain_history, self._global_step, current_expert_counts
            )
        
        all_domains_plot_bytes = self._visualizer.create_all_domains_combined_plot(
            self._domain_gate_distribution_history, self._global_step
        )
        
        self._mlflow_logger.log_all_plots(
            layer_id=self._layer_id,
            global_step=self._global_step,
            overall_heatmap_bytes=overall_heatmap_bytes,
            overall_histogram_bytes=overall_histogram_bytes,
            combined_plot_bytes=combined_plot_bytes,
            domain_plot_bytes=domain_plot_bytes,
            all_domains_plot_bytes=all_domains_plot_bytes,
            domain_id=domain_id
        )
    


    def enable_logging(self, log_gates=True, log_activations=True, log_frequency=100):
        self._log_gates = log_gates
        self._log_activations = log_activations
        self._log_frequency = log_frequency


    def set_global_step(self, global_step):
        self._global_step = global_step


    def set_layer_id(self, layer_id):
        self._layer_id = layer_id
    

    def set_special_tokens(self, soi_id=None, eoi_id=None, sov_id=None, eov_id=None):
        self._soi_id = soi_id
        self._eoi_id = eoi_id
        self._sov_id = sov_id
        self._eov_id = eov_id
    

    def set_mlflow_logger(self, mlflow_logger):
        self._mlflow_logger = mlflow_logger
    

    def set_visualizer(self, visualizer):
        self._visualizer = visualizer
    

    def _get_token_modality(self, input_ids_flat):
        if input_ids_flat is None or self._soi_id is None or self._eoi_id is None:
            return None
        modality = torch.zeros(input_ids_flat.shape[0], dtype=torch.long, device=input_ids_flat.device)
        soi_positions = (input_ids_flat == self._soi_id).nonzero(as_tuple=True)[0]
        eoi_positions = (input_ids_flat == self._eoi_id).nonzero(as_tuple=True)[0]
        for soi_pos, eoi_pos in zip(soi_positions, eoi_positions):
            if soi_pos < eoi_pos:
                modality[soi_pos:eoi_pos+1] = 1  # 1 = image tokens
        if self._sov_id is not None and self._eov_id is not None:
            sov_positions = (input_ids_flat == self._sov_id).nonzero(as_tuple=True)[0]
            eov_positions = (input_ids_flat == self._eov_id).nonzero(as_tuple=True)[0]
            
            for sov_pos, eov_pos in zip(sov_positions, eov_positions):
                if sov_pos < eov_pos:
                    modality[sov_pos:eov_pos+1] = 2  # 2 = video tokens
        
        return modality

    
    def _log_to_mlflow_gates(self, expert_counts, total_activations, gate_score):
        if self._mlflow_logger is None:
            return
        self._mlflow_logger.log_gate_metrics(
            layer_id=self._layer_id,
            global_step=self._global_step,
            expert_counts=expert_counts,
            total_activations=total_activations,
            gate_score_mean=gate_score.mean().item(),
            gate_score_std=gate_score.std().item()
        )
    

    def _log_to_mlflow_modality_gates(self, expert_counts, total_activations, gate_score, modality_name):
        if self._mlflow_logger is None:
            return
        
        self._mlflow_logger.log_modality_gate_metrics(
            layer_id=self._layer_id,
            global_step=self._global_step,
            expert_counts=expert_counts,
            total_activations=total_activations,
            gate_score_mean=gate_score.mean().item(),
            gate_score_std=gate_score.std().item(),
            modality_name=modality_name
        )
        
    
    def _save_distribution_to_json(self, expert_counts, modality_name="overall"):
        json_dir = "./gate_distributions"
        if self._layer_id is not None:
            json_dir = os.path.join(json_dir, f"layer_{self._layer_id}")
        os.makedirs(json_dir, exist_ok=True)
        filename = f"gate_distribution_{modality_name}_step_{self._global_step}.json"
        filepath = os.path.join(json_dir, filename)
        data = {
            "step": self._global_step,
            "layer_id": self._layer_id,
            "modality": modality_name,
            "expert_counts": expert_counts,
            "total_activations": sum(expert_counts.values())
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    

    
    def log_distribution_heatmap_to_mlflow(self, modality_name="overall", alpha_value=None):
        if self._mlflow_logger is None or self._visualizer is None:
            return
        
        if modality_name == "overall":
            history = self._gate_distribution_history
        else:
            history = self._modality_gate_distribution_history.get(modality_name, {})
        
        if not history:
            return
        
        heatmap_bytes = self._visualizer.create_distribution_heatmap(
            history, modality_name, self._global_step, alpha_value
        )
        if heatmap_bytes is None:
            return
        
        self._mlflow_logger.log_distribution_heatmap(
            layer_id=self._layer_id,
            global_step=self._global_step,
            heatmap_bytes=heatmap_bytes,
            modality_name=modality_name
        )