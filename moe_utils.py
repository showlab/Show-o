import json
import os
import tempfile
import io
import base64
import logging
import copy
from collections import defaultdict
from typing import List, Dict, Optional

import mlflow
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch
import torch.nn as nn
from mlflow.tracking import MlflowClient

from models.moe_gates.soft_router import SoftTopKRouter
from models.phi import PhiMLP, PhiConfig
from models.moe_gates.naive_gate import NaiveGate
from models.moe_gates.switch_gate import SwitchGate
from models.moe_gates.faster_gate import FasterGate
from models.moe_gates.gshard_gate import GShardGate


logger = logging.getLogger(__name__)

# Константы для статистик
STATS_TYPES = ['max_abs', 'min_abs', 'var']


class Stats:
    """Класс для сбора и хранения статистик активаций"""
    def __init__(self, device: str = "cpu"):
        self._lists: Dict[str, List[torch.Tensor]] = {name: [] for name in STATS_TYPES}
        self.device = device

    def append(self, name: str, tensor: torch.Tensor) -> None:
        if name not in self._lists:
            raise KeyError(f"Unknown stat name '{name}'")
        self._lists[name].append(tensor.detach().to('cpu'))

    def append_many(self, stats: Dict[str, torch.Tensor]) -> None:
        for k, v in stats.items():
            if v is None:
                continue
            self.append(k, v)

    @torch.no_grad()
    def collect_from_tensor(self, out: torch.Tensor) -> None:
        """Собирает статистики из тензора активаций"""
        max_abs = out.abs().amax(dim=1, keepdim=False)           # (batch_size,)
        var = out.var(dim=1, unbiased=False, keepdim=False)      # (batch_size,)
        min_abs = out.abs().amin(dim=1, keepdim=False)   
        
        self.append("max_abs", max_abs)
        self.append("var", var)
        self.append("min_abs", min_abs)

    def cat(self, name: str) -> Optional[torch.Tensor]:
        """Объединяет все тензоры для данной статистики"""
        lst = self._lists.get(name)
        if not lst:
            return None
        return torch.cat(lst, dim=0)
    
    def clear(self) -> None:
        """Очищает все собранные данные"""
        for k in list(self._lists.keys()):
            self._lists[k].clear()

    def __repr__(self):
        s = ", ".join(f"{k}: {len(v)}" for k, v in self._lists.items())
        return f"Stats({s})"


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
        # Модальность-специфичная инициализация: первые num_experts//2 для текста, вторые для изображений
        num_text_experts = num_experts // 2
        # Bias для логитов гейта: большие значения для "правильных" экспертов
        tot_expert = num_experts * 4  # world_size=4
        self.register_buffer('modality_bias_text', torch.zeros(tot_expert))
        self.register_buffer('modality_bias_image', torch.zeros(tot_expert))
        # Первые num_text_experts экспертов на каждом шарде получают +bias для текста
        # Вторые num_text_experts — для изображений
        init_bias_val = 10.0  # Большой bias для жесткого разделения
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
        # История распределений гейтов для heatmap: {step: {expert_id: count}}
        self._gate_distribution_history = {}
        self._modality_gate_distribution_history = {}  # {modality_name: {step: {expert_id: count}}}

    def set_global_step(self, global_step):
        self._global_step = global_step


    def forward(self, hidden_states, input_ids=None, temperature: Optional[float] = None):
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
            self._log_gate_distribution(gate_idx, gate_score.detach(), input_ids)

        return output
    def _log_gate_distribution(self, gate_idx, gate_score, input_ids=None):
        import logging
        logger = logging.getLogger(__name__)
        
        # Определяем модальность токенов
        modality = self._get_token_modality(input_ids.view(-1) if input_ids is not None else None)
        
        # Общее распределение гейтов
        expert_counts = {}
        for expert_id in range(self.num_experts):
            count = (gate_idx == expert_id).sum().item()
            expert_counts[expert_id] = count
        
        total_activations = sum(expert_counts.values())
        
        # Всегда сохраняем данные (даже если total_activations == 0)
        self._gate_distribution_history[self._global_step] = expert_counts.copy()
        self._save_distribution_to_json(expert_counts, "overall")
        
        # Сохраняем expert_counts для text и image
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
                    
                    # Всегда сохраняем данные
                    if modality_name not in self._modality_gate_distribution_history:
                        self._modality_gate_distribution_history[modality_name] = {}
                    self._modality_gate_distribution_history[modality_name][self._global_step] = modality_expert_counts.copy()
                    self._save_distribution_to_json(modality_expert_counts, modality_name)
                    
                    if modality_name == "text":
                        text_expert_counts = modality_expert_counts
                    elif modality_name == "image":
                        image_expert_counts = modality_expert_counts
                    
                    # Логируем метрики
                    if modality_total > 0:
                        self._log_to_mlflow_modality_gates(
                            modality_expert_counts, modality_total, modality_gate_score, modality_name
                        )
        
        # Логируем метрики overall
        if total_activations > 0:
            self._log_to_mlflow_gates(expert_counts, total_activations, gate_score)
        
        # Создаем и отправляем все графики одновременно
        self._log_all_plots_to_mlflow(
            overall_expert_counts=expert_counts,
            overall_gate_score=gate_score,
            text_expert_counts=text_expert_counts,
            image_expert_counts=image_expert_counts
        )
    
    
    def _log_all_plots_to_mlflow(self, overall_expert_counts, overall_gate_score, 
                                   text_expert_counts=None, image_expert_counts=None):
        """Создает все графики одновременно и отправляет их в MLflow за один раз.
        Всегда создает графики, даже если данных нет (показывает "No data")."""
        # Получаем client и run_id
        if hasattr(self, '_mlflow_client') and hasattr(self, '_mlflow_run_id'):
            client = self._mlflow_client
            run_id = self._mlflow_run_id
        else:
            run_id = mlflow.active_run().info.run_id if mlflow.active_run() else None
            if run_id is None:
                return
            client = MlflowClient()
        
        layer_prefix = f"moe/layer_{self._layer_id}" if self._layer_id is not None else "moe"
        temp_dir = tempfile.mkdtemp()
        
        try:
            # 1. Создаем overall heatmap (всегда создаем)
            overall_heatmap_bytes = self._create_distribution_heatmap(
                self._gate_distribution_history, "overall"
            )
            
            # 2. Создаем overall expert histogram (всегда создаем)
            overall_histogram_bytes = self._create_expert_activation_histogram(overall_expert_counts)
            
            # 3. Создаем combined plot (text + image) (всегда создаем)
            text_history = self._modality_gate_distribution_history.get("text", {})
            image_history = self._modality_gate_distribution_history.get("image", {})
            combined_plot_bytes = self._create_modality_combined_plot(
                text_history=text_history if text_history else None,
                image_history=image_history if image_history else None,
                text_expert_counts=text_expert_counts,
                image_expert_counts=image_expert_counts
            )
            
            # Отправляем все графики в MLflow (всегда отправляем)
            tmp_file = os.path.join(temp_dir, f"gate_distribution_heatmap_step_{self._global_step}.png")
            with open(tmp_file, 'wb') as f:
                f.write(overall_heatmap_bytes)
            client.log_artifact(run_id, tmp_file, layer_prefix)
            print(f"📊 Heatmap распределений (overall) отправлена в MLflow: {layer_prefix}/gate_distribution_heatmap_step_{self._global_step}.png")
            
            tmp_file = os.path.join(temp_dir, f"expert_token_counts_step_{self._global_step}.png")
            with open(tmp_file, 'wb') as f:
                f.write(overall_histogram_bytes)
            client.log_artifact(run_id, tmp_file, layer_prefix)
            print(f"📊 Количество токенов по экспертам отправлено: {layer_prefix}/expert_token_counts_step_{self._global_step}.png")
            
            tmp_file = os.path.join(temp_dir, f"gate_distribution_combined_text_image_step_{self._global_step}.png")
            with open(tmp_file, 'wb') as f:
                f.write(combined_plot_bytes)
            client.log_artifact(run_id, tmp_file, layer_prefix)
            print(f"📊 Объединенный график (text + image) отправлен в MLflow: {layer_prefix}/gate_distribution_combined_text_image_step_{self._global_step}.png")
        
        except Exception as e:
            print(f"❌ Ошибка логирования графиков: {e}")
        finally:
            # Очистка временных файлов
            for f in os.listdir(temp_dir):
                os.unlink(os.path.join(temp_dir, f))
            os.rmdir(temp_dir)

    def _log_activation_stats(self, expert_stats):
        self._log_to_mlflow_activations(expert_stats)

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

    def _create_gate_histogram(self, gate_score):
        """Создает гистограмму распределения гейтов"""
        plt.figure(figsize=(10, 6))
        gate_weights = gate_score.detach().cpu().numpy().flatten()
        plt.hist(gate_weights, bins=50, alpha=0.7, color='blue', edgecolor='black')
        plt.title(f'Gate Weights Distribution (Overall) - Layer {self._layer_id} - Step {self._global_step}')
        plt.xlabel('Gate Weight Value')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        mean_val = np.mean(gate_weights)
        std_val = np.std(gate_weights)
        plt.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.4f}')
        plt.axvline(mean_val + std_val, color='orange', linestyle=':', label=f'+1σ: {mean_val + std_val:.4f}')
        plt.axvline(mean_val - std_val, color='orange', linestyle=':', label=f'-1σ: {mean_val - std_val:.4f}')
        plt.legend()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        plt.close()
        
        return buf.getvalue()
    
    def _log_to_mlflow_gates(self, expert_counts, total_activations, gate_score):
        """Логирует метрики гейтов в MLflow"""
        import tempfile
        import os
        import mlflow
        
        try:
            # Используем переданный client или создаем новый
            if hasattr(self, '_mlflow_client') and hasattr(self, '_mlflow_run_id'):
                client = self._mlflow_client
                run_id = self._mlflow_run_id
            else:
                from mlflow.tracking import MlflowClient
                
                # Получаем текущий run_id
                run_id = mlflow.active_run().info.run_id if mlflow.active_run() else None
                if run_id is None:
                    return
                
                client = MlflowClient()
            
            # Логируем агрегированные метрики для слоя
            layer_prefix = f"moe/layer_{self._layer_id}"
            
            # Агрегированная статистика экспертов
            expert_balance = max(expert_counts.values()) - min(expert_counts.values()) if expert_counts else 0
            client.log_metric(run_id, f"{layer_prefix}/expert_balance", expert_balance, step=self._global_step)
            client.log_metric(run_id, f"{layer_prefix}/total_activations", total_activations, step=self._global_step)
            
            # Статистика весов
            client.log_metric(run_id, f"{layer_prefix}/gate_weights_mean", gate_score.mean().item(), step=self._global_step)
            client.log_metric(run_id, f"{layer_prefix}/gate_weights_std", gate_score.std().item(), step=self._global_step)
            
            # Общая гистограмма весов гейтов удалена - нужны только модальность-специфичные
            # Графики теперь логируются через _log_all_plots_to_mlflow
            
        except Exception as e:
            # Игнорируем ошибки MLflow, чтобы не прерывать обучение
            pass
    
    def _log_to_mlflow_modality_gates(self, expert_counts, total_activations, gate_score, modality_name):
        if hasattr(self, '_mlflow_client') and hasattr(self, '_mlflow_run_id'):
            client = self._mlflow_client
            run_id = self._mlflow_run_id
        else:
            run_id = mlflow.active_run().info.run_id if mlflow.active_run() else None
            if run_id is None:
                return
            client = MlflowClient()
        
        # Логируем только метрики для конкретной модальности (без графиков)
        layer_prefix = f"moe/layer_{self._layer_id}/{modality_name}"
        
        # Агрегированная статистика экспертов для данной модальности
        expert_balance = max(expert_counts.values()) - min(expert_counts.values()) if expert_counts else 0
        client.log_metric(run_id, f"{layer_prefix}/expert_balance", expert_balance, step=self._global_step)
        client.log_metric(run_id, f"{layer_prefix}/total_activations", total_activations, step=self._global_step)
        
        # Статистика весов для данной модальности
        client.log_metric(run_id, f"{layer_prefix}/gate_weights_mean", gate_score.mean().item(), step=self._global_step)
        client.log_metric(run_id, f"{layer_prefix}/gate_weights_std", gate_score.std().item(), step=self._global_step)
        
    
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
    

    def _create_distribution_heatmap(self, distribution_history, modality_name="overall", alpha_value=None):
        """Создает heatmap. Всегда возвращает bytes, даже если данных нет."""
        if not distribution_history:
            # Создаем пустой график с сообщением "No data"
            fig, ax = plt.subplots(figsize=(14, 8))
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes, fontsize=20)
            ax.set_title(f'Gate Distribution ({modality_name})', fontsize=14, fontweight='bold')
            ax.set_xlabel('Iteration', fontsize=12)
            ax.set_ylabel('Expert ID', fontsize=12)
            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            plt.close()
            return buf.getvalue()
        
        steps = sorted(distribution_history.keys())
        if not steps:
            # Создаем пустой график с сообщением "No data"
            fig, ax = plt.subplots(figsize=(14, 8))
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes, fontsize=20)
            ax.set_title(f'Gate Distribution ({modality_name})', fontsize=14, fontweight='bold')
            ax.set_xlabel('Iteration', fontsize=12)
            ax.set_ylabel('Expert ID', fontsize=12)
            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            plt.close()
            return buf.getvalue()
        all_expert_ids = set()
        for step_counts in distribution_history.values():
            all_expert_ids.update(step_counts.keys())
        expert_ids = sorted(all_expert_ids)
        matrix = np.zeros((len(expert_ids), len(steps)))
        
        for col_idx, step in enumerate(steps):
            step_counts = distribution_history[step]
            total = sum(step_counts.values())
            if total > 0:
                for row_idx, expert_id in enumerate(expert_ids):
                    matrix[row_idx, col_idx] = step_counts.get(expert_id, 0) / total
        
        fig, ax = plt.subplots(figsize=(14, 8))
        title_suffix = f" — constant α={alpha_value}" if alpha_value is not None else ""
        title = f"Gate Distribution{title_suffix}"
        if modality_name != "overall":
            title = f"Gate Distribution ({modality_name}){title_suffix}"
        
        im = ax.imshow(matrix, aspect='auto', cmap='viridis', interpolation='nearest')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Expert ID', fontsize=12)
        
        step_indices = np.arange(len(steps))
        if len(steps) > 20:
            tick_step = max(1, len(steps) // 20)
            ax.set_xticks(step_indices[::tick_step])
            ax.set_xticklabels([steps[i] for i in step_indices[::tick_step]], rotation=45)
        else:
            ax.set_xticks(step_indices)
            ax.set_xticklabels(steps, rotation=45)
        
        ax.set_yticks(np.arange(len(expert_ids)))
        ax.set_yticklabels(expert_ids)
        
        # Добавляем colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Gate Distribution', fontsize=11)
        
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        plt.close()
        
        return buf.getvalue()
    
    def log_distribution_heatmap_to_mlflow(self, modality_name="overall", alpha_value=None):
        if modality_name == "overall":
            history = self._gate_distribution_history
        else:
            history = self._modality_gate_distribution_history.get(modality_name, {})
        
        if not history:
            return
        
        # Создаем heatmap
        heatmap_bytes = self._create_distribution_heatmap(history, modality_name, alpha_value)
        if heatmap_bytes is None:
            return
        
        # Получаем MLflow client и run_id
        if hasattr(self, '_mlflow_client') and hasattr(self, '_mlflow_run_id'):
            client = self._mlflow_client
            run_id = self._mlflow_run_id
        else:
            run_id = mlflow.active_run().info.run_id if mlflow.active_run() else None
            if run_id is None:
                return
            client = MlflowClient()
        
        temp_dir = tempfile.mkdtemp()
        layer_prefix = f"moe/layer_{self._layer_id}" if self._layer_id is not None else "moe"
        suffix = f"_{modality_name}" if modality_name != "overall" else ""
        filename = f"gate_distribution_heatmap{suffix}_step_{self._global_step}.png"
        tmp_file_path = os.path.join(temp_dir, filename)
        with open(tmp_file_path, 'wb') as f:
            f.write(heatmap_bytes)
        client.log_artifact(run_id, tmp_file_path, layer_prefix)
        print(f"📊 Heatmap распределений ({modality_name}) отправлена в MLflow: {layer_prefix}/{filename}")
        os.unlink(tmp_file_path)
        os.rmdir(temp_dir)
        

    def _log_to_mlflow_activations(self, expert_stats):
        try:
            # Используем переданный client или создаем новый
            if hasattr(self, '_mlflow_client') and hasattr(self, '_mlflow_run_id'):
                client = self._mlflow_client
                run_id = self._mlflow_run_id
            else:
                import mlflow
                from mlflow.tracking import MlflowClient
                # Получаем текущий run_id
                run_id = mlflow.active_run().info.run_id if mlflow.active_run() else None
                if run_id is None:
                    return
                client = MlflowClient()
            # Только гистограммы по-прежнему должны логироваться (см. другое место вызова)
            pass
        except Exception as e:
            # Игнорируем ошибки MLflow, чтобы не прерывать обучение
            pass

    def _create_expert_activation_histogram(self, expert_counts, modality_name=None, ax=None):
        """Создает гистограмму активаций экспертов (количество столбцов = количество экспертов)
        Если ax не передан, создает новый figure и возвращает bytes.
        Если ax передан, рисует на нем и ничего не возвращает."""
        experts = list(expert_counts.keys())
        activations = list(expert_counts.values())
        
        create_new_figure = (ax is None)
        if create_new_figure:
            plt.figure(figsize=(10, 6))
            ax = plt.gca()
        
        # Создаем столбчатую диаграмму
        bars = ax.bar(experts, activations, alpha=0.7, color='green', edgecolor='black')
        title = f'Expert Token Counts'
        if modality_name:
            title += f' ({modality_name})'
        ax.set_title(title, fontsize=11)
        ax.set_xlabel('Expert ID', fontsize=10)
        ax.set_ylabel('Number of Activations', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Добавляем значения на столбцы
        for bar, count in zip(bars, activations):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    str(count), ha='center', va='bottom', fontsize=8)
        
        # Добавляем статистики
        total_activations = sum(activations)
        balance = max(activations) - min(activations) if activations else 0
        ax.text(0.02, 0.98, f'Total: {total_activations}\nBalance: {balance}', 
                transform=ax.transAxes, verticalalignment='top', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Если создали новый figure, возвращаем bytes
        if create_new_figure:
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            plt.close()
            return buf.getvalue()
        
    def _create_modality_combined_plot(self, text_history=None, image_history=None, 
                                       text_expert_counts=None, image_expert_counts=None, alpha_value=None):
        """Создает объединенный plot с 4 subplot'ами: 2 heatmap (text, image) и 2 гистограммы (text, image)"""
        fig, axes = plt.subplots(2, 2, figsize=(20, 12))
        fig.suptitle(f'Gate Distribution Analysis - Layer {self._layer_id} - Step {self._global_step}', 
                     fontsize=16, fontweight='bold')
        
        # Верхний ряд: Heatmap для text и image
        # Heatmap для text (верхний левый)
        if text_history and len(text_history) > 0:
            success = self._create_single_heatmap(axes[0, 0], text_history, "text", alpha_value)
            if not success:
                axes[0, 0].text(0.5, 0.5, 'No text data', ha='center', va='center', transform=axes[0, 0].transAxes, fontsize=14)
                axes[0, 0].set_title('Gate Distribution (text)', fontsize=12, fontweight='bold')
                axes[0, 0].set_xlabel('Iteration', fontsize=10)
                axes[0, 0].set_ylabel('Expert ID', fontsize=10)
        else:
            axes[0, 0].text(0.5, 0.5, 'No text data', ha='center', va='center', transform=axes[0, 0].transAxes, fontsize=14)
            axes[0, 0].set_title('Gate Distribution (text)', fontsize=12, fontweight='bold')
            axes[0, 0].set_xlabel('Iteration', fontsize=10)
            axes[0, 0].set_ylabel('Expert ID', fontsize=10)
        
        # Heatmap для image (верхний правый)
        if image_history and len(image_history) > 0:
            success = self._create_single_heatmap(axes[0, 1], image_history, "image", alpha_value)
            if not success:
                axes[0, 1].text(0.5, 0.5, 'No image data', ha='center', va='center', transform=axes[0, 1].transAxes, fontsize=14)
                axes[0, 1].set_title('Gate Distribution (image)', fontsize=12, fontweight='bold')
                axes[0, 1].set_xlabel('Iteration', fontsize=10)
                axes[0, 1].set_ylabel('Expert ID', fontsize=10)
        else:
            axes[0, 1].text(0.5, 0.5, 'No image data', ha='center', va='center', transform=axes[0, 1].transAxes, fontsize=14)
            axes[0, 1].set_title('Gate Distribution (image)', fontsize=12, fontweight='bold')
            axes[0, 1].set_xlabel('Iteration', fontsize=10)
            axes[0, 1].set_ylabel('Expert ID', fontsize=10)
        
        # Нижний ряд: Гистограммы активаций экспертов
        # Гистограмма для text (нижний левый)
        if text_expert_counts and len(text_expert_counts) > 0:
            bars = axes[1, 0].bar(list(text_expert_counts.keys()), list(text_expert_counts.values()), 
                          alpha=0.7, color='blue', edgecolor='black')
            axes[1, 0].set_title('Expert Token Counts (text)', fontsize=12, fontweight='bold')
            axes[1, 0].set_xlabel('Expert ID', fontsize=10)
            axes[1, 0].set_ylabel('Number of Activations', fontsize=10)
            axes[1, 0].grid(True, alpha=0.3)
            # Добавляем значения на столбцы
            for bar, count in zip(bars, text_expert_counts.values()):
                axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                        str(count), ha='center', va='bottom', fontsize=8)
        else:
            axes[1, 0].text(0.5, 0.5, 'No text data', ha='center', va='center', transform=axes[1, 0].transAxes, fontsize=14)
            axes[1, 0].set_title('Expert Token Counts (text)', fontsize=12, fontweight='bold')
            axes[1, 0].set_xlabel('Expert ID', fontsize=10)
            axes[1, 0].set_ylabel('Number of Activations', fontsize=10)
        
        # Гистограмма для image (нижний правый)
        if image_expert_counts and len(image_expert_counts) > 0:
            bars = axes[1, 1].bar(list(image_expert_counts.keys()), list(image_expert_counts.values()), 
                          alpha=0.7, color='orange', edgecolor='black')
            axes[1, 1].set_title('Expert Token Counts (image)', fontsize=12, fontweight='bold')
            axes[1, 1].set_xlabel('Expert ID', fontsize=10)
            axes[1, 1].set_ylabel('Number of Activations', fontsize=10)
            axes[1, 1].grid(True, alpha=0.3)
            # Добавляем значения на столбцы
            for bar, count in zip(bars, image_expert_counts.values()):
                axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                        str(count), ha='center', va='bottom', fontsize=8)
        else:
            axes[1, 1].text(0.5, 0.5, 'No image data', ha='center', va='center', transform=axes[1, 1].transAxes, fontsize=14)
            axes[1, 1].set_title('Expert Token Counts (image)', fontsize=12, fontweight='bold')
            axes[1, 1].set_xlabel('Expert ID', fontsize=10)
            axes[1, 1].set_ylabel('Number of Activations', fontsize=10)
        
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        plt.close()
        
        return buf.getvalue()

    def _create_single_heatmap(self, ax, distribution_history, modality_name="overall", alpha_value=None):
        """Создает одну heatmap на переданной оси. Если данных нет, ничего не рисует (вызывающий код должен показать 'No data')"""
        if not distribution_history:
            return False
        
        steps = sorted(distribution_history.keys())
        if not steps:
            return False
        
        all_expert_ids = set()
        for step_counts in distribution_history.values():
            all_expert_ids.update(step_counts.keys())
        expert_ids = sorted(all_expert_ids)
        
        if len(expert_ids) == 0:
            return False
        
        matrix = np.zeros((len(expert_ids), len(steps)))
        has_data = False
        
        for col_idx, step in enumerate(steps):
            step_counts = distribution_history[step]
            total = sum(step_counts.values())
            if total > 0:
                has_data = True
                for row_idx, expert_id in enumerate(expert_ids):
                    matrix[row_idx, col_idx] = step_counts.get(expert_id, 0) / total
        
        if not has_data or matrix.max() == 0:
            return False
        
        title_suffix = f" — constant α={alpha_value}" if alpha_value is not None else ""
        title = f"Gate Distribution ({modality_name}){title_suffix}"
        
        im = ax.imshow(matrix, aspect='auto', cmap='viridis', interpolation='nearest', vmin=0, vmax=1)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Iteration', fontsize=10)
        ax.set_ylabel('Expert ID', fontsize=10)
        
        step_indices = np.arange(len(steps))
        if len(steps) > 20:
            tick_step = max(1, len(steps) // 20)
            ax.set_xticks(step_indices[::tick_step])
            ax.set_xticklabels([steps[i] for i in step_indices[::tick_step]], rotation=45, fontsize=8)
        else:
            ax.set_xticks(step_indices)
            ax.set_xticklabels(steps, rotation=45, fontsize=8)
        
        ax.set_yticks(np.arange(len(expert_ids)))
        ax.set_yticklabels(expert_ids, fontsize=8)
        
        plt.colorbar(im, ax=ax, label='Gate Distribution')
        return True


def patch_model_with_moe(model, count_layers_to_patch=3, num_experts=4, top_k=2, special_tokens=None, noise_std: float = 1e-2,
    temp_init: float = 100.0, temp_warmup_steps: int = 500, temp_gamma: float = 0.98, temp_min: float = 1.0,
    modality_init_hardness: float = 1.0, modality_init_steps: int = 1000, modality_init_hardness_min: float = 0.2,
    use_gumbel: bool = False):
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
                modality_init_hardness=modality_init_hardness,
                modality_init_steps=modality_init_steps,
                modality_init_hardness_min=modality_init_hardness_min,
                use_gumbel=use_gumbel
            )
            moe_layer.to(next(original_mlp.parameters()).device)
            print(f"    Инициализированы {num_experts} экспертов как копии FFN + шум (std={noise_std})")
            moe_layer.enable_logging(log_gates=True, log_activations=True, log_frequency=100)
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
    print("✓ Патчинг завершен")
    print(f"✓ Заменено слоев: {len(patched_layers)} из {total_layers} ({100*len(patched_layers)/total_layers:.1f}%)")
    print(f"✓ Слои с MoE: {patched_layers}")
    print(f"✓ Слои с оригинальным FFN (предобученным): {[i for i in range(total_layers) if i not in patched_layers]}")
    return model


def freeze_non_moe_params(model):
    print("❄️  Замораживаем все параметры кроме суффикса с MoE...")

    # Сначала заморозим всё
    for param in model.parameters():
        param.requires_grad = False
    
    # Найдем минимальный индекс слоя с MoE
    min_moe_layer = None
    for name, param in model.named_parameters():
        if 'showo.model.layers' in name and any(x in name for x in ['mlp.experts', 'mlp.gate', 'mlp.alpha']):
            # Извлекаем номер слоя из имени типа "showo.model.layers.21.mlp..."
            import re
            match = re.search(r'layers\.(\d+)\.', name)
            if match:
                layer_idx = int(match.group(1))
                if min_moe_layer is None or layer_idx < min_moe_layer:
                    min_moe_layer = layer_idx
    
    if min_moe_layer is None:
        print("⚠️  MoE слои не найдены")
        return model
    
    print(f"🔥 Размораживаем все слои начиная с {min_moe_layer} (первый MoE слой)")
    
    # Размораживаем весь суффикс начиная с первого MoE слоя
    trainable_params = 0
    total_params = 0
    trainable_param_names = []
    
    for name, param in model.named_parameters():
        total_params += param.numel()
        
        # Проверяем, относится ли параметр к слоям >= min_moe_layer
        if 'showo.model.layers' in name:
            import re
            match = re.search(r'layers\.(\d+)\.', name)
            if match:
                layer_idx = int(match.group(1))
                if layer_idx >= min_moe_layer:
                    param.requires_grad = True
                    trainable_params += param.numel()
                    trainable_param_names.append(name)
    
    print(f"✓ Параметров: {total_params:,} total, {trainable_params:,} trainable ({100*trainable_params/total_params:.1f}%)")
    if len(trainable_param_names) > 0:
        print(f"✓ Разморожено {len(trainable_param_names)} параметров в суффиксе (слои {min_moe_layer}+):")
        for name in trainable_param_names[:5]:  # показываем первые 5
            print(f"    - {name}")
        if len(trainable_param_names) > 5:
            print(f"    ... и еще {len(trainable_param_names) - 5}")
    return model


def patch_and_freeze_moe(
    model,
    count_layers_to_patch: int = 3, 
    num_experts: int = 8, 
    top_k: int = 2, 
    mlflow_client: Optional[mlflow.client.MlflowClient] = None,
    mlflow_run_id: Optional[str] = None,
    special_tokens: Optional[dict] = None,
    noise_std: float = 1e-2,
    temp_init: float = 100.0,
    temp_warmup_steps: int = 500,
    temp_gamma: float = 0.98,
    temp_min: float = 1,
    modality_init_hardness: float = 1.0,
    modality_init_steps: int = 1000,
    modality_init_hardness_min: float = 0.2,
    use_gumbel: bool = False
):
    model = patch_model_with_moe(model, count_layers_to_patch, num_experts, top_k, special_tokens, noise_std=noise_std,
        temp_init=temp_init, temp_warmup_steps=temp_warmup_steps, temp_gamma=temp_gamma, temp_min=temp_min,
        modality_init_hardness=modality_init_hardness, modality_init_steps=modality_init_steps,
        modality_init_hardness_min=modality_init_hardness_min, use_gumbel=use_gumbel)
    if mlflow_client is not None and mlflow_run_id is not None:
        for layer in model.showo.model.layers:
            if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'experts'):
                layer.mlp._mlflow_client = mlflow_client
                layer.mlp._mlflow_run_id = mlflow_run_id
    model = freeze_non_moe_params(model)
    return model


def save_moe_weights(model, path):
    moe_state = {
        k: v for k, v in model.state_dict().items() 
        if any(x in k for x in ['mlp.experts', 'mlp.gate', 'mlp.alpha'])
    }
    torch.save(moe_state, path)
    print(f"💾 Сохранено {len(moe_state)} MoE параметров в {path}")


def load_moe_weights(model, path):
    moe_weights = torch.load(path)
    model.load_state_dict(moe_weights, strict=False)
    print(f"📥 Загружено {len(moe_weights)} MoE параметров из {path}")
    return model


def get_activations(recorder, stat: str = "max_abs", layers: list[int] = None):
    """Получает активации для указанных слоев и статистики"""
    if stat not in STATS_TYPES:
        raise ValueError(f"stat must be one of {STATS_TYPES}, got {stat}")
    if layers is None:
        layers = list(range(24))
    layer_names = []
    acts_list = []

    for ind in layers:
        layer_name = f"showo.model.layers.{ind}.mlp.fc2"
        if layer_name not in recorder.outputs:
            continue

        stats_obj: Stats = recorder.outputs[layer_name]
        t = stats_obj.cat(stat)
        if t is None:
            continue

        t = t.detach().cpu().flatten()
        if t.numel() == 0:
            continue

        layer_names.append(layer_name)
        acts_list.append(t)

    return layer_names, acts_list


def create_stats_boxplots(recorder, stat: str = "max_abs", layers: list[int] = None,
                         figsize=(14,6), showfliers=False):
    """Создает boxplot для статистик активаций"""
    layer_names, acts_list = get_activations(recorder, stat=stat, layers=layers)

    if len(acts_list) == 0:
        print("No activations collected for the requested layers/stat.")
        return None

    data = [a.numpy() if isinstance(a, torch.Tensor) else np.asarray(a) for a in acts_list]
    plt.figure(figsize=figsize)
    labels = [ln.split(".")[3] if len(ln.split(".")) > 3 else ln for ln in layer_names]
    plt.boxplot(data, labels=labels, showfliers=showfliers)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel(stat)
    plt.title(f"Boxplots of '{stat}' across layers ({len(data)} layers)")
    plt.tight_layout()
    
    # Сохраняем в байты
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    
    return buf.getvalue()


def create_all_stats_boxplots(recorder, layers: list[int] = None, figsize=(18,6), showfliers=False):
    """Создает все boxplot'ы для всех статистик"""
    # Нормализуем в словарь
    if not isinstance(recorder, dict):
        recorders = {'': recorder}
    else:
        recorders = recorder
    
    stats = list(STATS_TYPES)
    n_stats = len(stats)
    n_recorders = len(recorders)
    
    # 2 строки: boxplots и средние
    fig, axes = plt.subplots(2, n_stats, figsize=(figsize[0], figsize[1] * 1.5), sharey=False)
    if n_stats == 1:
        axes = axes.reshape(-1, 1)
    
    colors = plt.cm.Set2(range(n_recorders))
    
    for col, stat in enumerate(stats):
        ax_box = axes[0, col]
        ax_mean = axes[1, col]
        
        # Собираем данные от всех рекордеров
        recorders_data = {}
        layer_names_all = None
        
        for rec_name, rec in recorders.items():
            layer_names, acts_list = get_activations(rec, stat=stat, layers=layers)
            if len(acts_list) == 0:
                continue
            
            if layer_names_all is None:
                layer_names_all = layer_names
            
            data = [a.numpy() if isinstance(a, torch.Tensor) else np.asarray(a) for a in acts_list]
            recorders_data[rec_name] = data
        
        if len(recorders_data) == 0:
            ax_box.text(0.5, 0.5, f"No data", ha='center', va='center')
            ax_box.set_title(stat)
            ax_mean.text(0.5, 0.5, f"No data", ha='center', va='center')
            continue
        
        # === ВЕРХНИЙ РЯД: Boxplots ===
        n_layers = len(next(iter(recorders_data.values())))
        width = 0.8 / n_recorders
        
        legend_patches = []
        for i, (rec_name, data) in enumerate(recorders_data.items()):
            positions = [j + 1 + (i - n_recorders/2 + 0.5) * width for j in range(n_layers)]
            bp = ax_box.boxplot(data, positions=positions, widths=width*0.9, 
                               showfliers=showfliers, patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor(colors[i])
            
            if rec_name:
                from matplotlib.patches import Patch
                legend_patches.append(Patch(facecolor=colors[i], label=rec_name))
        
        labels = [ln.split(".")[3] if len(ln.split("."))>3 else ln for ln in layer_names_all]
        ax_box.set_xticks(range(1, n_layers + 1))
        ax_box.set_xticklabels(labels, rotation=45, ha='right')
        ax_box.set_title(stat)
        
        if len(legend_patches) > 0:
            ax_box.legend(handles=legend_patches)
        for i, (rec_name, data) in enumerate(recorders_data.items()):
            means = [d.mean() for d in data]
            x = range(1, len(means) + 1)
            ax_mean.plot(x, means, marker='o', color=colors[i], 
                        label=rec_name if rec_name else None)
        
        ax_mean.set_xticks(range(1, n_layers + 1))
        ax_mean.set_xticklabels(labels, rotation=45, ha='right')
        ax_mean.set_ylabel("Mean")
        ax_mean.grid(True, alpha=0.3)
        
        if len(legend_patches) > 0:
            ax_mean.legend()
    
    plt.tight_layout()
    
    # Сохраняем в байты
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    
    return buf.getvalue()


class LayerOutputRecorder:
    """Рекордер для сбора статистик активаций из слоев модели"""
    def __init__(self, device='cuda'):
        self.outputs = defaultdict(lambda: Stats(device=device))
        self.inputs_shapes = defaultdict(list)
        self.handles = []
        self.device = device

    def build_hook_fn(self, name):
        def hook_fn(module, input_, output):
            with torch.no_grad():
                out = output.detach()
                self.outputs[name].collect_from_tensor(out)
                self.inputs_shapes[name].append(input_[0].shape)
        return hook_fn

    def register_hook(self, module_name, module):
        handle = module.register_forward_hook(self.build_hook_fn(module_name))
        self.handles.append(handle)

    def register_hooks(self, modules: list[tuple[str, torch.nn.Module]]) -> None:
        for module_name, module in modules:
            self.register_hook(module_name, module)

    def remove_hooks(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def clear(self):
        for stats in self.outputs.values():
            stats.clear()
        self.outputs.clear()
        self.inputs_shapes.clear()
        torch.cuda.empty_cache()


def log_stats_to_mlflow(recorder, mlflow_client, run_id, step, layer_prefix="activations"):
    """Логирует статистики активаций в MLflow"""
    try:
        # Создаем общий график всех статистик
        all_stats_bytes = create_all_stats_boxplots(recorder)
        
        if all_stats_bytes:
            # Сохраняем временный файл
            temp_dir = tempfile.mkdtemp()
            tmp_file_path = os.path.join(temp_dir, f"all_stats_step_{step}.png")
            
            with open(tmp_file_path, 'wb') as f:
                f.write(all_stats_bytes)
            
            # Логируем в MLflow
            mlflow_client.log_artifact(run_id, tmp_file_path, layer_prefix)
            print(f"📊 Статистики активаций отправлены: {layer_prefix}/all_stats_step_{step}.png")
            
            # Очищаем временный файл
            os.unlink(tmp_file_path)
            os.rmdir(temp_dir)
            
    except Exception as e:
        print(f"❌ Ошибка логирования статистик: {e}")


if __name__ == "__main__":
    from models import Showo
    model = Showo.from_pretrained("showlab/show-o-w-clip-vit")
    model = patch_and_freeze_moe(model, num_experts=4, top_k=2)

    print("\nГотово! Теперь можно обучать.")
    print("Пример:")
    print("  optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)")

