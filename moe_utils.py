"""
Утилиты для добавления MoE в модель Show-o
Минимальный код для патчинга и обучения

Использование:
    from moe_utils import patch_and_freeze_moe
    
    model = Showo.from_pretrained("showlab/show-o-w-clip-vit")
    model = patch_and_freeze_moe(model, num_experts=4, top_k=2)
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
import logging
import tempfile
import os
import mlflow
from models.moe_gates.soft_router import SoftTopKRouter
import copy
from models.phi import PhiMLP, PhiConfig
from models.moe_gates.naive_gate import NaiveGate
from models.moe_gates.switch_gate import SwitchGate
from models.moe_gates.faster_gate import FasterGate
from models.moe_gates.gshard_gate import GShardGate
from collections import defaultdict
from typing import List, Dict, Optional


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
    def __init__(self, num_experts, hidden_size, top_k, config: PhiConfig, template_mlp: Optional[nn.Module] = None, noise_std: float = 1e-3):
        super().__init__()
        self.gate = GShardGate(hidden_size, num_experts, world_size=4, top_k=top_k, gate_bias=True)
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.top_k = top_k
        self.experts = nn.ModuleList()
        # Init experts from ffn
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

    def set_global_step(self, global_step):
        self._global_step = global_step


    def forward(self, hidden_states, input_ids=None, temperature: Optional[float] = None):
        device = hidden_states.device
        batch_size, seq_len, hidden_size = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_size)  # [B*L, H]
        B = hidden_states_flat.shape[0]

        # 1) Получаем top-k индексы и веса от гейта (вся логика роутинга внутри гейта)
        gate_idx, gate_score = self.gate(hidden_states_flat, temperature=temperature)

        # Баланс-лосс устанавливается внутри GShardGate; ничего дополнительно не считаем здесь

        # 3) Считаем выход: сумма по экспертам с весами y_st * alpha (sparse по маскам)
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
        if total_activations > 0:
            self._log_to_mlflow_gates(expert_counts, total_activations, gate_score)
            if modality is not None:
                self._log_modality_specific_gates(gate_idx, gate_score, modality)
    
    def _log_modality_specific_gates(self, gate_idx, gate_score, modality):
        """Логирует распределение гейтов отдельно для каждой модальности"""
        import logging
        logger = logging.getLogger(__name__)
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
                expert_counts = {}
                for expert_id in range(self.num_experts):
                    count = (modality_gate_idx == expert_id).sum().item()
                    expert_counts[expert_id] = count
                
                total_activations = sum(expert_counts.values())
                if total_activations > 0:
                    self._log_to_mlflow_modality_gates(expert_counts, total_activations, modality_gate_score, modality_name)

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
            
            # Создаем и отправляем гистограмму активаций экспертов
            expert_histogram_bytes = self._create_expert_activation_histogram(expert_counts)
            
            # Логируем гистограмму активаций экспертов
            try:
                # Создаем файл с правильным именем
                temp_dir = tempfile.mkdtemp()
                tmp_file_path = os.path.join(temp_dir, f"expert_token_counts_step_{self._global_step}.png")
                
                with open(tmp_file_path, 'wb') as f:
                    f.write(expert_histogram_bytes)
                
                client.log_artifact(run_id, tmp_file_path, layer_prefix)
                print(f"📊 Количество токенов по экспертам отправлено: {layer_prefix}/expert_token_counts_step_{self._global_step}.png")
            except Exception as artifact_error:
                print(f"⚠️ Не удалось отправить гистограмму экспертов: {artifact_error}")
                # Fallback: сохраняем локально
                artifacts_dir = f"./mlruns/artifacts/{layer_prefix}"
                os.makedirs(artifacts_dir, exist_ok=True)
                local_path = f"{artifacts_dir}/expert_activations_histogram_step_{self._global_step}.png"
                with open(local_path, 'wb') as f:
                    f.write(expert_histogram_bytes)
                print(f"📊 Гистограмма экспертов сохранена локально: {local_path}")
            finally:
                os.unlink(tmp_file_path)
                os.rmdir(temp_dir)
            
        except Exception as e:
            # Игнорируем ошибки MLflow, чтобы не прерывать обучение
            print(f"❌ Ошибка логирования гистограммы: {e}")
    
    def _log_to_mlflow_modality_gates(self, expert_counts, total_activations, gate_score, modality_name):
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
        
        # Логируем метрики для конкретной модальности
        layer_prefix = f"moe/layer_{self._layer_id}/{modality_name}"
        
        # Агрегированная статистика экспертов для данной модальности
        expert_balance = max(expert_counts.values()) - min(expert_counts.values()) if expert_counts else 0
        client.log_metric(run_id, f"{layer_prefix}/expert_balance", expert_balance, step=self._global_step)
        client.log_metric(run_id, f"{layer_prefix}/total_activations", total_activations, step=self._global_step)
        
        # Статистика весов для данной модальности
        client.log_metric(run_id, f"{layer_prefix}/gate_weights_mean", gate_score.mean().item(), step=self._global_step)
        client.log_metric(run_id, f"{layer_prefix}/gate_weights_std", gate_score.std().item(), step=self._global_step)
        
        # Создаем и отправляем гистограмму для данной модальности
        histogram_bytes = self._create_modality_histogram(gate_score, modality_name)
        
        # Исправленный способ логирования изображений
        # Создаем файл с правильным именем
        temp_dir = tempfile.mkdtemp()
        tmp_file_path = os.path.join(temp_dir, f"gate_distribution_{modality_name}_step_{self._global_step}.png")
        
        with open(tmp_file_path, 'wb') as f:
            f.write(histogram_bytes)
        
        client.log_artifact(run_id, tmp_file_path, layer_prefix)
        print(f"📊 Распределение гейтов {modality_name} отправлено: {layer_prefix}/gate_distribution_{modality_name}_step_{self._global_step}.png")
        expert_histogram_bytes = self._create_expert_activation_histogram(expert_counts)
        emp_dir = tempfile.mkdtemp()
        tmp_file_path = os.path.join(temp_dir, f"expert_token_counts_{modality_name}_step_{self._global_step}.png")
        
        with open(tmp_file_path, 'wb') as f:
            f.write(expert_histogram_bytes)
        
        client.log_artifact(run_id, tmp_file_path, layer_prefix)
        print(f"📊 Количество токенов по экспертам {modality_name} отправлено: {layer_prefix}/expert_token_counts_{modality_name}_step_{self._global_step}.png")
    
    
    def _create_modality_histogram(self, gate_score, modality_name):
        """Создает гистограмму распределения гейтов для конкретной модальности"""
        plt.figure(figsize=(10, 6))
        
        # Создаем гистограмму весов гейтов
        gate_weights = gate_score.detach().cpu().numpy().flatten()
        
        plt.hist(gate_weights, bins=50, alpha=0.7, color='blue', edgecolor='black')
        plt.title(f'Gate Distribution - Layer {self._layer_id} - {modality_name.capitalize()} Tokens - Step {self._global_step}')
        plt.xlabel('Gate Weight Value')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        # Добавляем статистики на график
        mean_val = np.mean(gate_weights)
        std_val = np.std(gate_weights)
        plt.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.4f}')
        plt.axvline(mean_val + std_val, color='orange', linestyle=':', label=f'+1σ: {mean_val + std_val:.4f}')
        plt.axvline(mean_val - std_val, color='orange', linestyle=':', label=f'-1σ: {mean_val - std_val:.4f}')
        plt.legend()
        
        # Сохраняем в байты
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        plt.close()
        
        return buf.getvalue()

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

    def _create_expert_activation_histogram(self, expert_counts):
        """Создает гистограмму активаций экспертов (количество столбцов = количество экспертов)"""
        plt.figure(figsize=(10, 6))
        
        # Подготавливаем данные для гистограммы
        experts = list(expert_counts.keys())
        activations = list(expert_counts.values())
        
        # Создаем столбчатую диаграмму
        bars = plt.bar(experts, activations, alpha=0.7, color='green', edgecolor='black')
        plt.title(f'Expert Token Counts - Layer {self._layer_id} - Step {self._global_step}')
        plt.xlabel('Expert ID')
        plt.ylabel('Number of Activations')
        plt.grid(True, alpha=0.3)
        
        # Добавляем значения на столбцы
        for bar, count in zip(bars, activations):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    str(count), ha='center', va='bottom')
        
        # Добавляем статистики
        total_activations = sum(activations)
        balance = max(activations) - min(activations) if activations else 0
        plt.text(0.02, 0.98, f'Total: {total_activations}\nBalance: {balance}', 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Сохраняем в байты
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        plt.close()
        
        return buf.getvalue()


def patch_model_with_moe(model, count_layers_to_patch=3, num_experts=4, top_k=2, special_tokens=None, noise_std: float = 1e-2,
    temp_init: float = 100.0, temp_warmup_steps: int = 500, temp_gamma: float = 0.98, temp_min: float = 1.0):
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
                noise_std=noise_std
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
    temp_min: float = 1
):
    model = patch_model_with_moe(model, count_layers_to_patch, num_experts, top_k, special_tokens, noise_std=noise_std,
        temp_init=temp_init, temp_warmup_steps=temp_warmup_steps, temp_gamma=temp_gamma, temp_min=temp_min)
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

