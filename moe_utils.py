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
from models.phi import PhiMLP, PhiConfig
from models.moe_gates.gshard_gate import GShardGate


logger = logging.getLogger(__name__)

class SmallPhiMLP(nn.Module):
    """Уменьшенный PhiMLP для экспертов MoE"""
    def __init__(self, config: PhiConfig, scale_factor: int = 4):
        super().__init__()
        self.config = config
        small_intermediate_size = config.intermediate_size // scale_factor
        if config.hidden_act == "gelu_new":
            self.activation_fn = torch.nn.functional.gelu
        else:
            self.activation_fn = getattr(torch.nn.functional, config.hidden_act)
        self.fc1 = nn.Linear(config.hidden_size, small_intermediate_size)
        self.fc2 = nn.Linear(small_intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class MoE(nn.Module):
    """Mixture of Experts слой"""
    def __init__(self, num_experts, hidden_size, top_k, config: PhiConfig):
        super().__init__()
        self.gate = GShardGate(hidden_size, num_experts, 1, top_k)
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.top_k = top_k
        self.experts = nn.ModuleList([SmallPhiMLP(config, scale_factor=4) for _ in range(self.num_experts)])
        self.alpha = nn.Parameter(torch.randn(self.num_experts))
        self._step_count = 0
        self._log_frequency = 10  # Логируем каждые 10 шагов
        self._global_step = 0  # Глобальный шаг для MLflow
        self._layer_id = None  # ID слоя для уникального логирования

        self._soi_id = None  # Start of Image
        self._eoi_id = None  # End of Image
        self._sov_id = None  # Start of Video
        self._eov_id = None  # End of Video

    def forward(self, hidden_states, input_ids=None):
        batch_size, seq_len, hidden_size = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_size)
        gate_idx, gate_score = self.gate(hidden_states_flat)
        self._step_count += 1
        
        # Логирование распределения гейтов (только периодически)
        print(f'step_count: {self._step_count}, log_frequency: {self._log_frequency}')
        should_log = (self._step_count % self._log_frequency == 0)
        if hasattr(self, '_log_gates') and self._log_gates and should_log:
            print(f'logging gates')
            self._log_gate_distribution(gate_idx, gate_score, input_ids)
        
        output_flat = torch.zeros_like(hidden_states_flat)  # используем zeros_like вместо zeros
        expert_stats = {}
        
        for k in range(self.top_k):
            expert_indices = gate_idx[:, k]
            weights = gate_score[:, k]
            for expert_id in range(self.num_experts):
                mask = expert_indices == expert_id
                if mask.any():
                    expert_input = hidden_states_flat[mask]
                    expert_output = self.experts[expert_id](expert_input)
                    weight = weights[mask] * self.alpha[expert_id]
                    output_flat[mask] += expert_output * weight.unsqueeze(-1)
                    
                    if hasattr(self, '_log_activations') and self._log_activations and should_log:
                        input_abs = torch.abs(expert_input)
                        output_abs = torch.abs(expert_output)
                        
                        expert_stats[expert_id] = {
                            'input_min': expert_input.min().item(),
                            'input_max': expert_input.max().item(),
                            'input_mean': expert_input.mean().item(),
                            'input_abs_min': input_abs.min().item(),
                            'input_abs_max': input_abs.max().item(),
                            'input_abs_mean': input_abs.mean().item(),
                            'output_min': expert_output.min().item(),
                            'output_max': expert_output.max().item(),
                            'output_mean': expert_output.mean().item(),
                            'output_abs_min': output_abs.min().item(),
                            'output_abs_max': output_abs.max().item(),
                            'output_abs_mean': output_abs.mean().item(),
                            'weight_min': weight.min().item(),
                            'weight_max': weight.max().item(),
                            'weight_mean': weight.mean().item(),
                            'alpha': self.alpha[expert_id].item(),
                            'num_tokens': mask.sum().item()
                        }
                    del expert_input, expert_output, weight
        if hasattr(self, '_log_activations') and self._log_activations and should_log and expert_stats:
            self._log_activation_stats(expert_stats)
        
        result = output_flat.view(batch_size, seq_len, hidden_size)
        del output_flat, hidden_states_flat, gate_idx, gate_score
        return result

    def _log_gate_distribution(self, gate_idx, gate_score, input_ids=None):
        """Логирует распределение гейтов с разделением по модальностям"""
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

    def enable_logging(self, log_gates=True, log_activations=True, log_frequency=10):
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
        plt.title(f'Gate Weights Distribution - Layer {self._layer_id}')
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
            
            # Создаем и отправляем гистограмму
            histogram_bytes = self._create_gate_histogram(gate_score)
            
            
            
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                tmp_file.write(histogram_bytes)
                tmp_file_path = tmp_file.name
            
            # Логируем через MLflow client (правильный способ для локального хранилища)
            try:
                # Используем client.log_artifact вместо mlflow.log_artifact
                client.log_artifact(run_id, tmp_file_path, f"{layer_prefix}/")
                print(f"📊 Гистограмма отправлена: {layer_prefix}/gate_weights_histogram.png")
            except Exception as artifact_error:
                print(f"⚠️ Не удалось отправить через client.log_artifact: {artifact_error}")
                # Fallback: сохраняем локально
                artifacts_dir = f"./mlruns/artifacts/{layer_prefix}"
                os.makedirs(artifacts_dir, exist_ok=True)
                local_path = f"{artifacts_dir}/gate_weights_histogram.png"
                with open(local_path, 'wb') as f:
                    f.write(histogram_bytes)
                print(f"📊 Гистограмма сохранена локально: {local_path}")
            finally:
                os.unlink(tmp_file_path)
            
        except Exception as e:
            # Игнорируем ошибки MLflow, чтобы не прерывать обучение
            print(f"❌ Ошибка логирования гистограммы: {e}")
    
    def _log_to_mlflow_modality_gates(self, expert_counts, total_activations, gate_score, modality_name):
        """Логирует метрики гейтов для конкретной модальности в MLflow"""
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
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                tmp_file.write(histogram_bytes)
                tmp_file_path = tmp_file.name
            
            # Логируем через MLflow client (правильный способ для локального хранилища)
            try:
                # Используем client.log_artifact вместо mlflow.log_artifact
                client.log_artifact(run_id, tmp_file_path, f"{layer_prefix}/")
                print(f"📊 Гистограмма отправлена: {layer_prefix}/gate_weights_histogram.png")
            except Exception as artifact_error:
                print(f"⚠️ Не удалось отправить через client.log_artifact: {artifact_error}")
                # Fallback: сохраняем локально
                artifacts_dir = f"./mlruns/artifacts/{layer_prefix}"
                os.makedirs(artifacts_dir, exist_ok=True)
                local_path = f"{artifacts_dir}/gate_weights_histogram.png"
                with open(local_path, 'wb') as f:
                    f.write(histogram_bytes)
                print(f"📊 Гистограмма сохранена локально: {local_path}")
            finally:
                os.unlink(tmp_file_path)
            
        except Exception as e:
            # Игнорируем ошибки MLflow, чтобы не прерывать обучение
            print(f"❌ Ошибка логирования гистограммы модальности: {e}")
    
    def _create_modality_histogram(self, gate_score, modality_name):
        """Создает гистограмму распределения гейтов для конкретной модальности"""
        plt.figure(figsize=(10, 6))
        
        # Создаем гистограмму весов гейтов
        gate_weights = gate_score.detach().cpu().numpy().flatten()
        
        plt.hist(gate_weights, bins=50, alpha=0.7, color='blue', edgecolor='black')
        plt.title(f'Gate Weights Distribution - Layer {self._layer_id} - {modality_name.capitalize()} Tokens')
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
        """Логирует метрики активаций в MLflow"""
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
            
            # Логируем детальные метрики активаций для каждого эксперта
            layer_prefix = f"moe/layer_{self._layer_id}"
            
            if expert_stats:
                # Логируем статистики для каждого эксперта отдельно
                for expert_id, stats in expert_stats.items():
                    expert_prefix = f"{layer_prefix}/expert_{expert_id}"
                    
                    # Статистики входных активаций
                    client.log_metric(run_id, f"{expert_prefix}/input_min", stats['input_min'], step=self._global_step)
                    client.log_metric(run_id, f"{expert_prefix}/input_max", stats['input_max'], step=self._global_step)
                    client.log_metric(run_id, f"{expert_prefix}/input_mean", stats['input_mean'], step=self._global_step)
                    client.log_metric(run_id, f"{expert_prefix}/input_abs_min", stats['input_abs_min'], step=self._global_step)
                    client.log_metric(run_id, f"{expert_prefix}/input_abs_max", stats['input_abs_max'], step=self._global_step)
                    client.log_metric(run_id, f"{expert_prefix}/input_abs_mean", stats['input_abs_mean'], step=self._global_step)
                    
                    # Статистики выходных активаций
                    client.log_metric(run_id, f"{expert_prefix}/output_min", stats['output_min'], step=self._global_step)
                    client.log_metric(run_id, f"{expert_prefix}/output_max", stats['output_max'], step=self._global_step)
                    client.log_metric(run_id, f"{expert_prefix}/output_mean", stats['output_mean'], step=self._global_step)
                    client.log_metric(run_id, f"{expert_prefix}/output_abs_min", stats['output_abs_min'], step=self._global_step)
                    client.log_metric(run_id, f"{expert_prefix}/output_abs_max", stats['output_abs_max'], step=self._global_step)
                    client.log_metric(run_id, f"{expert_prefix}/output_abs_mean", stats['output_abs_mean'], step=self._global_step)
                    
                    # Дополнительные метрики
                    client.log_metric(run_id, f"{expert_prefix}/alpha", stats['alpha'], step=self._global_step)
                    client.log_metric(run_id, f"{expert_prefix}/num_tokens", stats['num_tokens'], step=self._global_step)
                
                # Агрегированные метрики по всем экспертам
                all_input_mins = [stats['input_min'] for stats in expert_stats.values()]
                all_input_maxs = [stats['input_max'] for stats in expert_stats.values()]
                all_output_mins = [stats['output_min'] for stats in expert_stats.values()]
                all_output_maxs = [stats['output_max'] for stats in expert_stats.values()]
                all_input_abs_mins = [stats['input_abs_min'] for stats in expert_stats.values()]
                all_input_abs_maxs = [stats['input_abs_max'] for stats in expert_stats.values()]
                all_output_abs_mins = [stats['output_abs_min'] for stats in expert_stats.values()]
                all_output_abs_maxs = [stats['output_abs_max'] for stats in expert_stats.values()]
                all_alphas = [stats['alpha'] for stats in expert_stats.values()]
                all_tokens = [stats['num_tokens'] for stats in expert_stats.values()]
                
                # Логируем агрегированные метрики
                client.log_metric(run_id, f"{layer_prefix}/input_range", max(all_input_maxs) - min(all_input_mins), step=self._global_step)
                client.log_metric(run_id, f"{layer_prefix}/output_range", max(all_output_maxs) - min(all_output_mins), step=self._global_step)
                client.log_metric(run_id, f"{layer_prefix}/input_abs_range", max(all_input_abs_maxs) - min(all_input_abs_mins), step=self._global_step)
                client.log_metric(run_id, f"{layer_prefix}/output_abs_range", max(all_output_abs_maxs) - min(all_output_abs_mins), step=self._global_step)
                client.log_metric(run_id, f"{layer_prefix}/alpha_mean", sum(all_alphas) / len(all_alphas), step=self._global_step)
                client.log_metric(run_id, f"{layer_prefix}/total_tokens", sum(all_tokens), step=self._global_step)
                client.log_metric(run_id, f"{layer_prefix}/active_experts", len(expert_stats), step=self._global_step)
            
        except Exception as e:
            # Игнорируем ошибки MLflow, чтобы не прерывать обучение
            pass


def patch_model_with_moe(model, count_layers_to_patch=3, num_experts=4, top_k=2, special_tokens=None):
    """
    Заменяет все MLP слои на MoE
    
    Args:
        model: Show-o модель
        num_experts: количество экспертов
        top_k: сколько активных экспертов
        special_tokens: словарь с ID специальных токенов
    
    Returns:
        model с MoE слоями
    """
    print(f"🔧 Патчим модель с MoE (experts={num_experts}, top_k={top_k})...")
    config_phi = model.showo.config
    
    for layer_idx, layer in list(enumerate(model.showo.model.layers))[::-1]:
        if count_layers_to_patch == 0:
            break
        if hasattr(layer, 'mlp'):
            print(f"  → Слой {layer_idx}")
            moe_layer = MoE(
                num_experts=num_experts,
                hidden_size=config_phi.hidden_size,
                top_k=top_k,
                config=config_phi
            )
            # Включаем логирование для MoE слоя (каждые 10 шагов)
            moe_layer.enable_logging(log_gates=True, log_activations=True, log_frequency=10)
            moe_layer.set_layer_id(layer_idx)  # Устанавливаем ID слоя
            
            # Устанавливаем специальные токены для определения модальности
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
    return model


def freeze_non_moe_params(model):
    print("❄️  Замораживаем все параметры кроме MoE...")

    for param in model.parameters():
        param.requires_grad = False
    
    # Размораживаем только MoE
    trainable_params = 0
    total_params = 0
    moe_param_names = []
    
    for name, param in model.named_parameters():
        total_params += param.numel()
        # Проверяем что это MoE параметры (есть experts, gate, alpha - специфичные для MoE)
        if 'showo.model.layers' in name and 'mlp' in name:
            if any(x in name for x in ['mlp.experts', 'mlp.gate', 'mlp.alpha']):
                param.requires_grad = True
                trainable_params += param.numel()
                moe_param_names.append(name)
    
    print(f"✓ Параметров: {total_params:,} total, {trainable_params:,} trainable ({100*trainable_params/total_params:.1f}%)")
    if len(moe_param_names) > 0:
        print(f"✓ Разморожено {len(moe_param_names)} MoE параметров:")
        for name in moe_param_names[:5]:  # показываем первые 5
            print(f"    - {name}")
        if len(moe_param_names) > 5:
            print(f"    ... и еще {len(moe_param_names) - 5}")
    return model


def patch_and_freeze_moe(model, count_layers_to_patch=3, num_experts=4, top_k=2, mlflow_client=None, mlflow_run_id=None, special_tokens=None):
    model = patch_model_with_moe(model, count_layers_to_patch, num_experts, top_k, special_tokens)
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


if __name__ == "__main__":
    from models import Showo
    model = Showo.from_pretrained("showlab/show-o-w-clip-vit")
    model = patch_and_freeze_moe(model, num_experts=4, top_k=2)
    print("\nГотово! Теперь можно обучать.")
    print("Пример:")
    print("  optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)")

