#!/usr/bin/env python3
"""
Модели с поддержкой конфигурации через get_config()
"""
import torch
from typing import Optional, Union, Dict, Any
from pathlib import Path
from hydra.utils import instantiate
from training.utils import get_config
from .modeling_utils import ModelMixin


class ConfigModelMixin(ModelMixin):
    """Базовый класс для моделей с поддержкой конфигурации через get_config()"""
    
    @classmethod
    def from_config_path(
        cls,
        config_path: str,
        **kwargs
    ):
        """
        Создает модель из конфигурационного файла
        
        Args:
            config_path: Путь к конфигурационному файлу
            **kwargs: Дополнительные параметры для модели
            
        Returns:
            Созданная модель
        """
        import sys
        original_argv = sys.argv.copy()
        sys.argv = ['config_model.py', f'config={config_path}']
        
        try:
            cfg = get_config()
            return cls._create_from_config(cfg, **kwargs)
        finally:
            sys.argv = original_argv
    
    @classmethod
    def _create_from_config(cls, cfg, **kwargs):
        """Создает модель из конфигурации - должен быть переопределен в наследниках"""
        raise NotImplementedError("Subclasses must implement _create_from_config")


class PhiDecoderLayerModel(ConfigModelMixin):
    """PhiDecoderLayer с поддержкой конфигурации"""
    
    @classmethod
    def _create_from_config(cls, cfg, layer_idx: int = 0, **kwargs):
        """Создает PhiDecoderLayer из конфигурации"""
        from transformers.models.phi.configuration_phi import PhiConfig
        
        # Создаем PhiConfig
        showo_config = cfg.model.showo
        phi_config = PhiConfig(
            hidden_size=showo_config.hidden_size,
            intermediate_size=showo_config.intermediate_size,
            num_attention_heads=showo_config.num_attention_heads,
            num_hidden_layers=showo_config.num_hidden_layers,
            vocab_size=showo_config.vocab_size,
            resid_pdrop=showo_config.resid_pdrop,
            layer_norm_eps=showo_config.layer_norm_eps,
            _attn_implementation=showo_config._attn_implementation,
        )
        
        # Добавляем MoE конфигурацию если есть
        if hasattr(showo_config, 'moe'):
            phi_config.moe = type('obj', (object,), {
                'num_experts': showo_config.moe.num_experts,
                'top_k': showo_config.moe.top_k,
                'gate': {
                    '_target_': 'models.moe_gates.GShardGate',
                    'd_model': showo_config.moe.gate.d_model,
                    'num_expert': showo_config.moe.gate.num_expert,
                    'world_size': showo_config.moe.gate.world_size,
                    'top_k': showo_config.moe.gate.top_k
                }
            })()
        
        # Создаем decoder_layer конфигурацию с вложенным FFN
        decoder_layer_config = {
            '_target_': 'models.decoder.PhiDecoderLayer',
            'config': phi_config,
            'layer_idx': layer_idx,
            'ffn': {
                '_target_': cfg.model.showo.decoder_layer.ffn._target_,
                'config': phi_config
            }
        }
        
        # Инстанцируем decoder layer
        layer = instantiate(decoder_layer_config, **kwargs)
        return layer


class FFNModel(ConfigModelMixin):
    """FFN (MoE или FFNDefault) с поддержкой конфигурации"""
    
    @classmethod
    def _create_from_config(cls, cfg, **kwargs):
        """Создает FFN из конфигурации"""
        from transformers.models.phi.configuration_phi import PhiConfig
        
        # Создаем PhiConfig
        showo_config = cfg.model.showo
        phi_config = PhiConfig(
            hidden_size=showo_config.hidden_size,
            intermediate_size=showo_config.intermediate_size,
            num_attention_heads=showo_config.num_attention_heads,
            num_hidden_layers=showo_config.num_hidden_layers,
            vocab_size=showo_config.vocab_size,
            resid_pdrop=showo_config.resid_pdrop,
            layer_norm_eps=showo_config.layer_norm_eps,
            _attn_implementation=showo_config._attn_implementation,
        )
        
        # Добавляем MoE конфигурацию если есть
        if hasattr(showo_config, 'moe'):
            phi_config.moe = type('obj', (object,), {
                'num_experts': showo_config.moe.num_experts,
                'top_k': showo_config.moe.top_k,
                'gate': {
                    '_target_': 'models.moe_gates.GShardGate',
                    'd_model': showo_config.moe.gate.d_model,
                    'num_expert': showo_config.moe.gate.num_expert,
                    'world_size': showo_config.moe.gate.world_size,
                    'top_k': showo_config.moe.gate.top_k
                }
            })()
        
        # Создаем FFN конфигурацию
        ffn_config = {
            '_target_': cfg.model.showo.decoder_layer.ffn._target_,
            'config': phi_config
        }
        
        # Инстанцируем FFN
        ffn = instantiate(ffn_config, **kwargs)
        return ffn


# Удобные функции для создания моделей
def create_decoder_layer(config_path: str, layer_idx: int = 0, **kwargs):
    """Создает PhiDecoderLayer из конфигурации"""
    return PhiDecoderLayerModel.from_config_path(config_path, layer_idx=layer_idx, **kwargs)


def create_ffn(config_path: str, **kwargs):
    """Создает FFN из конфигурации"""
    return FFNModel.from_config_path(config_path, **kwargs)
