#!/usr/bin/env python3
"""
Скрипт для инициализации модели Show-o с нуля для обучения.
Создает модель без загрузки предобученных весов.
"""

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from models.modeling.modeling_showo import Showo


def get_config():
    """Загружает конфигурацию из командной строки и YAML файла"""
    cli_conf = OmegaConf.from_cli()
    yaml_conf = OmegaConf.load(cli_conf.config)
    conf = OmegaConf.merge(yaml_conf, cli_conf)
    return conf


def create_model_from_config(config):
    """Создает модель Show-o с нуля на основе конфигурации"""
    print("Создаем модель Show-o с нуля...")
    
    # Создаем модель с нуля, используя load_from_showo=True для инициализации с нуля
    model = Showo(
        w_clip_vit=config.model.showo.w_clip_vit,
        vocab_size=config.model.showo.vocab_size,
        llm_vocab_size=config.model.showo.llm_vocab_size,
        llm_model_path=config.model.showo.llm_model_path,
        codebook_size=config.model.showo.get('codebook_size', 8192),
        num_vq_tokens=config.model.showo.get('num_vq_tokens', 256),
        load_from_showo=True  # Это заставит модель инициализироваться с нуля
    )
    
    print(f"Модель создана с {sum(p.numel() for p in model.parameters())} параметрами")
    return model


def initialize_model_weights(model):
    """Инициализирует веса модели"""
    print("Инициализируем веса модели...")
    
    def init_weights(module):
        """Рекурсивная инициализация весов"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    # Применяем инициализацию ко всем модулям
    model.apply(init_weights)
    print("Веса инициализированы")


def save_model(model, config, save_path):
    """Сохраняет модель"""
    print(f"Сохраняем модель в {save_path}")
    
    # Создаем директорию если не существует
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Сохраняем модель
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'model_type': 'showo'
    }, save_path)
    
    print(f"Модель сохранена в {save_path}")


def main():
    """Основная функция"""
    print("=== Инициализация модели Show-o для обучения ===")
    
    # Загружаем конфигурацию
    config = get_config()
    print(f"Конфигурация загружена из {config.config}")
    
    # Создаем модель
    model = create_model_from_config(config)
    
    # Инициализируем веса
    initialize_model_weights(model)
    
    # Перемещаем модель на устройство
    device = config.device
    print(f"Перемещаем модель на {device}")
    model = model.to(device)
    
    # Сохраняем модель
    save_path = config.get('save_path', 'models/showo_initialized.pth')
    save_model(model, config, save_path)
    
    print("=== Инициализация завершена ===")
    print(f"Модель готова для обучения")
    print(f"Сохранена в: {save_path}")
    
    # Показываем информацию о модели
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nИнформация о модели:")
    print(f"  Всего параметров: {total_params:,}")
    print(f"  Обучаемых параметров: {trainable_params:,}")
    print(f"  Размер модели: {total_params * 4 / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()
