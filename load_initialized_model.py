#!/usr/bin/env python3
"""
Скрипт для загрузки инициализированной модели Show-o.
Показывает как загрузить модель, созданную с помощью init_model_for_training.py
"""

import torch
from omegaconf import OmegaConf
from models.modeling.modeling_showo import Showo


def load_initialized_model(model_path, config_path=None):
    """Загружает инициализированную модель"""
    print(f"Загружаем модель из {model_path}")
    
    # Загружаем сохраненную модель
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Если есть конфигурация в чекпоинте, используем её
    if 'config' in checkpoint:
        config = checkpoint['config']
        print("Используем конфигурацию из чекпоинта")
    elif config_path:
        # Иначе загружаем из файла
        config = OmegaConf.load(config_path)
        print(f"Используем конфигурацию из {config_path}")
    else:
        raise ValueError("Необходимо указать config_path или сохранить конфигурацию в чекпоинте")
    
    # Создаем модель с нуля
    model = Showo(
        w_clip_vit=config.model.showo.w_clip_vit,
        vocab_size=config.model.showo.vocab_size,
        llm_vocab_size=config.model.showo.llm_vocab_size,
        llm_model_path=config.model.showo.llm_model_path,
        codebook_size=config.model.showo.get('codebook_size', 8192),
        num_vq_tokens=config.model.showo.get('num_vq_tokens', 256),
        load_from_showo=True
    )
    
    # Загружаем веса
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print("Модель успешно загружена")
    return model, config


def test_model(model, device='cuda'):
    """Тестирует загруженную модель"""
    print(f"Тестируем модель на {device}")
    
    # Перемещаем модель на устройство
    model = model.to(device)
    model.eval()
    
    # Создаем тестовые данные
    batch_size = 2
    seq_len = 10
    
    # Тестовые токены
    test_tokens = torch.randint(0, 1000, (batch_size, seq_len)).to(device)
    
    print(f"Тестовые данные: {test_tokens.shape}")
    
    # Тестовый forward pass
    with torch.no_grad():
        try:
            outputs = model(test_tokens)
            print(f"Forward pass успешен! Выход: {outputs.shape}")
            return True
        except Exception as e:
            print(f"Ошибка при forward pass: {e}")
            return False


def main():
    """Основная функция"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Загрузка инициализированной модели Show-o')
    parser.add_argument('--model_path', type=str, default='models/showo_initialized.pth',
                       help='Путь к сохраненной модели')
    parser.add_argument('--config_path', type=str, default='configs/showo_demo_w_clip_vit_512x512.yaml',
                       help='Путь к конфигурации')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Устройство для тестирования')
    parser.add_argument('--test', action='store_true',
                       help='Запустить тест модели')
    
    args = parser.parse_args()
    
    print("=== Загрузка инициализированной модели Show-o ===")
    
    try:
        # Загружаем модель
        model, config = load_initialized_model(args.model_path, args.config_path)
        
        # Показываем информацию о модели
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"\nИнформация о модели:")
        print(f"  Всего параметров: {total_params:,}")
        print(f"  Обучаемых параметров: {trainable_params:,}")
        print(f"  Размер модели: {total_params * 4 / 1024 / 1024:.2f} MB")
        
        # Тестируем модель если нужно
        if args.test:
            success = test_model(model, args.device)
            if success:
                print("✅ Модель работает корректно!")
            else:
                print("❌ Модель не работает корректно!")
        
        print("=== Загрузка завершена ===")
        
    except Exception as e:
        print(f"❌ Ошибка при загрузке модели: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

