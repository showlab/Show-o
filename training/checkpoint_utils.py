import os
import json
import shutil
from pathlib import Path

import torch
from accelerate.logging import get_logger

logger = get_logger(__name__, log_level="INFO")


def save_checkpoint(model, config, accelerator, global_step):
    output_dir = config.experiment.output_dir
    checkpoints_total_limit = config.experiment.get("checkpoints_total_limit", None)
    save_moe_only = config.get("moe", {}).get("save_only_moe_weights", False)

    # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
    if accelerator.is_main_process and checkpoints_total_limit is not None:
        checkpoints = os.listdir(output_dir)
        checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

        # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
        if len(checkpoints) >= checkpoints_total_limit:
            num_to_remove = len(checkpoints) - checkpoints_total_limit + 1
            removing_checkpoints = checkpoints[0:num_to_remove]

            logger.info(
                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
            )
            logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

            for removing_checkpoint in removing_checkpoints:
                removing_checkpoint = os.path.join(output_dir, removing_checkpoint)
                shutil.rmtree(removing_checkpoint)

    save_path = Path(output_dir) / f"checkpoint-{global_step}"

    # retrieve the model on all processes for deepspeed stage 3 to work then save on one process (we are not using stage 3 yet)
    # XXX: could also make this conditional on deepspeed
    state_dict = accelerator.get_state_dict(model)
    if accelerator.is_main_process:
        save_path.mkdir(parents=True, exist_ok=True)
        
        if save_moe_only and config.get("moe", {}).get("enabled", False):
            moe_state_dict = {
                k: v for k, v in state_dict.items() 
                if 'showo.model.layers' in k and 'mlp' in k
            }
            torch.save(moe_state_dict, save_path / "moe_weights.pt")
            logger.info(f"💾 Saved {len(moe_state_dict)} MoE parameters to {save_path} ({sum(p.numel() for p in moe_state_dict.values()):,} params)")
        else:
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                save_path / "unwrapped_model",
                save_function=accelerator.save,
                state_dict=state_dict,
                safe_serialization=False
            )
            logger.info(f"Saved full model to {save_path}")
        
        json.dump({"global_step": global_step}, (save_path / "metadata.json").open("w+"))


def load_checkpoint(model, config, accelerator, checkpoint_path):
    """
    Загружает чекпоинт модели
    
    Args:
        model: Модель для загрузки весов
        config: Конфигурация
        accelerator: Accelerator объект
        checkpoint_path: Путь к чекпоинту (например, "output/showo-vanilla-mmu/checkpoint-24000")
    
    Returns:
        global_step: Номер шага из чекпоинта
    """
    save_moe_only = config.get("moe", {}).get("save_only_moe_weights", False)
    moe_enabled = config.get("moe", {}).get("enabled", False)
    
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Загружаем метаданные
    metadata_path = checkpoint_path / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        global_step = metadata.get("global_step", 0)
    else:
        global_step = 0
        logger.warning(f"No metadata.json found in {checkpoint_path}, assuming step 0")
    
    if save_moe_only and moe_enabled:
        # Загружаем только MoE веса
        moe_weights_path = checkpoint_path / "moe_weights.pt"
        if moe_weights_path.exists():
            moe_state_dict = torch.load(moe_weights_path, map_location="cpu")
            model.load_state_dict(moe_state_dict, strict=False)
            logger.info(f"💾 Loaded {len(moe_state_dict)} MoE parameters from {moe_weights_path}")
        else:
            raise FileNotFoundError(f"MoE weights not found: {moe_weights_path}")
    else:
        # Загружаем полную модель
        model_path = checkpoint_path / "unwrapped_model"
        if model_path.exists():
            # Проверяем, есть ли pytorch_model.bin
            pytorch_model_path = model_path / "pytorch_model.bin"
            if pytorch_model_path.exists():
                state_dict = torch.load(pytorch_model_path, map_location="cpu")
                model.load_state_dict(state_dict, strict=False)
                logger.info(f"💾 Loaded full model from {pytorch_model_path}")
            else:
                # Пытаемся загрузить через from_pretrained
                try:
                    model = model.__class__.from_pretrained(model_path)
                    logger.info(f"💾 Loaded full model from {model_path}")
                except Exception as e:
                    logger.error(f"Failed to load model from {model_path}: {e}")
                    raise
        else:
            raise FileNotFoundError(f"Model not found: {model_path}")
    
    logger.info(f"✅ Checkpoint loaded successfully from {checkpoint_path} (step {global_step})")
    return global_step

