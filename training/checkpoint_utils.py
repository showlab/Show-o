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
    save_moe_only = config.get("moe", {}).get("save_only_moe_weights", True)

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

