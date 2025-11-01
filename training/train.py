import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"
import json
import logging
import math
import shutil
import time
from pathlib import Path
from typing import Union
from utils import get_optimizer

import numpy as np
from PIL import Image
from omegaconf import OmegaConf
import mlflow
from mlflow.tracking import MlflowClient
import torch
from tqdm import tqdm
from lightning.pytorch.utilities import CombinedLoader

from transformers import AutoTokenizer
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedType, set_seed

import sys

sys.path.insert(0, "/home/jovyan/vasiliev/notebooks/Show-o")

import mlflow
from mlflow.tracking import MlflowClient

from models import Showo, MAGVITv2, get_mask_chedule
from training.prompting_utils import (
    UniversalPrompting,
    create_attention_mask_predict_next,
    create_attention_mask_for_mmu,
)
from models.lr_schedulers import get_scheduler
import torch.nn as nn
from torch.optim.lr_scheduler import SequentialLR, LinearLR, ExponentialLR
from models.logger import set_verbosity_info, set_verbosity_error
from moe_utils import patch_and_freeze_moe  # MoE support
from training.eval_utils import (
    visualize_predictions,
    generate_images,
    log_grad_norm,
    collect_and_log_moe_activations,
    log_training_metrics,
    evaluate_mmu,
)
from training.dataset_utils import create_dataloaders
from training.checkpoint_utils import save_checkpoint

from training.utils import (
    get_config,
    flatten_omega_conf,
    mask_or_random_replace_tokens,
    AverageMeter,
)


logger = get_logger(__name__, log_level="INFO")


@torch.no_grad()
def prepare_inputs_and_labels(
    pixel_values_or_image_ids: Union[torch.FloatTensor, torch.LongTensor],
    texts: Union[str, str],
    vq_model,
    uni_prompting,
    mask_id,
    config,
    mask_schedule,
    min_masking_rate: float = 0.0,
    is_train: bool = True,
):
    image_tokens = vq_model.get_code(pixel_values_or_image_ids)
    image_tokens = image_tokens + len(uni_prompting.text_tokenizer)

    # create MLM mask and labels
    input_ids, labels, loss_weight, mask_prob = mask_or_random_replace_tokens(
        image_tokens,
        mask_id,
        config,
        mask_schedule=mask_schedule,
        is_train=is_train,
    )
    # Correct T2I sequence: T2I → SOT → [Text tokens] → EOT → SOI → [Image tokens] → EOI
    input_ids, masks, labels = uni_prompting((texts, input_ids, labels), "t2i")

    return input_ids, labels, mask_prob, image_tokens


def train_step(
    batch,
    epoch,
    global_step,
    model,
    optimizer,
    lr_scheduler,
    balance_scheduler,
    temp_scheduler,
    accelerator,
    config,
    uni_prompting,
    vq_model,
    mask_dtype,
    mask_id,
    mask_schedule,
    batch_time_m,
    data_time_m,
    total_batch_size_per_gpu,
    mlflow_client,
    mlflow_run_id,
    pbar,
):
    batch_size_t2i = batch["t2i_flow"]["images"].shape[0]
    batch_size_lm = len(batch["lm_flow"]["input_ids"])
    batch_size_mmu = batch["mmu_flow"]["images"].shape[0]

    # Build T2I sequences
    pixel_values, texts = batch["t2i_flow"]["images"], batch["t2i_flow"]["input_ids"]
    pixel_values = pixel_values.to(accelerator.device, non_blocking=True)
    input_ids_t2i, labels_t2i, mask_prob, image_tokens_ori = prepare_inputs_and_labels(
        pixel_values,
        texts,
        vq_model,
        uni_prompting,
        mask_id,
        config,
        mask_schedule,
        config.training.min_masking_rate,
    )
    attention_mask = create_attention_mask_predict_next(
        input_ids_t2i,
        pad_id=int(uni_prompting.sptids_dict["<|pad|>"]),
        soi_id=int(uni_prompting.sptids_dict["<|soi|>"]),
        eoi_id=int(uni_prompting.sptids_dict["<|eoi|>"]),
        rm_pad_in_image=True,
        return_inverse_mask=True,
    )
    attention_mask = attention_mask.to(mask_dtype)

    # Build LM sequences
    texts_lm = batch["lm_flow"]["input_ids"]
    input_ids_lm, _, labels_lm = uni_prompting(
        (texts_lm, input_ids_t2i.shape[-1]), "lm"
    )
    attention_mask_lm = create_attention_mask_predict_next(
        input_ids_lm.to(input_ids_t2i.device),
        pad_id=int(uni_prompting.sptids_dict["<|pad|>"]),
        soi_id=int(uni_prompting.sptids_dict["<|soi|>"]),
        eoi_id=int(uni_prompting.sptids_dict["<|eoi|>"]),
    )
    attention_mask_lm = attention_mask_lm.to(mask_dtype)
    attention_mask = torch.cat([attention_mask, attention_mask_lm], dim=0)
    input_ids = torch.cat((input_ids_t2i, input_ids_lm.to(input_ids_t2i.device)), dim=0)
    labels = torch.cat((labels_t2i, labels_lm.to(input_ids_t2i.device)), dim=0)

    # Build MMU sequences
    if "llava" in config.dataset.und_type:
        pixel_values_mmu, input_ids_mmu, labels_mmu = (
            batch["mmu_flow"]["images"],
            batch["mmu_flow"]["input_ids"],
            batch["mmu_flow"]["labels"],
        )
        pixel_values_mmu = pixel_values_mmu.to(accelerator.device, non_blocking=True)
        input_ids_mmu = input_ids_mmu.to(accelerator.device, non_blocking=True)
        image_tokens_mmu = vq_model.get_code(pixel_values_mmu)
        image_tokens_mmu = image_tokens_mmu + len(uni_prompting.text_tokenizer)

        input_ids_mmu = torch.cat([
            (torch.ones(input_ids_mmu.shape[0], 1) * uni_prompting.sptids_dict['<|mmu|>']).to(accelerator.device),
            (torch.ones(input_ids_mmu.shape[0], 1) * uni_prompting.sptids_dict['<|soi|>']).to(accelerator.device),
            image_tokens_mmu,
            (torch.ones(input_ids_mmu.shape[0], 1) * uni_prompting.sptids_dict['<|eoi|>']).to(accelerator.device),
            input_ids_mmu,
        ], dim=1).long()

        labels_mmu = torch.cat([
            (torch.ones(input_ids_mmu.shape[0], 1) * uni_prompting.ignore_id).to(accelerator.device),
            (torch.ones(input_ids_mmu.shape[0], 1) * uni_prompting.ignore_id).to(accelerator.device),
            torch.ones_like(image_tokens_mmu) * uni_prompting.ignore_id,
            (torch.ones(input_ids_mmu.shape[0], 1) * uni_prompting.ignore_id).to(accelerator.device),
            labels_mmu.to(accelerator.device)
        ], dim=1).long()

    else:
        pixel_values_mmu, texts_mmu = (
            batch["mmu_flow"]["images"],
            batch["mmu_flow"]["input_ids"],
        )
        pixel_values_mmu = pixel_values_mmu.to(accelerator.device, non_blocking=True)
        image_tokens_mmu = vq_model.get_code(pixel_values_mmu)
        image_tokens_mmu = image_tokens_mmu + len(uni_prompting.text_tokenizer)
        input_ids_mmu, _, labels_mmu = uni_prompting(
            (image_tokens_mmu, texts_mmu), "mmu"
        )
        input_ids_mmu = input_ids_mmu.to(accelerator.device, non_blocking=True)

    attention_mask_mmu = create_attention_mask_for_mmu(
        input_ids_mmu.to(input_ids.device),
        eoi_id=int(uni_prompting.sptids_dict["<|eoi|>"]),
    )
    attention_mask_mmu = attention_mask_mmu.to(mask_dtype)
    
    # Debug: log shapes on first step
    if global_step == 0 and epoch == 0:
        logger.info(f"input_ids shape: {input_ids.shape}")
        logger.info(f"input_ids_mmu shape: {input_ids_mmu.shape}")
        logger.info(f"attention_mask shape: {attention_mask.shape}")
        logger.info(f"attention_mask_mmu shape: {attention_mask_mmu.shape}")
        logger.info(f"labels shape: {labels.shape}")
        logger.info(f"labels_mmu shape: {labels_mmu.shape}")
    
    # Pad sequences to the same length if needed
    max_len = max(input_ids.shape[1], input_ids_mmu.shape[1])
    
    if input_ids.shape[1] < max_len:
        pad_len = max_len - input_ids.shape[1]
        input_ids = torch.cat([input_ids, torch.full((input_ids.shape[0], pad_len), uni_prompting.pad_id, dtype=input_ids.dtype, device=input_ids.device)], dim=1)
        labels = torch.cat([labels, torch.full((labels.shape[0], pad_len), uni_prompting.ignore_id, dtype=labels.dtype, device=labels.device)], dim=1)
        attention_mask = torch.cat([attention_mask, torch.full((attention_mask.shape[0], 1, attention_mask.shape[2], pad_len), torch.finfo(mask_dtype).min, dtype=mask_dtype, device=attention_mask.device)], dim=3)
        attention_mask = torch.cat([attention_mask, torch.full((attention_mask.shape[0], 1, pad_len, max_len), torch.finfo(mask_dtype).min, dtype=mask_dtype, device=attention_mask.device)], dim=2)
    
    if input_ids_mmu.shape[1] < max_len:
        pad_len = max_len - input_ids_mmu.shape[1]
        input_ids_mmu = torch.cat([input_ids_mmu, torch.full((input_ids_mmu.shape[0], pad_len), uni_prompting.pad_id, dtype=input_ids_mmu.dtype, device=input_ids_mmu.device)], dim=1)
        labels_mmu = torch.cat([labels_mmu, torch.full((labels_mmu.shape[0], pad_len), uni_prompting.ignore_id, dtype=labels_mmu.dtype, device=labels_mmu.device)], dim=1)
        attention_mask_mmu = torch.cat([attention_mask_mmu, torch.full((attention_mask_mmu.shape[0], 1, attention_mask_mmu.shape[2], pad_len), torch.finfo(mask_dtype).min, dtype=mask_dtype, device=attention_mask_mmu.device)], dim=3)
        attention_mask_mmu = torch.cat([attention_mask_mmu, torch.full((attention_mask_mmu.shape[0], 1, pad_len, max_len), torch.finfo(mask_dtype).min, dtype=mask_dtype, device=attention_mask_mmu.device)], dim=2)
    
    attention_mask = torch.cat([attention_mask, attention_mask_mmu], dim=0)
    input_ids = torch.cat((input_ids, input_ids_mmu.to(input_ids.device)), dim=0)
    labels = torch.cat((labels, labels_mmu.to(input_ids.device)), dim=0)

    if global_step == 0 and epoch == 0:
        logger.info(f"📊 First training step diagnostics:")
        logger.info(f"   input_ids shape: {input_ids.shape}")
        logger.info(f"   input_ids range: [{input_ids.min().item()}, {input_ids.max().item()}]")
        logger.info(f"   labels range (non-ignore): [{labels[labels != uni_prompting.ignore_id].min().item()}, {labels[labels != uni_prompting.ignore_id].max().item()}]")
        logger.info(f"   mask_id used: {mask_id}")
        logger.info(f"   text_tokenizer size: {len(uni_prompting.text_tokenizer)}")
        logger.info(f"   Expected image token range: [{len(uni_prompting.text_tokenizer)}, {len(uni_prompting.text_tokenizer) + 8192 - 1}]")
        # logger.info("Input ids: {}".format(input_ids))
        # logger.info("Labels: {}".format(labels))

    current_temperature = temp_scheduler.get_last_lr()[0]
    logits, loss_t2i, loss_lm, loss_mmu = model(
        input_ids=input_ids,
        input_embeddings=None,
        attention_mask=attention_mask,
        labels=labels,
        label_smoothing=config.training.label_smoothing,
        batch_size_t2i=batch_size_t2i,
        batch_size_lm=batch_size_lm,
        batch_size_mmu=batch_size_mmu,
        max_seq_length=config.dataset.preprocessing.max_seq_length,
        moe_temperature=current_temperature,
    )

    # Gather the losses across all processes for logging (if we use distributed training).
    avg_loss_t2i = accelerator.gather(
        loss_t2i.repeat(config.training.batch_size_t2i)
    ).mean()
    avg_loss_lm = accelerator.gather(
        loss_lm.repeat(config.training.batch_size_lm)
    ).mean()
    avg_loss_mmu = accelerator.gather(
        loss_mmu.repeat(config.training.batch_size_mmu)
    ).mean()

    # MoE balance loss
    if config.get("moe", {}).get("enabled", False):
        balance_loss, num_moe_layers = collect_moe_balance_losses(model)
        if num_moe_layers == 0:
            logger.warning(f"⚠️ MoE enabled but num_moe_layers = {num_moe_layers}")
        balance_coeff = balance_scheduler.get_last_lr()[0]
        print(f'balance_coeff: {balance_coeff}')
    else:
        print(f'Setting balance_loss and balance_coeff to 0.0')
        balance_loss = torch.tensor(0.0, device=accelerator.device)
        balance_coeff = 0.0

    loss = (
        config.training.t2i_coeff * loss_t2i
        + config.training.lm_coeff * loss_lm
        + config.training.mmu_coeff * loss_mmu
        + balance_coeff * balance_loss
    )

    avg_masking_rate = accelerator.gather(
        mask_prob.repeat(config.training.batch_size_t2i)
    ).mean()

    # Backward pass
    accelerator.backward(loss)

    # Очищаем кэш GPU для освобождения памяти
    if (
        torch.cuda.is_available()
        and (global_step + 1) % config.training.gradient_accumulation_steps == 0
    ):
        torch.cuda.empty_cache()

    if config.training.max_grad_norm is not None and accelerator.sync_gradients:
        accelerator.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)

    optimizer.step()
    lr_scheduler.step()
    balance_scheduler.step()
    temp_scheduler.step()

    # log gradient norm before zeroing it
    if (
        accelerator.sync_gradients
        and (global_step + 1) % config.experiment.log_grad_norm_every == 0
        and accelerator.is_main_process
    ):
        log_grad_norm(model, accelerator, global_step + 1, mlflow_client, mlflow_run_id)

    optimizer.zero_grad(set_to_none=True)

    # Log metrics
    if (
        accelerator.sync_gradients
        and (global_step + 1) % config.experiment.log_every == 0
    ):
        samples_per_second_per_gpu = (
            config.training.gradient_accumulation_steps
            * total_batch_size_per_gpu
            / batch_time_m.val
        )
        log_training_metrics(
            avg_loss_t2i=avg_loss_t2i,
            avg_loss_lm=avg_loss_lm,
            avg_loss_mmu=avg_loss_mmu,
            balance_loss=balance_loss,
            balance_coeff=balance_coeff,
            avg_masking_rate=avg_masking_rate,
            lr_scheduler=lr_scheduler,
            batch_time_m=batch_time_m,
            data_time_m=data_time_m,
            samples_per_second_per_gpu=samples_per_second_per_gpu,
            global_step=global_step + 1,
            mlflow_client=mlflow_client,
            mlflow_run_id=mlflow_run_id,
            logger=logger,
        )

        if mlflow_client is not None and mlflow_run_id is not None and config.get("moe", {}).get("enabled", False):
            temperature = float(temp_scheduler.get_last_lr()[0])
            mlflow_client.log_metric(mlflow_run_id, "moe/temperature", temperature, step=global_step + 1)
            logger.info(f"[moe] temperature: {temperature:.4f}")
            mlflow_client.log_metric(mlflow_run_id, "moe/balance_coeff", float(balance_coeff), step=global_step + 1)
            logger.info(f"[moe] balance_coeff: {float(balance_coeff):.6f}")

        # Set global step for MoE layers
        if config.get("moe", {}).get("enabled", False):
            unwrapped_model = accelerator.unwrap_model(model)
            for layer in unwrapped_model.showo.model.layers:
                if hasattr(layer, "mlp") and hasattr(layer.mlp, "set_global_step"):
                    layer.mlp.set_global_step(global_step + 1)

        # Reset time meters
        batch_time_m.reset()
        data_time_m.reset()

    if accelerator.is_main_process and pbar is not None:
        pbar.update(1)
        all_lrs = lr_scheduler.get_last_lr()
        if len(all_lrs) >= 4:
            lr_str = f"moe={all_lrs[0]:.2e}/base={all_lrs[2]:.2e}"
        else:
            lr_str = f"{all_lrs[0]:.2e}"
        pbar.set_postfix(
            {
                "loss_mmu": f"{avg_loss_mmu.item():.4f}",
                "loss_t2i": f"{avg_loss_t2i.item():.4f}",
                "lr": lr_str,
            }
        )

    return {
        "avg_loss_t2i": avg_loss_t2i,
        "avg_loss_lm": avg_loss_lm,
        "avg_loss_mmu": avg_loss_mmu,
        "balance_loss": balance_loss,
        "avg_masking_rate": avg_masking_rate,
        "logits": logits,
        "input_ids": input_ids,
        "input_ids_t2i": input_ids_t2i,
        "labels": labels,
        "attention_mask": attention_mask,
        "image_tokens_ori": image_tokens_ori,
        "texts": texts,
        "batch_size_t2i": batch_size_t2i,
        "batch_size_lm": batch_size_lm,
        "batch_size_mmu": batch_size_mmu,
    }


def collect_moe_balance_losses(model):
    total_balance_loss = 0.0
    num_moe_layers = 0
    if hasattr(model, "module"):
        unwrapped_model = model.module
    else:
        unwrapped_model = model

    for layer_idx, layer in enumerate(unwrapped_model.showo.model.layers):
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "gate"):
            if hasattr(layer.mlp.gate, "get_loss") and layer.mlp.gate.has_loss:
                gate_loss = layer.mlp.gate.get_loss(
                    clear=True
                )  # clear=True чтобы не накапливать
                if gate_loss is not None:
                    total_balance_loss += gate_loss
                    num_moe_layers += 1
                    logger.debug(
                        f"MoE layer {layer_idx}: balance_loss = {gate_loss.item():.6f}"
                    )
    return total_balance_loss, num_moe_layers


def get_vq_model_class(model_type):
    if model_type == "magvitv2":
        return MAGVITv2
    elif model_type == "vq16":
        return VQ_16
    else:
        raise ValueError(f"model_type {model_type} not supported.")


def main():
    #########################
    # SETUP Accelerator     #
    #########################
    config = get_config()

    # Enable TF32 on Ampere GPUs
    if config.training.enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    config.experiment.logging_dir = str(Path(config.experiment.output_dir) / "logs")
    from accelerate.utils import DistributedDataParallelKwargs

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        mixed_precision=config.training.mixed_precision,
        log_with="mlflow"
        if config.get("mlflow", {}).get("enabled", False)
        else None,  # MLflow вместо wandb
        project_dir=config.experiment.logging_dir,
        split_batches=True,
        kwargs_handlers=[ddp_kwargs],
    )

    total_batch_size_per_gpu = (
        config.training.batch_size_t2i
        + config.training.batch_size_lm
        + config.training.batch_size_mmu
    )
    total_batch_size = (
        (
            config.training.batch_size_t2i
            + config.training.batch_size_lm
            + config.training.batch_size_mmu
        )
        * accelerator.num_processes
        * config.training.gradient_accumulation_steps
    )

    if accelerator.distributed_type == DistributedType.DEEPSPEED:
        accelerator.state.deepspeed_plugin.deepspeed_config[
            "train_micro_batch_size_per_gpu"
        ] = total_batch_size_per_gpu

    #####################################
    # SETUP LOGGING, SEED and CONFIG    #
    #####################################
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        set_verbosity_info()
    else:
        set_verbosity_error()

    mlflow_client = None
    mlflow_run_id = None
    if accelerator.is_main_process and config.get("mlflow", {}).get("enabled", False):
        mlflow_tracking_uri = config.mlflow.get("tracking_uri", "file:./mlruns")
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow_client = MlflowClient(tracking_uri=mlflow_tracking_uri)
        experiment_name = config.mlflow.get(
            "experiment_name", config.experiment.project
        )
        try:
            experiment = mlflow_client.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = mlflow_client.create_experiment(experiment_name)
            else:
                experiment_id = experiment.experiment_id
        except:
            experiment_id = mlflow_client.create_experiment(experiment_name)

        run = mlflow_client.create_run(
            experiment_id=experiment_id,
            run_name=config.experiment.name,
            tags=config.mlflow.get("tags", {}),
        )
        mlflow_run_id = run.info.run_id

        mlflow_client.log_param(
            mlflow_run_id, "batch_size_mmu", config.training.batch_size_mmu
        )
        mlflow_client.log_param(
            mlflow_run_id, "batch_size_t2i", config.training.batch_size_t2i
        )
        mlflow_client.log_param(
            mlflow_run_id, "batch_size_lm", config.training.batch_size_lm
        )
        mlflow_client.log_param(
            mlflow_run_id, "learning_rate", config.optimizer.params.learning_rate
        )
        mlflow_client.log_param(
            mlflow_run_id, "num_experts", config.moe.get("num_experts", 4)
        )
        mlflow_client.log_param(mlflow_run_id, "top_k", config.moe.get("top_k", 2))
        mlflow_client.log_param(
            mlflow_run_id, "max_train_steps", config.training.max_train_steps
        )
        mlflow_client.log_param(
            mlflow_run_id,
            "gradient_accumulation_steps",
            config.training.gradient_accumulation_steps,
        )

        logger.info(f"✅ MLflow run started: {mlflow_run_id}")

    if accelerator.is_main_process:
        os.makedirs(config.experiment.output_dir, exist_ok=True)
        config_path = Path(config.experiment.output_dir) / "config.yaml"
        logging.info(f"Saving config to {config_path}")
        OmegaConf.save(config, config_path)

    # If passed along, set the training seed now.
    if config.training.seed is not None:
        set_seed(config.training.seed)

    #########################
    # MODELS and OPTIMIZER  #
    #########################
    logger.info("Loading models and optimizer")

    tokenizer = AutoTokenizer.from_pretrained(
        config.model.showo.llm_model_path, padding_side="left"
    )

    # unified prompting for show-o
    uni_prompting = UniversalPrompting(
        tokenizer,
        max_text_len=config.dataset.preprocessing.max_seq_length,
        special_tokens=(
            "<|soi|>",
            "<|eoi|>",
            "<|sov|>",
            "<|eov|>",
            "<|t2i|>",
            "<|mmu|>",
            "<|t2v|>",
            "<|v2v|>",
            "<|lvg|>",
        ),
        ignore_id=-100,
        cond_dropout_prob=config.training.cond_dropout_prob,
    )

    print("special tokens : \n", uni_prompting.sptids_dict)

    # VQ model for processing image into discrete tokens
    vq_model = get_vq_model_class(config.model.vq_model.type)
    if config.model.vq_model.get("pretrained_model_path", None):
        vq_model = vq_model().to(accelerator.device)
        state_dict = torch.load(config.model.vq_model.pretrained_model_path)["model"]
        vq_model.load_state_dict(state_dict)
    else:
        vq_model = vq_model.from_pretrained(config.model.vq_model.vq_model_name).to(
            accelerator.device
        )
    vq_model.eval()
    vq_model.requires_grad_(False)

    # Initialize Show-o model
    if config.model.showo.load_from_showo:
        logger.info(f"Loading model from {config.model.showo.pretrained_model_path}")
        model = Showo.from_pretrained(config.model.showo.pretrained_model_path).to(
            accelerator.device
        )
        logger.info(f"Original model vocab_size: {model.vocab_size}, mask_token_id: {model.mask_token_id}")
        logger.info(f"Original model codebook_size: {model.config.codebook_size}")
        logger.info(f"Config vocab_size: {config.model.showo.vocab_size}, codebook_size: {config.model.showo.codebook_size}")
        
        if config.model.showo.vocab_size != model.vocab_size:
            logger.warning(f"⚠️ Vocab size mismatch! Resizing embeddings from {model.vocab_size} to {config.model.showo.vocab_size}")
            model.showo.resize_token_embeddings(config.model.showo.vocab_size)
            model.config.codebook_size = config.model.showo.codebook_size
            model.config.vocab_size = config.model.showo.vocab_size
            model.vocab_size = config.model.showo.vocab_size
            model.output_size = config.model.showo.vocab_size
            model.config.mask_token_id = config.model.showo.vocab_size - 1
            model.mask_token_id = config.model.showo.vocab_size - 1
            logger.info(f"After resize - vocab_size: {model.vocab_size}, mask_token_id: {model.mask_token_id}")
    else:
        model = Showo(**config.model.showo).to(accelerator.device)
        logger.info(f"Created new model - vocab_size: {model.vocab_size}, mask_token_id: {model.mask_token_id}")

    # MoE disabled for vanilla training
    if config.get("moe", None) and config.moe.get("enabled", False):
        special_tokens = {
            "soi_id": uni_prompting.sptids_dict["<|soi|>"].item()
            if "<|soi|>" in uni_prompting.sptids_dict
            else None,
            "eoi_id": uni_prompting.sptids_dict["<|eoi|>"].item()
            if "<|eoi|>" in uni_prompting.sptids_dict
            else None,
            "sov_id": uni_prompting.sptids_dict["<|sov|>"].item()
            if "<|sov|>" in uni_prompting.sptids_dict
            else None,
            "eov_id": uni_prompting.sptids_dict["<|eov|>"].item()
            if "<|eov|>" in uni_prompting.sptids_dict
            else None,
        }
    
        model = patch_and_freeze_moe(
            model,
            count_layers_to_patch=config.moe["count_layers_to_patch"],
            num_experts=config.moe["num_experts"],
            top_k=config.moe["top_k"],
            mlflow_client=mlflow_client,
            mlflow_run_id=mlflow_run_id,
            special_tokens=special_tokens,
            modality_init_hardness=config.moe["modality_init_hardness"],
            modality_init_steps=config.moe["modality_init_steps"],
            modality_init_hardness_min=config.moe["modality_init_hardness_min"],
            use_gumbel=config.moe["use_gumbel"])

    mask_id = model.mask_token_id

    ##################################
    #   Optimizer and LR scheduler   #
    #################################
    optimizer = get_optimizer(
        optimizer_config=config.optimizer,
        named_parameters=model.named_parameters(),
        logger=logger,
    )

    # Create mask scheduler
    if config.get("mask_schedule", None) is not None:
        schedule = config.mask_schedule.schedule
        args = config.mask_schedule.get("params", {})
        mask_schedule = get_mask_chedule(schedule, **args)
    else:
        mask_schedule = get_mask_chedule(config.training.get("mask_schedule", "cosine"))

    lr_scheduler = get_scheduler(
        config.lr_scheduler.scheduler,
        optimizer=optimizer,
        num_training_steps=config.training.max_train_steps,
        num_warmup_steps=config.lr_scheduler.params.warmup_steps,
    )

    balance_init_coeff = float(config.training["balance_coeff"])
    balance_warmup_steps = (
        config.moe["balance_warmup_steps"]
    )
    balance_gamma = float(config.moe["balance_gamma"])
    _dummy_param = nn.Parameter(torch.zeros((), device=accelerator.device))
    balance_optimizer = torch.optim.SGD([{"params": [_dummy_param], "lr": balance_init_coeff}])
    balance_warmup = LinearLR(balance_optimizer, start_factor=1e-8, total_iters=max(int(balance_warmup_steps), 1))
    balance_decay = ExponentialLR(balance_optimizer, gamma=float(balance_gamma))
    balance_scheduler = SequentialLR(
        balance_optimizer,
        schedulers=[balance_warmup, balance_decay],
        milestones=[max(int(balance_warmup_steps), 1)],
    )

    temp_start = float(config.moe["temp_start"]) if "temp_start" in config.moe else 100.0
    temp_end = float(config.moe["temp_end"]) if "temp_end" in config.moe else 1.0
    temp_steps = int(config.moe["temp_steps"]) if "temp_steps" in config.moe else int(config.training.max_train_steps)
    _temp_dummy_param = nn.Parameter(torch.zeros((), device=accelerator.device))
    temp_optimizer = torch.optim.SGD([{"params": [_temp_dummy_param], "lr": temp_start}])
    def _linear_factor(step: int):
        s = min(int(step), int(max(temp_steps, 1)))
        a = s / max(temp_steps, 1)
        return (1.0 - a) + a * (temp_end / max(temp_start, 1e-8))
    from torch.optim.lr_scheduler import LambdaLR
    temp_scheduler = LambdaLR(temp_optimizer, lr_lambda=_linear_factor)

    ##################################
    #         DATALOADER             #
    #################################
    # Проверяем, нужна ли функция create_imagetext_dataloader для parquet режимов
    create_imagetext_dataloader_fn = None
    if (
        config.dataset.gen_type == "t2i_parquet"
        or config.dataset.und_type == "captioning_parquet"
    ):
        try:
            from datasets import (
                create_imagetext_dataloader as create_imagetext_dataloader_fn,
            )
        except ImportError:
            logger.warning(
                "create_imagetext_dataloader not available, but required for parquet mode"
            )

    combined_dataloader, num_update_steps_per_epoch, num_train_epochs = (
        create_dataloaders(
            config=config,
            accelerator=accelerator,
            tokenizer=tokenizer,
            create_imagetext_dataloader=create_imagetext_dataloader_fn,
        )
    )

    ##################################
    #         MODEL RESUME          #
    #################################
    global_step = 0
    first_epoch = 0

    if config.experiment.resume_from_checkpoint:
        dirs = os.listdir(config.experiment.output_dir)
        dirs = [d for d in dirs if d.startswith("checkpoint")]
        dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
        path = dirs[-1] if len(dirs) > 0 else None
        if path is not None:
            path = os.path.join(config.experiment.output_dir, path)

            global_step = int(os.path.basename(path).split("-")[1])
            first_epoch = global_step // num_update_steps_per_epoch

            accelerator.print(
                f"Resuming from checkpoint {path}/unwrapped_model/pytorch_model.bin"
            )
            state_dict = torch.load(
                f"{path}/unwrapped_model/pytorch_model.bin", map_location="cpu"
            )
            model.load_state_dict(state_dict, strict=True)
            del state_dict

    ##################################
    #       Prepare accelerator     #
    #################################
    logger.info("Preparing model, optimizer and dataloaders")
    model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)

    vq_model.to(device=accelerator.device)

    if hasattr(model, "module"):
        mask_dtype = model.module.showo.model.embed_tokens.weight.dtype
    else:
        mask_dtype = accelerator.unwrap_model(
            model
        ).showo.model.embed_tokens.weight.dtype

    ##################################
    #             Training          #
    #################################
    logger.info("***** Running training *****")
    logger.info(f"  Num training steps = {config.training.max_train_steps}")
    logger.info(f"  Instantaneous batch size per device = {total_batch_size_per_gpu}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(
        f"  Gradient Accumulation steps = {config.training.gradient_accumulation_steps}"
    )

    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    for epoch in range(first_epoch, num_train_epochs):
        model.train()

        pbar = None
        if accelerator.is_main_process:
            pbar = tqdm(
                total=config.training.max_train_steps - global_step,
                desc=f"Epoch {epoch}",
                initial=0,
                unit="step",
                colour="green",
            )

        for batch, batch_idx, dataloader_idx in combined_dataloader:
            data_time_m.update(time.time() - end)

            with accelerator.accumulate(model):
                step_outputs = train_step(
                    batch=batch,
                    epoch=epoch,
                    global_step=global_step,
                    model=model,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    balance_scheduler=balance_scheduler,
                    temp_scheduler=temp_scheduler,
                    accelerator=accelerator,
                    config=config,
                    uni_prompting=uni_prompting,
                    vq_model=vq_model,
                    mask_dtype=mask_dtype,
                    mask_id=mask_id,
                    mask_schedule=mask_schedule,
                    batch_time_m=batch_time_m,
                    data_time_m=data_time_m,
                    total_batch_size_per_gpu=total_batch_size_per_gpu,
                    mlflow_client=mlflow_client,
                    mlflow_run_id=mlflow_run_id,
                    pbar=pbar,
                )

            input_ids = step_outputs["input_ids"]
            input_ids_t2i = step_outputs["input_ids_t2i"]
            attention_mask = step_outputs["attention_mask"]
            labels = step_outputs["labels"]
            batch_size_t2i = step_outputs["batch_size_t2i"]
            batch_size_lm = step_outputs["batch_size_lm"]
            batch_size_mmu = step_outputs["batch_size_mmu"]
            image_tokens_ori = step_outputs["image_tokens_ori"]
            texts = step_outputs["texts"]
            logits = step_outputs["logits"]
            # avg_loss_t2i = step_outputs["avg_loss_t2i"]
            # avg_loss_lm = step_outputs["avg_loss_lm"]
            # avg_loss_mmu = step_outputs["avg_loss_mmu"]
            # balance_loss = step_outputs["balance_loss"]
            # avg_masking_rate = step_outputs["avg_masking_rate"]

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                batch_time_m.update(time.time() - end)
                end = time.time()
                if (
                    (global_step + 1) % 100 == 0
                    and config.get("moe", {}).get("enabled", False)
                    and accelerator.is_main_process
                ):
                    collect_and_log_moe_activations(
                        model=model,
                        accelerator=accelerator,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        config=config,
                        batch_size_t2i=batch_size_t2i,
                        batch_size_lm=batch_size_lm,
                        batch_size_mmu=batch_size_mmu,
                        global_step=global_step + 1,
                        mlflow_client=mlflow_client,
                        mlflow_run_id=mlflow_run_id,
                    )

                # Save model checkpoint
                if (global_step + 1) % config.experiment.save_every == 0:
                    save_checkpoint(model, config, accelerator, global_step + 1)

                # print(f"global_step: {global_step + 1}, config.experiment.generate_every: {config.experiment.generate_every}")
                # Debug logging for generation
                should_generate = (global_step + 1) == 1 or (global_step + 1) % config.experiment.generate_every == 0
                if (global_step + 1) % config.experiment.generate_every == 0:
                    logger.info(f"🎨 Step {global_step + 1}: should_generate={should_generate}, is_main_process={accelerator.is_main_process}")

                if should_generate and accelerator.is_main_process:
                    generate_images(
                        model,
                        vq_model,
                        uni_prompting,
                        accelerator,
                        config,
                        global_step + 1,
                        mask_schedule=mask_schedule,
                        mlflow_client=mlflow_client,
                        mlflow_run_id=mlflow_run_id,
                    )

                    visualize_predictions(
                        model,
                        vq_model,
                        uni_prompting,
                        config,
                        global_step + 1,
                        input_ids_t2i,
                        image_tokens_ori,
                        batch["t2i_flow"]["images"],
                        texts,
                        logits,
                        mlflow_client=mlflow_client,
                        mlflow_run_id=mlflow_run_id,
                    )

                    # Отключаем evaluate_mmu для моделей с CLIP ViT (несовместимо с VQ tokens)
                    # if not config.model.showo.get("w_clip_vit", False):
                    #     evaluate_mmu(
                    #         model,
                    #         vq_model,
                    #         uni_prompting,
                    #         accelerator,
                    #         config,
                    #         global_step + 1,
                    #         batch["mmu_flow"],
                    #         mlflow_client=mlflow_client,
                    #         mlflow_run_id=mlflow_run_id,
                    #     )

                global_step += 1

            if global_step >= config.training.max_train_steps:
                break
        if accelerator.is_main_process:
            pbar.close()

    accelerator.wait_for_everyone()
    save_checkpoint(model, config, accelerator, global_step)

    if mlflow_client is not None and mlflow_run_id is not None:
        mlflow_client.set_terminated(mlflow_run_id, status="FINISHED")
        logger.info("MLflow run finished")

    if accelerator.is_main_process:
        model = accelerator.unwrap_model(model)
        model.save_pretrained(config.experiment.output_dir, safe_serialization=False)

    accelerator.end_training()


if __name__ == "__main__":
    main()
