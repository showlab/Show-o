# coding=utf-8
# Copyright 2025 NUS Show Lab, HuggingFace.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import json
import logging
import math
import shutil
import time
from pathlib import Path
from typing import Union
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
import wandb
import torch
from torch.optim import AdamW
from einops import rearrange
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedType, set_seed
from models import Showo2Qwen2_5, omni_attn_mask_naive
from models.lr_schedulers import get_scheduler
from models.my_logging import set_verbosity_info, set_verbosity_error
from models.misc import prepare_gen_input, get_text_tokenizer, get_weight_type
from torch.nn.attention.flex_attention import flex_attention

if torch.cuda.is_available():
    flex_attention = torch.compile(flex_attention)

from datasets import create_imagetext_dataloader, MixedDataLoader
from utils import get_config, flatten_omega_conf, AverageMeter, denorm, denorm_vid, get_hyper_params, \
    path_to_llm_name, _freeze_params

from transport import Sampler, create_transport

logger = get_logger(__name__, log_level="INFO")


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
    accelerator = Accelerator(
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        mixed_precision=config.training.mixed_precision,
        log_with="wandb",
        project_dir=config.experiment.logging_dir,
        split_batches=True,
    )

    bs_t2i = config.training.batch_size_t2i
    bs_mmu = config.training.batch_size_mmu

    if "concat" in config.dataset.mixed_loader_mode:
        assert config.dataset.accumulation == 1, "No need to enable accumulation in mixed-dataloader!"
        total_batch_size_per_gpu = bs_t2i + bs_mmu
        total_batch_size_without_accum = total_batch_size_per_gpu * accelerator.num_processes
        total_batch_size = total_batch_size_without_accum * config.training.gradient_accumulation_steps
    else:
        assert bs_t2i == bs_mmu, "We should ensure batch size is consistent at each iteration if we use FlexAttention!"
        total_batch_size_per_gpu = bs_t2i * config.dataset.accumulation
        total_batch_size_without_accum = total_batch_size_per_gpu * accelerator.num_processes
        total_batch_size = total_batch_size_without_accum * config.training.gradient_accumulation_steps

    if accelerator.distributed_type == DistributedType.DEEPSPEED:
        accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = (
            total_batch_size_per_gpu
        )

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

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        resume_wandb_run = config.wandb.resume
        run_id = config.wandb.get("run_id", None)
        if run_id is None:
            resume_wandb_run = False
            run_id = wandb.util.generate_id()
            config.wandb.run_id = run_id

        wandb_init_kwargs = dict(
            name=config.experiment.name,
            id=run_id,
            resume=resume_wandb_run,
            entity=config.wandb.get("entity", None),
            config_exclude_keys=[],
        )
        wandb_config = {k: v for k, v in flatten_omega_conf(config, resolve=True)}
        wandb_config.pop("experiment.resume_from_checkpoint")

        accelerator.init_trackers(
            config.experiment.project,
            config=wandb_config,
            init_kwargs={"wandb": wandb_init_kwargs},
        )

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

    weight_type = get_weight_type(config)

    # VQ model for processing image into discrete tokens
    if config.model.vae_model.type == 'wan21':
        from models import WanVAE
        vae_model = WanVAE(vae_pth=config.model.vae_model.pretrained_model_path, dtype=weight_type,
                           device=accelerator.device)
    else:
        raise NotImplementedError

    # Initialize Show-o model
    text_tokenizer, showo_token_ids = get_text_tokenizer(config.model.showo.llm_model_path, add_showo_tokens=True,
                                                         return_showo_token_ids=True,
                                                         llm_name=path_to_llm_name[config.model.showo.llm_model_path])
    config.model.showo.llm_vocab_size = len(text_tokenizer)

    if config.model.showo.load_from_showo:
        model = Showo2Qwen2_5.from_pretrained(config.model.showo.pretrained_model_path, use_safetensors=False).to(accelerator.device)
    else:
        model = Showo2Qwen2_5(**config.model.showo).to(accelerator.device)

    # load pre-distilled semantic layers
    model.load_state_dict(torch.load("pre-distilled_semantic_layers.pt", map_location="cpu"), strict=False)

    # Choose layers to freeze
    _freeze_params(model, config.model.showo.frozen_params)

    preproc_config = config.dataset.preprocessing
    dataset_config = config.dataset.params

    # for time embedding
    if config.model.showo.add_time_embeds:
        # we prepend the time embedding to vision tokens
        config.dataset.preprocessing.num_mmu_image_tokens += 1
        config.dataset.preprocessing.num_t2i_image_tokens += 1
        config.dataset.preprocessing.num_hq_image_tokens += 1
        config.dataset.preprocessing.num_video_tokens += 1
        config.dataset.preprocessing.num_mixed_modal_tokens += 1

    ##################################
    #   Optimizer and LR scheduler   #
    #################################
    optimizer_config = config.optimizer.params
    optimizer_type = config.optimizer.name

    if optimizer_type == "adamw":
        optimizer = AdamW(
            model.parameters(),
            betas=(optimizer_config.beta1, optimizer_config.beta2),
            weight_decay=optimizer_config.weight_decay,
            eps=optimizer_config.epsilon,
        )
    else:
        raise ValueError(f"Optimizer {optimizer_type} not supported")

    ##################################
    #         DATALOADER             #
    #################################
    logger.info("Creating dataloaders and lr_scheduler")

    # DataLoaders creation:
    # We use webdataset for data loading. The dataloaders are created with sampling with replacement.
    # We don't do dataset resuming here, instead we resample the shards and buffer each time. The sampling is stochastic.
    # This means that the dataloading is not deterministic, but it's fast and efficient.

    # Data for generation
    train_dataloader_t2i = create_imagetext_dataloader(
        train_shards_path_or_url=dataset_config.train_t2i_shards_path_or_url,
        batch_size=config.training.batch_size_t2i,
        text_tokenizer=text_tokenizer,
        image_size=preproc_config.resolution,
        max_seq_len=preproc_config.max_seq_length,
        num_image_tokens=preproc_config.num_t2i_image_tokens,
        latent_width=preproc_config.latent_width,
        latent_height=preproc_config.latent_height,
        cond_dropout_prob=config.training.cond_dropout_prob,
        num_workers=dataset_config.num_workers,
        drop_last=True,
        shuffle=True,
        min_res=preproc_config.min_res,
        random_und_or_gen=preproc_config.random_und_or_gen,
        showo_token_ids=showo_token_ids,
        system=("", "", ""),
        accelerator=accelerator
    )

    # Data for understanding
    train_dataloader_mmu = create_imagetext_dataloader(
        train_shards_path_or_url=dataset_config.train_mmu_shards_path_or_url,
        batch_size=config.training.batch_size_mmu,
        text_tokenizer=text_tokenizer,
        image_size=preproc_config.resolution,
        max_seq_len=preproc_config.max_seq_length,
        num_image_tokens=preproc_config.num_mmu_image_tokens,
        latent_width=preproc_config.latent_width,
        latent_height=preproc_config.latent_height,
        cond_dropout_prob=config.training.cond_dropout_prob,
        num_workers=dataset_config.num_workers,
        drop_last=True,
        shuffle=True,
        min_res=preproc_config.min_res,
        random_und_or_gen=preproc_config.random_und_or_gen,
        showo_token_ids=showo_token_ids,
        system=("", "", ""),
        is_captioning=True,
        accelerator=accelerator
    )

    num_update_steps_per_epoch = len(train_dataloader_t2i)
    num_train_epochs = math.ceil(config.training.max_train_steps / num_update_steps_per_epoch)

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

            accelerator.print(f"Resuming from checkpoint {path}/unwrapped_model/pytorch_model.bin")
            state_dict = torch.load(f'{path}/unwrapped_model/pytorch_model.bin', map_location="cpu")

            # not load some parameters
            if config.model.showo.params_not_load is not None:
                params_to_delete = []
                for k in state_dict:
                    for n in config.model.showo.params_not_load:
                        if n in k:
                            params_to_delete.append(k)
                for k in params_to_delete:
                    del state_dict[k]

            model.load_state_dict(state_dict, strict=False if config.model.showo.params_not_load is not None else True)
            del state_dict

        # recover_dataloader_state(accelerator.process_index, train_dataloader_t2i, config.experiment.output_dataloader_state_dir)

    # Combine these dataloaders into a single iterable model
    mixed_loader = MixedDataLoader(
        loader_list=[train_dataloader_t2i, train_dataloader_mmu],
        samp_probs=config.dataset.samp_probs,
        accumulation=config.dataset.accumulation,
        mode=config.dataset.mixed_loader_mode
    )

    lr_scheduler = get_scheduler(
        config.lr_scheduler.scheduler,
        optimizer=optimizer,
        num_training_steps=config.training.max_train_steps - global_step,
        num_warmup_steps=config.lr_scheduler.params.warmup_steps,
    )

    ##################################
    #       Prepare accelerator     #
    #################################
    logger.info("Preparing model, optimizer and dataloaders")
    model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)

    ##################################
    #             Training          #
    #################################
    logger.info("***** Running training *****")
    logger.info(f"  Num training steps = {config.training.max_train_steps}")
    logger.info(f"  Instantaneous batch size per device = {total_batch_size_per_gpu}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.training.gradient_accumulation_steps}")

    # default: 1000 steps, linear noise schedule
    transport = create_transport(
        path_type=config.transport.path_type,
        prediction=config.transport.prediction,
        loss_weight=config.transport.loss_weight,
        train_eps=config.transport.train_eps,
        sample_eps=config.transport.sample_eps,
        snr_type=config.transport.snr_type,
        do_shift=config.transport.do_shift,
        seq_len=preproc_config.num_t2i_image_tokens,
    )  # default: velocity;

    sampler = Sampler(transport)

    @torch.no_grad()
    def prepare_latents_and_labels(
            pixel_values: Union[torch.FloatTensor, torch.LongTensor],
            data_type,
            shape,
            image_masks,
            modality_positions
    ):

        if config.model.vae_model.type == 'wan21':
            if len(pixel_values.shape) == 4:
                pixel_values = pixel_values.unsqueeze(2)
            image_latents = vae_model.sample(pixel_values)
            recons_images = vae_model.batch_decode(image_latents)
            if pixel_values.shape[2] == 1:
                image_latents = image_latents.squeeze(2)
                recons_images = recons_images.squeeze(2)
        else:
            raise NotImplementedError

        c, h, w = image_latents.shape[1:]
        # timesteps, noise, original image
        # each for loop takes around 0.002, which is affordable
        t_list, xt_list, ut_list, masks = [], [], [], []
        for i, tp in enumerate(data_type):
            # x0->noise x1->image
            t, x0, x1 = transport.sample(image_latents[i][None],
                                         config.training.und_max_t0 if tp in ['mmu', 'mmu_vid'] else None)
            # timesteps, noised image, velocity
            t, xt, ut = transport.path_sampler.plan(t, x0, x1)
            t_list.append(t)
            xt_list.append(xt)
            ut_list.append(ut)
            if data_type[0] != 'interleaved_data':
                if tp in ['mmu', 'mmu_vid'] and config.training.und_max_t0 == 1.0:
                    masks.append(image_masks[i][None] * 0.0)
                else:
                    masks.append(image_masks[i][None])

        t = torch.stack(t_list, dim=0).squeeze(-1)
        xt = torch.cat(xt_list, dim=0)
        ut = torch.cat(ut_list, dim=0)

        if len(masks) != 0:
            masks = torch.cat(masks, dim=0)
        else:
            masks = image_masks

        if data_type[0] == 'interleaved_data':
            b, n = shape
            image_latents = image_latents.reshape(b, n, c, h, w)
            ut = ut.reshape(b, n, c, h, w)
            xt = xt.reshape(b, n, c, h, w)
            t = t.reshape(b, n)

            # only denoise the last image
            if preproc_config.only_denoise_last_image:
                for i in range(b):
                    non_zero_max_idx = max([i for i, pos in enumerate(modality_positions[i]) if pos[1] != 0])
                    xt[i, :non_zero_max_idx] = image_latents[i][None][:, :non_zero_max_idx].clone()
                    # ut[i, :non_zero_max_idx] = torch.zeros_like(image_latents[i][None][:, :non_zero_max_idx])
                    t[i, :non_zero_max_idx] = t[i, :non_zero_max_idx] * 0.0 + 1.0

                    for j in range(non_zero_max_idx):
                        img_sid, length = modality_positions[i, j]
                        masks[i, img_sid: img_sid + length] = 0

            ut = ut.reshape(b * n, c, h, w)
            xt = xt.reshape(b * n, c, h, w)
            t = t.reshape(b * n)

        return xt, t, ut, recons_images, masks

    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    for epoch in range(first_epoch, num_train_epochs):
        model.train()
        for batch in mixed_loader:

            text_tokens = batch['text_tokens'].to(accelerator.device)
            text_labels = batch['text_labels'].to(accelerator.device)
            pixel_values = batch['images'].to(accelerator.device).to(weight_type)
            if batch['data_type'][0] == 'interleaved_data':
                b, n = pixel_values.shape[:2]
                pixel_values = rearrange(pixel_values, "b n c h w -> (b n) c h w")
                batch['data_type'] = batch['data_type'] * n
            else:
                b, n = 0, 0

            text_masks = batch['text_masks'].to(accelerator.device)
            image_masks = batch['image_masks'].to(accelerator.device)
            modality_positions = batch['modality_positions'].to(accelerator.device)
            # prepare image latents and labels
            image_latents, t, image_labels, recons_images, image_masks = prepare_latents_and_labels(pixel_values,
                                                                                                    batch['data_type'],
                                                                                                    (b, n),
                                                                                                    image_masks,
                                                                                                    modality_positions)
            # B=None would potentially induce loss spike when there are a lot of ignored labels (-100) in the batch
            # we must set B=text_tokens.shape[0] (loss spike may still happen sometimes)
            # omni_mask_fn = omni_attn_mask(modality_positions)
            # block_mask = create_block_mask(omni_mask_fn, B=text_tokens.shape[0], H=None,
            #                                Q_LEN=preproc_config.max_seq_length,
            #                                KV_LEN=preproc_config.max_seq_length, device=accelerator.device)
            # or use naive omni attention mask, which is more stable
            block_mask = omni_attn_mask_naive(text_tokens.size(0),
                                              text_tokens.size(1),
                                              modality_positions,
                                              accelerator.device).to(weight_type)

            logits, loss_ntp, loss_flow = model(text_tokens=text_tokens,
                                                image_latents=image_latents,
                                                t=t.to(weight_type),
                                                attention_mask=block_mask,
                                                text_masks=text_masks,
                                                image_masks=image_masks,
                                                text_labels=text_labels,
                                                image_labels=image_labels,
                                                modality_positions=modality_positions,
                                                output_hidden_states=True,
                                                max_seq_len=text_tokens.size(1),
                                                device=accelerator.device,
                                                )

            # Gather the losses across all processes for logging (if we use distributed training).
            avg_loss_ntp = accelerator.gather(loss_ntp.repeat(total_batch_size_per_gpu)).mean()
            avg_loss_flow = accelerator.gather(loss_flow.repeat(total_batch_size_per_gpu)).mean()
            loss = config.training.ntp_coeff * loss_ntp + config.training.flow_coeff * loss_flow

            accelerator.backward(loss.to(weight_type) / config.training.gradient_accumulation_steps)

            if config.training.max_grad_norm is not None and accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)

            if (global_step + 1) % config.training.gradient_accumulation_steps == 0:
                optimizer.step()
                lr_scheduler.step()

            # log gradient norm before zeroing it
            if (
                    accelerator.sync_gradients
                    and (global_step + 1) % config.experiment.log_grad_norm_every == 0
                    and accelerator.is_main_process
            ):
                log_grad_norm(model, accelerator, global_step + 1)

            if (global_step + 1) % config.training.gradient_accumulation_steps == 0:
                optimizer.zero_grad(set_to_none=True)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:

                batch_time_m.update(time.time() - end)
                end = time.time()

                # Log metrics
                if (global_step + 1) % config.experiment.log_every == 0:
                    samples_per_second_per_gpu = (
                            config.training.gradient_accumulation_steps * total_batch_size_per_gpu / batch_time_m.val
                    )
                    lr = [group["lr"] for group in optimizer.param_groups]
                    if len(lr) == 3:
                        logs = {
                            "step_loss_ntp": avg_loss_ntp.item(),
                            "step_loss_flow": avg_loss_flow.item(),
                            "lr_ve": lr[0],
                            "lr_proj": lr[1],
                            "lr_showo": lr[2],
                            "samples/sec/gpu": samples_per_second_per_gpu,
                            "data_time": data_time_m.val,
                            "batch_time": batch_time_m.val,
                        }
                        accelerator.log(logs, step=global_step + 1)
                        logger.info(
                            f"Epoch: {epoch} "
                            f"Step: {global_step + 1} "
                            f"Loss_NTP: {avg_loss_ntp.item():0.4f} "
                            f"Loss_FLOW: {avg_loss_flow.item():0.4f} "
                            f"Data (t): {data_time_m.val:0.4f}, {samples_per_second_per_gpu:0.2f}/s/gpu "
                            f"Batch (t): {batch_time_m.val:0.4f} "
                            f"LR_ve: {lr[0]:0.6f} "
                            f"LR_proj: {lr[1]:0.6f} "
                            f"LR_showo: {lr[2]:0.6f}"
                        )
                    else:
                        logs = {
                            "step_loss_ntp": avg_loss_ntp.item(),
                            "step_loss_flow": avg_loss_flow.item(),
                            "lr_proj": lr[0],
                            "samples/sec/gpu": samples_per_second_per_gpu,
                            "data_time": data_time_m.val,
                            "batch_time": batch_time_m.val,
                        }
                        accelerator.log(logs, step=global_step + 1)
                        logger.info(
                            f"Epoch: {epoch} "
                            f"Step: {global_step + 1} "
                            f"Loss_NTP: {avg_loss_ntp.item():0.4f} "
                            f"Loss_FLOW: {avg_loss_flow.item():0.4f} "
                            f"Data (t): {data_time_m.val:0.4f}, {samples_per_second_per_gpu:0.2f}/s/gpu "
                            f"Batch (t): {batch_time_m.val:0.4f} "
                            f"LR_proj: {lr[0]:0.6f}"
                        )
                    # resetting batch / data time meters per log window
                    batch_time_m.reset()
                    data_time_m.reset()

                # Save model checkpoint
                if (global_step + 1) % config.experiment.save_every == 0:
                    save_checkpoint(model, config, accelerator, global_step + 1)
                    # save_dataloader_state(accelerator.process_index, train_dataloader_t2i, config.experiment.output_dataloader_state_dir)

                if (global_step + 1) % config.experiment.generate_every == 0 and accelerator.is_main_process:
                    # generate_videos(
                    #     model,
                    #     vae_model,
                    #     text_tokenizer,
                    #     config,
                    #     global_step + 1,
                    #     accelerator.device,
                    #     weight_type,
                    #     sampler,
                    #     showo_token_ids,
                    # )

                    generate_images(
                        model,
                        vae_model,
                        text_tokenizer,
                        config,
                        global_step + 1,
                        accelerator.device,
                        weight_type,
                        sampler,
                        showo_token_ids
                    )

                    if batch['data_type'][0] != "interleaved_data":
                        if len(image_latents.shape) == 5:
                            visualize_reconstruction_video(
                                pixel_values,
                                recons_images,
                                batch['texts'],
                                global_step + 1,
                            )
                        else:
                            visualize_reconstruction(
                                pixel_values,
                                recons_images,
                                batch['texts'],
                                global_step + 1,
                            )

                global_step += 1

            # Stop training if max steps is reached
            if global_step >= config.training.max_train_steps:
                break
            # End for

    accelerator.wait_for_everyone()

    # Evaluate and save checkpoint at the end of training
    save_checkpoint(model, config, accelerator, "final")
    # save_dataloader_state(accelerator.process_index, train_dataloader_t2i, config.experiment.output_dataloader_state_dir)
    # logger.info(f"Saved dataloader state to {config.experiment.output_dataloader_state_dir}")

    # Save the final trained checkpoint
    if accelerator.is_main_process:
        model = accelerator.unwrap_model(model)
        model.save_pretrained(config.experiment.output_dir, safe_serialization=False)

    accelerator.end_training()


@torch.no_grad()
def generate_images(
        model,
        vae_model,
        text_tokenizer,
        config,
        global_step,
        device,
        weight_type,
        sampler,
        showo_token_ids,
):
    logger.info("Generating images...")
    model.eval()

    # read validation prompts from file
    with open(config.dataset.params.validation_prompts_file, "r") as f:
        prompts = f.read().splitlines()[:config.training.batch_size_t2i]

    num_image_tokens, num_video_tokens, max_seq_len, max_text_len, image_latent_dim, patch_size, latent_width, \
    latent_height, pad_id, bos_id, eos_id, boi_id, eoi_id, bov_id, eov_id, image_pad_id, video_pad_id, guidance_scale \
        = get_hyper_params(config, text_tokenizer, showo_token_ids, is_hq=True)

    batch_text_tokens, batch_text_tokens_null, batch_modality_positions, batch_modality_positions_null = \
        prepare_gen_input(
            prompts, text_tokenizer, num_image_tokens, bos_id, eos_id, boi_id, eoi_id, pad_id, image_pad_id,
            max_text_len, device
        )

    z = torch.randn((len(prompts),
                     image_latent_dim, latent_height * patch_size,
                     latent_width * patch_size)).to(weight_type).to(device)

    if guidance_scale > 0:
        z = torch.cat([z, z], dim=0)
        text_tokens = torch.cat([batch_text_tokens, batch_text_tokens_null], dim=0)
        modality_positions = torch.cat([batch_modality_positions, batch_modality_positions_null], dim=0)
        # B=None would potentially induce loss spike when there are a lot of ignored labels (-100) in the batch
        # we must set B=text_tokens.shape[0] (loss spike may still happen sometimes)
        # omni_mask_fn = omni_attn_mask(modality_positions)
        # block_mask = create_block_mask(omni_mask_fn, B=z.size(0), H=None, Q_LEN=max_seq_len,
        #                                KV_LEN=max_seq_len, device=device)
        # or use naive omni attention mask, which is more stable
        block_mask = omni_attn_mask_naive(text_tokens.size(0),
                                          max_seq_len,
                                          modality_positions,
                                          device).to(weight_type)
    else:
        text_tokens = batch_text_tokens
        modality_positions = batch_modality_positions
        # B=None would potentially induce loss spike when there are a lot of ignored labels (-100) in the batch
        # we must set B=text_tokens.shape[0] (loss spike may still happen sometimes)
        # omni_mask_fn = omni_attn_mask(modality_positions)
        # block_mask = create_block_mask(omni_mask_fn, B=z.size(0), H=None, Q_LEN=max_seq_len,
        #                                KV_LEN=max_seq_len, device=device)
        block_mask = omni_attn_mask_naive(text_tokens.size(0),
                                          max_seq_len,
                                          modality_positions,
                                          device).to(weight_type)

    model_kwargs = dict(
        text_tokens=torch.cat([batch_text_tokens, batch_text_tokens_null], dim=0),
        attention_mask=block_mask,
        modality_positions=torch.cat([batch_modality_positions,
                                      batch_modality_positions_null], dim=0),
        output_hidden_states=True,
        max_seq_len=max_seq_len,
        guidance_scale=guidance_scale
    )

    sample_fn = sampler.sample_ode(
        sampling_method=config.transport.sampling_method,
        num_steps=config.transport.num_inference_steps,
        atol=config.transport.atol,
        rtol=config.transport.rtol,
        reverse=config.transport.reverse,
        time_shifting_factor=config.transport.time_shifting_factor
    )
    samples = sample_fn(z, model.t2i_generate, **model_kwargs)[-1]
    samples = torch.chunk(samples, 2)[0]

    if config.model.vae_model.type == 'wan21':
        samples = samples.unsqueeze(2)
        images = vae_model.batch_decode(samples)
        images = images.squeeze(2)
    else:
        raise NotImplementedError

    model.train()

    # Convert to PIL images
    images = denorm(images)
    pil_images = [Image.fromarray(image) for image in images]

    # Log images
    wandb_images = [wandb.Image(image, caption=prompts[i]) for i, image in enumerate(pil_images)]
    wandb.log({"Generated images": wandb_images}, step=global_step)


@torch.no_grad()
def visualize_reconstruction(
        pixel_values,
        recons_images,
        captions,
        global_step
):
    logger.info("Visualizing images...")

    # Convert to PIL images
    images = denorm(pixel_values)
    recons_images = denorm(recons_images)
    visualized_images = np.concatenate((images, recons_images), 2)
    pil_images = [Image.fromarray(image) for image in visualized_images]

    # Log images
    wandb_images = [wandb.Image(image, caption=captions[i]) for i, image in enumerate(pil_images)]
    wandb.log({"Original images vs. Reconstructed": wandb_images}, step=global_step)


@torch.no_grad()
def generate_videos(
        model,
        vae_model,
        text_tokenizer,
        config,
        global_step,
        device,
        weight_type,
        sampler,
        showo_token_ids
):
    logger.info("Generating videos...")
    model.eval()

    # read validation prompts from file
    with open(config.dataset.params.validation_prompts_file, "r") as f:
        prompts = f.read().splitlines()[:config.training.batch_size_t2i]

    num_image_tokens, num_video_tokens, max_seq_len, max_text_len, image_latent_dim, patch_size, latent_width, \
    latent_height, pad_id, bos_id, eos_id, boi_id, eoi_id, bov_id, eov_id, image_pad_id, video_pad_id, guidance_scale \
        = get_hyper_params(config, text_tokenizer, showo_token_ids, is_video=True)

    batch_text_tokens, batch_text_tokens_null, batch_modality_positions, batch_modality_positions_null = \
        prepare_gen_input(
            prompts, text_tokenizer, num_video_tokens, bos_id, eos_id, bov_id, eov_id, pad_id, video_pad_id,
            max_text_len, device
        )

    T = 5
    z = torch.randn((len(prompts), image_latent_dim, T, latent_height * patch_size, latent_width * patch_size)).to(
        device).to(weight_type)

    if guidance_scale > 0:
        z = torch.cat([z, z], dim=0)
        text_tokens = torch.cat([batch_text_tokens, batch_text_tokens_null], dim=0)
        modality_positions = torch.cat([batch_modality_positions, batch_modality_positions_null], dim=0)
        # B=None would potentially induce loss spike when there are a lot of ignored labels (-100) in the batch
        # we must set B=text_tokens.shape[0] (loss spike may still happen sometimes)
        # omni_mask_fn = omni_attn_mask(modality_positions)
        # block_mask = create_block_mask(omni_mask_fn, B=z.size(0), H=None, Q_LEN=max_seq_len,
        #                                KV_LEN=max_seq_len, device=device)
        # or use naive omni attention mask, which is more stable
        block_mask = omni_attn_mask_naive(text_tokens.size(0),
                                          max_seq_len,
                                          modality_positions,
                                          device).to(weight_type)
    else:
        text_tokens = batch_text_tokens
        modality_positions = batch_modality_positions
        # B=None would potentially induce loss spike when there are a lot of ignored labels (-100) in the batch
        # we must set B=text_tokens.shape[0] (loss spike may still happen sometimes)
        # omni_mask_fn = omni_attn_mask(modality_positions)
        # block_mask = create_block_mask(omni_mask_fn, B=z.size(0), H=None, Q_LEN=max_seq_len,
        #                                KV_LEN=max_seq_len, device=device)
        block_mask = omni_attn_mask_naive(text_tokens.size(0),
                                          max_seq_len,
                                          modality_positions,
                                          device).to(weight_type)

    model_kwargs = dict(
        text_tokens=text_tokens,
        attention_mask=block_mask,
        modality_positions=modality_positions,
        output_hidden_states=True,
        max_seq_len=max_seq_len,
        guidance_scale=guidance_scale
    )

    sample_fn = sampler.sample_ode(
        sampling_method=config.transport.sampling_method,
        num_steps=config.transport.num_inference_steps,
        atol=config.transport.atol,
        rtol=config.transport.rtol,
        reverse=config.transport.reverse,
        time_shifting_factor=config.transport.time_shifting_factor
    )
    samples = sample_fn(z, model.t2i_generate, **model_kwargs)[-1]
    samples = torch.chunk(samples, 2)[0]

    if config.model.vae_model.type == 'wan21':
        images = vae_model.batch_decode(samples)
    else:
        raise NotImplementedError

    model.train()

    # Convert to PIL images
    images = denorm_vid(images)

    # Log images
    wandb_images = [wandb.Video(image, caption=prompts[i], fps=8, format="mp4") for i, image in enumerate(images)]
    wandb.log({"Generated videos": wandb_images}, step=global_step)


@torch.no_grad()
def visualize_reconstruction_video(
        pixel_values,
        recons_images,
        captions,
        global_step
):
    logger.info("Visualizing videos...")

    # Convert to PIL images
    images = denorm_vid(pixel_values)
    recons_images = denorm_vid(recons_images)
    visualized_images = np.concatenate((images, recons_images), 4)

    # Log images
    wandb_images = [wandb.Video(image, caption=captions[i], fps=8, format="mp4") for i, image in
                    enumerate(visualized_images)]
    wandb.log({"Original videos vs. Reconstructed": wandb_images}, step=global_step)


def save_checkpoint(model, config, accelerator, global_step):
    output_dir = config.experiment.output_dir
    checkpoints_total_limit = config.experiment.get("checkpoints_total_limit", None)

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
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            save_path / "unwrapped_model",
            save_function=accelerator.save,
            state_dict=state_dict,
            safe_serialization=False
        )
        json.dump({"global_step": global_step}, (save_path / "metadata.json").open("w+"))
        logger.info(f"Saved state to {save_path}")


def log_grad_norm(model, accelerator, global_step):
    for name, param in model.named_parameters():
        if param.grad is not None:
            grads = param.grad.detach().data
            grad_norm = (grads.norm(p=2) / grads.numel()).item()
            accelerator.log({"grad_norm/" + name: grad_norm}, step=global_step)


if __name__ == "__main__":
    main()
