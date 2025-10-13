# coding=utf-8
# Copyright 2024 HuggingFace, NUS Show Lab.
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

os.environ["TOKENIZERS_PARALLELISM"] = "true"
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
# import wandb  # отключено для работы в России
import mlflow
from mlflow.tracking import MlflowClient
import torch
from torch.optim import AdamW
from tqdm import tqdm
from lightning.pytorch.utilities import CombinedLoader

from transformers import AutoTokenizer
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedType, set_seed

import sys
sys.path.insert(0, '/home/jovyan/vasiliev/notebooks/Show-o')

import mlflow
from mlflow.tracking import MlflowClient

from training.data import Text2ImageDataset
from training.imagenet_dataset import ImageNetDataset
# from parquet import RefinedWebDataset  # закомментировано, используем только MMU

from models import Showo, MAGVITv2, get_mask_chedule
from training.prompting_utils import UniversalPrompting, create_attention_mask_predict_next, \
    create_attention_mask_for_mmu
from models.lr_schedulers import get_scheduler
from models.logger import set_verbosity_info, set_verbosity_error
from moe_utils import patch_and_freeze_moe, log_stats_to_mlflow, LayerOutputRecorder  # MoE support

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from models.llava.llava_data_vq_unified import get_instruct_data_loader

SYSTEM_PROMPT_LEN = 28

from training.utils import get_config, flatten_omega_conf, mask_or_random_replace_tokens, AverageMeter

try:
    import apex

    is_apex_available = True
except ImportError:
    is_apex_available = False

logger = get_logger(__name__, log_level="INFO")


def collect_moe_balance_losses(model):
    total_balance_loss = 0.0
    num_moe_layers = 0
    if hasattr(model, 'module'):
        unwrapped_model = model.module
    else:
        unwrapped_model = model
    
    for layer_idx, layer in enumerate(unwrapped_model.showo.model.layers):
        if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'gate'):
            if hasattr(layer.mlp.gate, 'get_loss') and layer.mlp.gate.has_loss:
                gate_loss = layer.mlp.gate.get_loss(clear=True)  # clear=True чтобы не накапливать
                if gate_loss is not None:
                    total_balance_loss += gate_loss
                    num_moe_layers += 1
                    logger.debug(f"MoE layer {layer_idx}: balance_loss = {gate_loss.item():.6f}")
    return total_balance_loss


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
    accelerator = Accelerator(
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        mixed_precision=config.training.mixed_precision,
        log_with="mlflow" if config.get("mlflow", {}).get("enabled", False) else None,  # MLflow вместо wandb
        project_dir=config.experiment.logging_dir,
        split_batches=True,
    )

    total_batch_size_per_gpu = (config.training.batch_size_t2i
                                + config.training.batch_size_lm
                                + config.training.batch_size_mmu)
    total_batch_size = (
            (config.training.batch_size_t2i + config.training.batch_size_lm + config.training.batch_size_mmu)
            * accelerator.num_processes * config.training.gradient_accumulation_steps
    )

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

    mlflow_client = None
    mlflow_run_id = None
    if accelerator.is_main_process and config.get("mlflow", {}).get("enabled", False):
        mlflow_tracking_uri = config.mlflow.get("tracking_uri", "file:./mlruns")
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow_client = MlflowClient(tracking_uri=mlflow_tracking_uri)
        experiment_name = config.mlflow.get("experiment_name", config.experiment.project)
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
            tags=config.mlflow.get("tags", {})
        )
        mlflow_run_id = run.info.run_id
        
        mlflow_client.log_param(mlflow_run_id, "batch_size_mmu", config.training.batch_size_mmu)
        mlflow_client.log_param(mlflow_run_id, "batch_size_t2i", config.training.batch_size_t2i)
        mlflow_client.log_param(mlflow_run_id, "batch_size_lm", config.training.batch_size_lm)
        mlflow_client.log_param(mlflow_run_id, "learning_rate", config.optimizer.params.learning_rate)
        mlflow_client.log_param(mlflow_run_id, "num_experts", config.moe.get("num_experts", 4))
        mlflow_client.log_param(mlflow_run_id, "top_k", config.moe.get("top_k", 2))
        mlflow_client.log_param(mlflow_run_id, "max_train_steps", config.training.max_train_steps)
        mlflow_client.log_param(mlflow_run_id, "gradient_accumulation_steps", config.training.gradient_accumulation_steps)
        
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

    tokenizer = AutoTokenizer.from_pretrained(config.model.showo.llm_model_path, padding_side="left")

    # unified prompting for show-o
    uni_prompting = UniversalPrompting(tokenizer, max_text_len=config.dataset.preprocessing.max_seq_length,
                                       special_tokens=(
                                           "<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>", "<|t2i|>",
                                           "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>"
                                       ),
                                       ignore_id=-100, cond_dropout_prob=config.training.cond_dropout_prob)

    print('special tokens : \n', uni_prompting.sptids_dict)

    # VQ model for processing image into discrete tokens
    vq_model = get_vq_model_class(config.model.vq_model.type)
    if config.model.vq_model.get("pretrained_model_path", None):
        vq_model = vq_model().to(accelerator.device)
        state_dict = torch.load(config.model.vq_model.pretrained_model_path)['model']
        vq_model.load_state_dict(state_dict)
    else:
        vq_model = vq_model.from_pretrained(config.model.vq_model.vq_model_name).to(accelerator.device)
    vq_model.eval()
    vq_model.requires_grad_(False)

    # Initialize Show-o model
    if config.model.showo.load_from_showo:
        model = Showo.from_pretrained(config.model.showo.pretrained_model_path).to(accelerator.device)
        if config.model.showo.vocab_size != model.vocab_size:
            model.showo.resize_token_embeddings(config.model.showo.vocab_size)
            model.config.codebook_size = config.model.showo.codebook_size
            model.config.vocab_size = config.model.showo.vocab_size
            model.vocab_size = config.model.showo.vocab_size
            model.output_size = config.model.showo.vocab_size
            model.config.mask_token_id = config.model.showo.vocab_size - 1
            model.mask_token_id = config.model.showo.vocab_size - 1
    else:
        model = Showo(**config.model.showo).to(accelerator.device)
    
    if config.get("moe", None) and config.moe.get("enabled", False):
        special_tokens = {
            'soi_id': uni_prompting.sptids_dict['<|soi|>'].item() if '<|soi|>' in uni_prompting.sptids_dict else None,
            'eoi_id': uni_prompting.sptids_dict['<|eoi|>'].item() if '<|eoi|>' in uni_prompting.sptids_dict else None,
            'sov_id': uni_prompting.sptids_dict['<|sov|>'].item() if '<|sov|>' in uni_prompting.sptids_dict else None,
            'eov_id': uni_prompting.sptids_dict['<|eov|>'].item() if '<|eov|>' in uni_prompting.sptids_dict else None,
        }
        
        model = patch_and_freeze_moe(
            model, 
            num_experts=config.moe.get("num_experts", 4), 
            top_k=config.moe.get("top_k", 2),
            mlflow_client=mlflow_client,
            mlflow_run_id=mlflow_run_id,
            special_tokens=special_tokens
        )
    
    mask_id = model.mask_token_id

    ##################################
    #   Optimizer and LR scheduler   #
    #################################
    optimizer_config = config.optimizer.params

    # no decay on bias and layernorm and embedding
    no_decay = ["bias", "layer_norm.weight", "mlm_ln.weight", "embeddings.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if
                       p.requires_grad and not any(nd in n for nd in no_decay)],
            "weight_decay": optimizer_config.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if
                       p.requires_grad and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer_type = config.optimizer.name
    if optimizer_type == "adamw":
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=optimizer_config.learning_rate,
            betas=(optimizer_config.beta1, optimizer_config.beta2),
            weight_decay=optimizer_config.weight_decay,
            eps=optimizer_config.epsilon,
        )
    else:
        raise ValueError(f"Optimizer {optimizer_type} not supported")

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

    ##################################
    #         DATALOADER             #
    #################################
    logger.info("Creating dataloaders and lr_scheduler")

    total_batch_size_t2i_without_accum = config.training.batch_size_t2i * accelerator.num_processes
    total_batch_size_t2i = (
            config.training.batch_size_t2i * accelerator.num_processes * config.training.gradient_accumulation_steps
    )

    # DataLoaders creation:
    # We use webdataset for data loading. The dataloaders are created with sampling with replacement.
    # We don't do dataset resuming here, instead we resample the shards and buffer each time. The sampling is stochastic.
    # This means that the dataloading is not deterministic, but it's fast and efficient.
    preproc_config = config.dataset.preprocessing
    dataset_config = config.dataset.params

    # Data for generation
    if config.dataset.gen_type == "t2i":
        dataset = Text2ImageDataset(
            train_shards_path_or_url=dataset_config.train_t2i_shards_path_or_url,
            tokenizer=None,  # we want to get raw texts
            max_seq_length=preproc_config.max_seq_length,
            num_train_examples=config.experiment.max_train_examples_t2i,
            per_gpu_batch_size=config.training.batch_size_t2i,
            global_batch_size=total_batch_size_t2i_without_accum,
            num_workers=dataset_config.num_workers,
            resolution=preproc_config.resolution,
            shuffle_buffer_size=dataset_config.shuffle_buffer_size,
            pin_memory=dataset_config.pin_memory,
            persistent_workers=dataset_config.persistent_workers,
            external_caption_path=dataset_config.external_caption_path,
            external_journeydb_caption_path=dataset_config.external_journeydb_caption_path,
            external_laion12m_caption_path=dataset_config.external_laion12m_caption_path,
            external_cc12m_caption_path=dataset_config.external_cc12m_caption_path,
        )
        train_dataloader_t2i = dataset.train_dataloader
        num_update_steps_per_epoch = math.ceil(
            train_dataloader_t2i.num_batches / config.training.gradient_accumulation_steps)
        num_train_epochs = math.ceil(config.training.max_train_steps / num_update_steps_per_epoch)

    elif config.dataset.gen_type == "t2i_parquet":
        # this part relies on the internal packages, which will not be released
        num_update_steps_per_epoch = math.ceil(config.experiment.max_train_examples_t2i / total_batch_size_t2i)
        num_train_epochs = math.ceil(config.training.max_train_steps / num_update_steps_per_epoch)

        train_dataloader_t2i = create_imagetext_dataloader(
            train_shards_path_or_url=dataset_config.train_t2i_shards_path_or_url,
            batch_size=config.training.batch_size_t2i,
            image_size=preproc_config.resolution,
            num_workers=dataset_config.num_workers,
            num_readers=32,
            predefined_steps=num_update_steps_per_epoch,
            drop_last=True,
            shuffle=True,
            shuffle_buffer_size=dataset_config.shuffle_buffer_size
        )

    elif config.dataset.gen_type == "imagenet1k":
        dataset_imagenet = ImageNetDataset(
            dataset_config.train_t2i_shards_path_or_url,
            image_size=preproc_config.resolution,
        )

        print('process index : ',
              accelerator.process_index, ', ', accelerator.num_processes,
              "Length: ", len(dataset_imagenet))

        if accelerator.num_processes > 1:
            sampler = DistributedSampler(dataset_imagenet,
                                         num_replicas=accelerator.num_processes,
                                         rank=accelerator.process_index,
                                         shuffle=True,
                                         )
            shuffle = False
        else:
            sampler = None
            shuffle = True

        train_dataloader_t2i = DataLoader(dataset_imagenet, batch_size=config.training.batch_size_t2i,
                                          sampler=sampler, collate_fn=dataset_imagenet.collate_fn,
                                          shuffle=shuffle, num_workers=dataset_config.num_workers)
        num_update_steps_per_epoch = math.ceil(len(dataset_imagenet) / total_batch_size_t2i)
        num_train_epochs = math.ceil(config.training.max_train_steps / num_update_steps_per_epoch)

    else:
        raise ValueError(f"Unsupported dataset type {config.dataset.type}")

    total_batch_size_mmu_without_accum = config.training.batch_size_mmu * accelerator.num_processes
    # Data for image captioning
    if config.dataset.und_type == "captioning":
        dataset_mmu = Text2ImageDataset(
            train_shards_path_or_url=dataset_config.train_mmu_shards_path_or_url,
            tokenizer=None,  # we want to get raw texts
            max_seq_length=preproc_config.max_seq_length,
            num_train_examples=config.experiment.max_train_examples_mmu,
            per_gpu_batch_size=config.training.batch_size_mmu,
            global_batch_size=total_batch_size_mmu_without_accum,
            num_workers=dataset_config.num_workers,
            resolution=preproc_config.resolution,
            shuffle_buffer_size=dataset_config.shuffle_buffer_size,
            pin_memory=dataset_config.pin_memory,
            persistent_workers=dataset_config.persistent_workers,
            external_caption_path=dataset_config.external_caption_path,
            external_journeydb_caption_path=dataset_config.external_journeydb_caption_path,
            external_laion12m_caption_path=dataset_config.external_laion12m_caption_path,
            external_cc12m_caption_path=dataset_config.external_cc12m_caption_path,
            is_captioning=True,
            add_caption_prompt=dataset_config.add_caption_prompt,
        )
        train_dataloader_mmu = dataset_mmu.train_dataloader

    elif config.dataset.und_type == "captioning_parquet":
        train_dataloader_mmu = create_imagetext_dataloader(
            train_shards_path_or_url=dataset_config.train_mmu_shards_path_or_url,
            batch_size=config.training.batch_size_mmu,
            image_size=preproc_config.resolution,
            num_workers=dataset_config.num_workers,
            num_readers=32,
            predefined_steps=num_update_steps_per_epoch,
            drop_last=True,
            shuffle=True,
            shuffle_buffer_size=dataset_config.shuffle_buffer_size,
            is_captioning=True
        )

    elif config.dataset.und_type == "llava_pretrain":
        train_dataloader_mmu = get_instruct_data_loader(
            tokenizer,
            batch_size=config.training.batch_size_mmu,
            num_workers=dataset_config.num_workers,
            world_size=accelerator.num_processes,
            local_rank=accelerator.process_index,
            max_length=preproc_config.max_seq_length if config.dataset.add_system_prompt else preproc_config.max_seq_length + SYSTEM_PROMPT_LEN,
            phase="pretrain"
        )

    elif config.dataset.und_type == "llava_tuning":
        train_dataloader_mmu = get_instruct_data_loader(
            tokenizer,
            batch_size=config.training.batch_size_mmu,
            num_workers=dataset_config.num_workers,
            world_size=accelerator.num_processes,
            local_rank=accelerator.process_index,
            max_length=preproc_config.max_seq_length if config.dataset.add_system_prompt else preproc_config.max_seq_length + SYSTEM_PROMPT_LEN,
            phase="tuning"
        )

    else:
        raise NotImplementedError(f"Unsupported dataset type {config.dataset.und_type}")

    class DummyLMDataset:
        def __len__(self):
            return 1000000
        def __getitem__(self, idx):
            return {'input_ids': [0]}  # dummy
        def collate_fn(self, batch):
            return {'input_ids': ['dummy text'] * len(batch)}
    
    train_dataloader_lm = torch.utils.data.DataLoader(
        DummyLMDataset(), 
        batch_size=config.training.batch_size_lm,
        sampler=None, 
        collate_fn=DummyLMDataset().collate_fn
    )

    iterables = {
        "t2i_flow": train_dataloader_t2i,
        "lm_flow": train_dataloader_lm,
        "mmu_flow": train_dataloader_mmu,
    }

    combined_dataloader = CombinedLoader(iterables, mode=config.dataset.combined_loader_mode)

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
            model.load_state_dict(state_dict, strict=True)
            del state_dict

    ##################################
    #       Prepare accelerator     #
    #################################
    logger.info("Preparing model, optimizer and dataloaders")
    model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)

    vq_model.to(device=accelerator.device)

    if hasattr(model, 'module'):
        mask_dtype = model.module.showo.model.embed_tokens.weight.dtype
    else:
        mask_dtype = accelerator.unwrap_model(model).showo.model.embed_tokens.weight.dtype

    ##################################
    #             Training          #
    #################################
    logger.info("***** Running training *****")
    logger.info(f"  Num training steps = {config.training.max_train_steps}")
    logger.info(f"  Instantaneous batch size per device = {total_batch_size_per_gpu}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.training.gradient_accumulation_steps}")

    @torch.no_grad()
    def prepare_inputs_and_labels(
            pixel_values_or_image_ids: Union[torch.FloatTensor, torch.LongTensor],
            texts: Union[str, str],
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
        input_ids, masks, labels = uni_prompting((texts, input_ids, labels), 't2i')

        return input_ids, labels, mask_prob, image_tokens

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
                colour="green"
            )
        
        for batch, batch_idx, dataloader_idx in combined_dataloader:
            # for loss calculation
            batch_size_t2i = batch["t2i_flow"]["images"].shape[0]
            batch_size_lm = len(batch["lm_flow"]["input_ids"])
            batch_size_mmu = batch["mmu_flow"]["images"].shape[0]

            # *-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*
            # Build formatted sequences for class-conditional/text-to-image generation
            # *-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*
            pixel_values, texts = batch["t2i_flow"]["images"], batch["t2i_flow"]["input_ids"]
            pixel_values = pixel_values.to(accelerator.device, non_blocking=True)
            data_time_m.update(time.time() - end)

            # Encode images to image tokens, mask them and create input and labels
            (
                input_ids,
                labels,
                mask_prob,
                image_tokens_ori
            ) = prepare_inputs_and_labels(pixel_values, texts, config.training.min_masking_rate)
            attention_mask = create_attention_mask_predict_next(input_ids,
                                                                pad_id=int(uni_prompting.sptids_dict['<|pad|>']),
                                                                soi_id=int(uni_prompting.sptids_dict['<|soi|>']),
                                                                eoi_id=int(uni_prompting.sptids_dict['<|eoi|>']),
                                                                rm_pad_in_image=True,
                                                                return_inverse_mask=True)
            attention_mask = attention_mask.to(mask_dtype)

            # *-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*
            # Build formatted sequences for language modeling
            # *-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*
            texts_lm = batch["lm_flow"]["input_ids"]
            input_ids_lm, _, labels_lm = uni_prompting((texts_lm, input_ids.shape[-1]), 'lm')
            attention_mask_lm = create_attention_mask_predict_next(input_ids_lm.to(input_ids.device),
                                                                   pad_id=int(uni_prompting.sptids_dict['<|pad|>']),
                                                                   soi_id=int(uni_prompting.sptids_dict['<|soi|>']),
                                                                   eoi_id=int(uni_prompting.sptids_dict['<|eoi|>']))
            attention_mask_lm = attention_mask_lm.to(mask_dtype)
            attention_mask = torch.cat([attention_mask, attention_mask_lm], dim=0)
            input_ids = torch.cat((input_ids, input_ids_lm.to(input_ids.device)), dim=0)
            labels = torch.cat((labels, labels_lm.to(input_ids.device)), dim=0)

            # *-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*
            # Build formatted sequences for captioning/multimodal understanding
            # *-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*
            if "llava" in config.dataset.und_type:
                pixel_values_mmu, input_ids_mmu, labels_mmu = (batch["mmu_flow"]["images"],
                                                               batch["mmu_flow"]["input_ids"],
                                                               batch["mmu_flow"]["labels"])
                pixel_values_mmu = pixel_values_mmu.to(accelerator.device, non_blocking=True)
                input_ids_mmu = input_ids_mmu.to(accelerator.device, non_blocking=True)
                image_tokens_mmu = vq_model.get_code(pixel_values_mmu)
                image_tokens_mmu = image_tokens_mmu + len(uni_prompting.text_tokenizer)

                input_ids_mmu = torch.cat([
                    (torch.ones(input_ids_mmu.shape[0], 1) * uni_prompting.sptids_dict['<|mmu|>']).to(
                        accelerator.device),
                    (torch.ones(input_ids_mmu.shape[0], 1) * uni_prompting.sptids_dict['<|soi|>']).to(
                        accelerator.device),
                    image_tokens_mmu,
                    (torch.ones(input_ids_mmu.shape[0], 1) * uni_prompting.sptids_dict['<|eoi|>']).to(
                        accelerator.device),
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

                pixel_values_mmu, texts_mmu = batch["mmu_flow"]["images"], batch["mmu_flow"]["input_ids"]
                pixel_values_mmu = pixel_values_mmu.to(accelerator.device, non_blocking=True)
                image_tokens_mmu = vq_model.get_code(pixel_values_mmu)
                image_tokens_mmu = image_tokens_mmu + len(uni_prompting.text_tokenizer)
                input_ids_mmu, _, labels_mmu = uni_prompting((image_tokens_mmu, texts_mmu), 'mmu')
                input_ids_mmu = input_ids_mmu.to(accelerator.device, non_blocking=True)

            attention_mask_mmu = create_attention_mask_for_mmu(input_ids_mmu.to(input_ids.device),
                                                               eoi_id=int(uni_prompting.sptids_dict['<|eoi|>']))
            attention_mask_mmu = attention_mask_mmu.to(mask_dtype)
            attention_mask = torch.cat([attention_mask, attention_mask_mmu], dim=0)
            input_ids = torch.cat((input_ids, input_ids_mmu.to(input_ids.device)), dim=0)
            labels = torch.cat((labels, labels_mmu.to(input_ids.device)), dim=0)

            if global_step == 0 and epoch == 0:
                logger.info("Input ids: {}".format(input_ids))
                logger.info("Labels: {}".format(labels))

            with accelerator.accumulate(model):
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
                )

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss_t2i = accelerator.gather(loss_t2i.repeat(config.training.batch_size_t2i)).mean()
                avg_loss_lm = accelerator.gather(loss_lm.repeat(config.training.batch_size_lm)).mean()
                avg_loss_mmu = accelerator.gather(loss_mmu.repeat(config.training.batch_size_mmu)).mean()
                
                # Собираем balance loss от MoE гейтов
                balance_loss = collect_moe_balance_losses(model)
                balance_coeff = config.training.get("balance_coeff", 0.01)  # Коэффициент для balance loss
                
                # Главный лосс с включением balance loss
                loss = config.training.t2i_coeff * loss_t2i + \
                       config.training.lm_coeff * loss_lm + \
                       config.training.mmu_coeff * loss_mmu + \
                       balance_coeff * balance_loss

                avg_masking_rate = accelerator.gather(mask_prob.repeat(config.training.batch_size_t2i)).mean()

                accelerator.backward(loss)
                
                # Очищаем кэш GPU для освобождения памяти
                if torch.cuda.is_available() and (global_step + 1) % config.training.gradient_accumulation_steps == 0:
                    torch.cuda.empty_cache()

                if config.training.max_grad_norm is not None and accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()

                # log gradient norm before zeroing it
                if (
                        accelerator.sync_gradients
                        and (global_step + 1) % config.experiment.log_grad_norm_every == 0
                        and accelerator.is_main_process
                ):
                    log_grad_norm(model, accelerator, global_step + 1)

                optimizer.zero_grad(set_to_none=True)
                if accelerator.is_main_process and pbar is not None:
                    pbar.update(1)
                    pbar.set_postfix({
                        'loss_mmu': f'{avg_loss_mmu.item():.4f}',
                        'loss_t2i': f'{avg_loss_t2i.item():.4f}',
                        'lr': f'{lr_scheduler.get_last_lr()[0]:.2e}',
                    })

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:

                batch_time_m.update(time.time() - end)
                end = time.time()

                # Log metrics
                if (global_step + 1) % config.experiment.log_every == 0:
                    samples_per_second_per_gpu = (
                            config.training.gradient_accumulation_steps * total_batch_size_per_gpu / batch_time_m.val
                    )
                    logs = {
                        "step_loss_t2i": avg_loss_t2i.item(),
                        "step_loss_mmu": avg_loss_mmu.item(),
                        "step_loss_lm": avg_loss_lm.item(),
                        "step_loss_balance": balance_loss.item(),
                        "balance_coeff": balance_coeff,
                        "lr": lr_scheduler.get_last_lr()[0],
                        "avg_masking_rate": avg_masking_rate.item(),
                        "samples/sec/gpu": samples_per_second_per_gpu,
                        "data_time": data_time_m.val,
                        "batch_time": batch_time_m.val,
                    }
                    if mlflow_client is not None and mlflow_run_id is not None:
                        for metric_name, metric_value in logs.items():
                            mlflow_client.log_metric(mlflow_run_id, metric_name, metric_value, step=global_step + 1)
                        unwrapped_model = accelerator.unwrap_model(model)
                        for layer in unwrapped_model.showo.model.layers:
                            if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'set_global_step'):
                                layer.mlp.set_global_step(global_step + 1)

                    logger.info(
                        f"Step: {global_step + 1} "
                        f"Loss_t2i: {avg_loss_t2i.item():0.4f} "
                        f"Loss_mmu: {avg_loss_mmu.item():0.4f} "
                        f"Loss_lm: {avg_loss_lm.item():0.4f} "
                        f"Loss_balance: {balance_loss.item():0.6f} "
                        f"Data (t): {data_time_m.val:0.4f}, {samples_per_second_per_gpu:0.2f}/s/gpu "
                        f"Batch (t): {batch_time_m.val:0.4f} "
                        f"LR: {lr_scheduler.get_last_lr()[0]:0.6f}"
                    )

                    # resetting batch / data time meters per log window
                    batch_time_m.reset()
                    data_time_m.reset()

                # Сбор статистик активаций для MoE (каждые 500 шагов)
                if (global_step + 1) % 500 == 0 and config.get("moe", {}).get("enabled", False) and accelerator.is_main_process:
                    try:
                        # Создаем рекордер для сбора статистик
                        recorder = LayerOutputRecorder(device=accelerator.device)
                        
                        # Регистрируем хуки на MoE слои
                        unwrapped_model = accelerator.unwrap_model(model)
                        moe_modules = []
                        for layer_idx, layer in enumerate(unwrapped_model.showo.model.layers):
                            if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'experts'):
                                moe_modules.append((f"layer_{layer_idx}", layer.mlp))
                        
                        if moe_modules:
                            recorder.register_hooks(moe_modules)
                            
                            # Выполняем один forward pass для сбора статистик
                            with torch.no_grad():
                                # Используем текущий батч для сбора статистик
                                _ = model(
                                    input_ids=input_ids,
                                    input_embeddings=None,
                                    attention_mask=attention_mask,
                                    labels=labels,
                                    label_smoothing=config.training.label_smoothing,
                                    batch_size_t2i=batch_size_t2i,
                                    batch_size_lm=batch_size_lm,
                                    batch_size_mmu=batch_size_mmu,
                                    max_seq_length=config.dataset.preprocessing.max_seq_length,
                                )
                            
                            # Логируем статистики в MLflow
                            if mlflow_client is not None and mlflow_run_id is not None:
                                log_stats_to_mlflow(recorder, mlflow_client, mlflow_run_id, global_step + 1, "moe_activations")
                            
                            # Очищаем рекордер
                            recorder.remove_hooks()
                            recorder.clear()
                            del recorder
                            
                    except Exception as e:
                        logger.warning(f"Ошибка сбора статистик активаций: {e}")

                # Save model checkpoint
                if (global_step + 1) % config.experiment.save_every == 0:
                    save_checkpoint(model, config, accelerator, global_step + 1)

                if (global_step + 1) % config.experiment.generate_every == 0 and accelerator.is_main_process:
                    generate_images(
                        model,
                        vq_model,
                        uni_prompting,
                        accelerator,
                        config,
                        global_step + 1,
                        mask_schedule=mask_schedule,
                    )

                    visualize_predictions(
                        model,
                        vq_model,
                        uni_prompting,
                        config,
                        global_step + 1,
                        input_ids,
                        image_tokens_ori,
                        batch["t2i_flow"]["images"],
                        texts,
                        logits,
                    )

                global_step += 1

            if global_step >= config.training.max_train_steps:
                break
        if accelerator.is_main_process:
            pbar.close()

    accelerator.wait_for_everyone()
    save_checkpoint(model, config, accelerator, global_step)
    
    if mlflow_client is not None and mlflow_run_id is not None:
        mlflow_client.set_terminated(mlflow_run_id, status="FINISHED")
        logger.info("✅ MLflow run завершен")

    if accelerator.is_main_process:
        model = accelerator.unwrap_model(model)
        model.save_pretrained(config.experiment.output_dir, safe_serialization=False)

    accelerator.end_training()


@torch.no_grad()
def visualize_predictions(
        model,
        vq_model,
        uni_prompting,
        config,
        global_step,
        input_ids,
        image_tokens_ori,
        ori_images,
        texts,
        logits,
):
    logger.info("Visualizing predictions...")
    model.eval()

    recons_images = vq_model.decode_code(image_tokens_ori - len(uni_prompting.text_tokenizer))
    recons_images = torch.clamp((recons_images + 1.0) / 2.0, min=0.0, max=1.0)
    recons_images *= 255.0
    recons_images = recons_images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)

    images = torch.clamp((ori_images + 1.0) / 2.0, min=0.0, max=1.0)
    images *= 255.0
    images = images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)

    predictions = logits[:config.training.batch_size_t2i, -(config.model.showo.num_vq_tokens + 1):-1:,
                  config.model.showo.llm_vocab_size + config.model.showo.num_new_special_tokens:-1]
    predictions = predictions.argmax(axis=-1)

    mask_token_id = config.model.showo.vocab_size - 1 - len(uni_prompting.text_tokenizer)
    input_ids = input_ids[:config.training.batch_size_t2i, -(config.model.showo.num_vq_tokens + 1):-1:] - len(
        uni_prompting.text_tokenizer)
    mask_ratio = list((torch.where(input_ids == mask_token_id, 1, 0).sum(
        dim=-1) / config.model.showo.num_vq_tokens).cpu().numpy())
    predicted_images = torch.where(input_ids == mask_token_id, predictions, input_ids)

    predicted_images = vq_model.decode_code(predicted_images)
    predicted_images = torch.clamp((predicted_images + 1.0) / 2.0, min=0.0, max=1.0)
    predicted_images *= 255.0
    predicted_images = predicted_images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
    predicted_images = np.concatenate((images, recons_images, predicted_images), 2)
    pil_images = [Image.fromarray(image) for image in predicted_images]

    # wandb_images = [wandb.Image(image, caption=f'mask ratio: {r:0.2f} \n caption: {texts[i]}') for i, (image, r) in
    #                 enumerate(zip(pil_images, mask_ratio))]
    # wandb.log({"Original images v.s. Reconstructed images v.s. Predicted images": wandb_images}, step=global_step)
    logger.info(f"Визуализация предсказаний завершена (wandb отключен)")

    model.train()


@torch.no_grad()
def generate_images(
        model,
        vq_model,
        uni_prompting,
        accelerator,
        config,
        global_step,
        mask_schedule,
):
    logger.info("Generating images...")
    model.eval()

    # read validation prompts from file
    with open(config.dataset.params.validation_prompts_file, "r") as f:
        validation_prompts = f.read().splitlines()

    if hasattr(model, 'module'):
        mask_dtype = model.module.showo.model.embed_tokens.weight.dtype
    else:
        mask_dtype = accelerator.unwrap_model(model).showo.model.embed_tokens.weight.dtype

    mask_token_id = config.model.showo.vocab_size - 1
    image_tokens = torch.ones((len(validation_prompts), config.model.showo.num_vq_tokens), dtype=torch.long,
                              device=accelerator.device) * mask_token_id
    input_ids, _ = uni_prompting((validation_prompts, image_tokens), 't2i_gen')
    if config.training.guidance_scale > 0:
        uncond_input_ids, _ = uni_prompting(([''] * len(validation_prompts), image_tokens), 't2i_gen')
        attention_mask = create_attention_mask_predict_next(torch.cat([input_ids, uncond_input_ids], dim=0),
                                                            pad_id=int(uni_prompting.sptids_dict['<|pad|>']),
                                                            soi_id=int(uni_prompting.sptids_dict['<|soi|>']),
                                                            eoi_id=int(uni_prompting.sptids_dict['<|eoi|>']),
                                                            rm_pad_in_image=True).to(mask_dtype)
    else:
        attention_mask = create_attention_mask_predict_next(input_ids,
                                                            pad_id=int(uni_prompting.sptids_dict['<|pad|>']),
                                                            soi_id=int(uni_prompting.sptids_dict['<|soi|>']),
                                                            eoi_id=int(uni_prompting.sptids_dict['<|eoi|>']),
                                                            rm_pad_in_image=True).to(mask_dtype)
        uncond_input_ids = None

    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    else:
        weight_dtype = torch.float32

    with torch.autocast("cuda", dtype=weight_dtype, enabled=accelerator.mixed_precision != "no"):
        # Generate images
        gen_token_ids = accelerator.unwrap_model(model).t2i_generate(
            input_ids=input_ids,
            uncond_input_ids=uncond_input_ids,
            attention_mask=attention_mask,
            guidance_scale=config.training.guidance_scale,
            temperature=config.training.get("generation_temperature", 1.0),
            timesteps=config.training.generation_timesteps,
            noise_schedule=mask_schedule,
            noise_type=config.training.get("noise_type", "mask"),
            predict_all_tokens=config.training.get("predict_all_tokens", False),
            seq_len=config.model.showo.num_vq_tokens,
            uni_prompting=uni_prompting,
            config=config,
        )
    # In the beginning of training, the model is not fully trained and the generated token ids can be out of range
    # so we clamp them to the correct range.
    gen_token_ids = torch.clamp(gen_token_ids, max=accelerator.unwrap_model(model).config.codebook_size - 1, min=0)
    images = vq_model.decode_code(gen_token_ids)

    model.train()

    if config.training.get("pre_encode", False):
        del vq_model

    # Convert to PIL images
    images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)
    images *= 255.0
    images = images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
    pil_images = [Image.fromarray(image) for image in images]

    # wandb_images = [wandb.Image(image, caption=validation_prompts[i]) for i, image in enumerate(pil_images)]
    # wandb.log({"Generated images": wandb_images}, step=global_step)
    logger.info(f"Генерация изображений завершена (wandb отключен)")


def save_checkpoint(model, config, accelerator, global_step):
    output_dir = config.experiment.output_dir
    checkpoints_total_limit = config.experiment.get("checkpoints_total_limit", None)
    save_moe_only = config.get("moe", {}).get("save_only_moe_weights", True)  # Новая опция

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


def log_grad_norm(model, accelerator, global_step):
    if hasattr(accelerator, 'trackers') and len(accelerator.trackers) > 0:
        for name, param in model.named_parameters():
            if param.grad is not None:
                grads = param.grad.detach().data
                grad_norm = (grads.norm(p=2) / grads.numel()).item()
                accelerator.log({"grad_norm/" + name: grad_norm}, step=global_step)


if __name__ == "__main__":
    main()
