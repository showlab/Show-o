import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

from llava.data.dataset_phi import collate_fn, HybridDataset
from llava.data.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                        AverageMeter, ProgressMeter, dict_to_cuda)
from torch.utils.data.distributed import DistributedSampler
import sys
import argparse
from functools import partial
import transformers
from llava.llava import conversation as conversation_lib


def parse_args(args):
    parser = argparse.ArgumentParser(description="LlmDiff-Phi-3 Model Training")
    parser.add_argument("--local_rank", default=0, type=int, help="node rank")
    parser.add_argument("--version", default="microsoft/phi-1_5")
    parser.add_argument(
        "--precision",
        default="bf16",
        type=str,
        choices=["fp32", "bf16", "fp16", "float32"],
        help="precision for inference",
    )
    parser.add_argument("--image_size", default=256, type=int, help="image size")
    parser.add_argument("--model_max_length", default=1024, type=int)
    parser.add_argument("--lora_r", default=128, type=int)
    parser.add_argument(
        "--vision-tower", default="openai/clip-vit-large-patch14-336", type=str
    )
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)

    parser.add_argument(
        "--dataset", default="vqa", type=str
    )
    parser.add_argument("--sample_rates", default="1", type=str)
    parser.add_argument("--vqa_data", default="llava_instruct_150k", type=str)

    parser.add_argument("--log_base_dir", default="./runs", type=str)
    parser.add_argument("--exp_name", default="llm_diff", type=str)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--steps_per_epoch", default=500, type=int)
    parser.add_argument(
        "--batch_size", default=2, type=int, help="batch size per device per step"
    )
    parser.add_argument(
        "--grad_accumulation_steps",
        default=10,
        type=int,
    )
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--lr", default=0.0003, type=float)
    parser.add_argument("--ce_loss_weight", default=1.0, type=float)
    parser.add_argument("--diff_loss_weight", default=1.0, type=float)
    parser.add_argument("--lora_alpha", default=256, type=int)
    parser.add_argument("--lora_dropout", default=0.05, type=float)
    parser.add_argument("--lora_target_modules", default="q_proj,v_proj", type=str)
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.95, type=float)
    parser.add_argument("--no_eval", action="store_true", default=False)
    parser.add_argument("--resume", default="", type=str)
    parser.add_argument("--print_freq", default=1, type=int)
    parser.add_argument("--render_test", default=500, type=int)
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--use_mm_start_end", action="store_true", default=False)
    parser.add_argument("--auto_resume", action="store_true", default=True)
    parser.add_argument(
        "--conv_type",
        default="phi1.5",
        type=str
    )
    return parser.parse_args(args)


def get_data_loader(version="microsoft/phi-1_5", model_max_length=2048, use_mm_start_end=False,
                    conv_type="phi1.5", vision_tower="openai/clip-vit-large-patch14-336",
                    grad_accumulation_steps=1, steps_per_epoch=500, precision="bf16", image_size=256,
                    dataset="vqa", sample_rates='1', batch_size=2, max_length=77, num_workers=10, world_size=1, local_rank=0):
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        version,
        cache_dir=None,
        model_max_length=model_max_length,
        padding_side="right",
        use_fast=False,
    )
    # world_size = torch.cuda.device_count()
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    # tokenizer.pad_token = tokenizer.unk_token

    if use_mm_start_end:
        tokenizer.add_tokens(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
        )

    assert str(conv_type).lower().startswith("phi")
    conversation_lib.default_conversation = conversation_lib.conv_templates[
        conv_type
    ]

    train_dataset = HybridDataset(
        vision_tower,
        samples_per_epoch=1 * grad_accumulation_steps
                          * steps_per_epoch
                          * world_size,
        precision=precision,
        image_size=image_size,
        dataset=dataset,
        sample_rate=[float(x) for x in sample_rates.split(",")],
        batch_size=batch_size,
    )
    datasampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=local_rank)
    dataloader = torch.utils.data.DataLoader(
        train_dataset,
        # batch_size=64,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=partial(
            collate_fn,
            tokenizer=tokenizer,
            conv_type=conv_type,
            use_mm_start_end=use_mm_start_end,
            local_rank=local_rank,
            max_length=max_length,
        ),
        sampler=datasampler
    )
    return dataloader