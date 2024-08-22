from functools import partial
import torch
from torch.utils.data.distributed import DistributedSampler

from llava.data.dataset_phi_epoch import collate_fn, HybridDataset
from llava.llava import conversation as conversation_lib


def get_data_loader(version="microsoft/phi-1_5",
                    tokenizer=None,
                    model_max_length=2048,
                    use_mm_start_end=False,
                    conv_type="phi1.5",
                    vision_tower="openai/clip-vit-large-patch14-336",
                    grad_accumulation_steps=1,
                    steps_per_epoch=50000,
                    precision="bf16",
                    image_size=256,
                    dataset="vqa",
                    sample_rates='1',
                    batch_size=2,
                    max_length=77,
                    num_workers=10,
                    world_size=1,
                    local_rank=0):

    assert str(conv_type).lower().startswith("phi")
    conversation_lib.default_conversation = conversation_lib.conv_templates[
        conv_type
    ]

    train_dataset = HybridDataset(
        vision_tower,
        precision=precision,
        image_size=image_size,
        dataset=dataset,
        sample_rate=[float(x) for x in sample_rates.split(",")],
    )
    datasampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=local_rank)
    dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=partial(
            collate_fn,
            tokenizer=tokenizer,
            max_length=max_length,
        ),
        sampler=datasampler
    )

    return dataloader
