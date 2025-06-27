# coding=utf-8
# Copyright 2025 NUS Show Lab.
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

import argparse
import collections
import json
import random
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from datasets.utils import (
    image_transform, remove_prefix, format_sequence_und, format_sequence_gen_qwen2_5
)


class ImageTextDataset(Dataset):
    """Dataset for loading image-text pairs."""

    def __init__(self,
                 anno_path: str,
                 text_tokenizer: Any,
                 max_seq_len: int = 318 + 576 + 2,
                 image_size: int = 768,
                 latent_height: int = 24,
                 latent_width: int = 24,
                 num_image_tokens: int = 576,
                 is_captioning: bool = False,
                 aes_score: Optional[float] = None,
                 cond_dropout_prob: float = 0.1,
                 min_res: List[int] = [512, 512],
                 random_und_or_gen: float = 0.0,
                 showo_token_ids: Optional[Dict[str, int]] = None,
                 system: Tuple[str, str, str] = ("", "", "")):
        super().__init__()

        self.anno = []
        with open(anno_path) as f:
            for i, line in enumerate(f):
                try:
                    self.anno.append(json.loads(line))
                except json.decoder.JSONDecodeError as e:
                    print(f"Error decoding the following jsonl line ({i}):\n{line.rstrip()}")
                    raise e

        self.text_tokenizer = text_tokenizer
        self.pad_id = self.text_tokenizer.pad_token_id
        self.bos_id = showo_token_ids['bos_id']
        self.eos_id = showo_token_ids['eos_id']
        self.boi_id = showo_token_ids['boi_id']
        self.eoi_id = showo_token_ids['eoi_id']
        self.img_pad_id = showo_token_ids['img_pad_id']
        self.max_seq_len = max_seq_len
        self.image_size = image_size
        self.num_image_tokens = num_image_tokens
        self.h = latent_height
        self.w = latent_width
        self.cond_dropout_prob = cond_dropout_prob
        self.is_captioning = is_captioning
        self.data_type = "mmu" if self.is_captioning else "t2i"
        self.image_transform = image_transform
        self.aes_score = aes_score
        self.clip_image_size = 384  # Hard code resolution
        self.clip_mean = (0.5, 0.5, 0.5)
        self.clip_std = (0.5, 0.5, 0.5)
        self.min_res = min_res  # h, w
        self.random_und_or_gen = random_und_or_gen
        self.system_tokens = self.text_tokenizer(system, add_special_tokens=False).input_ids
        self.system_token_len = sum(len(tokens) for tokens in self.system_tokens)
        if len(self.system_tokens[0]) == 0:
            # 4 for bos, eos, boi, and eoi tokens
            self.max_text_len = max_seq_len - num_image_tokens - 4
        else:
            # 4 for bos, eos, boi, and eoi tokens
            # 1 for eos after text token (a bit tricky)
            # see more details in def format_sequence_gen_qwen2_5(...)
            self.max_text_len = max_seq_len - num_image_tokens - 4 - self.system_token_len - 1

    def __len__(self) -> int:
        return len(self.anno)

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        try:
            data = self.anno[idx]

            img = Image.open(data['path'])
            width, height = img.size

            # Resolution filtering
            if height < self.min_res[0] or width < self.min_res[1]:
                return self.__getitem__(idx + 1)

            # Enable random understanding or generation data
            if self.random_und_or_gen > 0:
                self.is_captioning = random.random() < self.random_und_or_gen
                self.data_type = "mmu" if self.is_captioning else "t2i"

            mode = img.mode
            if mode == 'RGBA':
                img = img.convert('RGBA')
                img = np.array(img)[:, :, :3]
            elif img.mode == "P" or "transparency" in img.info:
                img = img.convert('RGBA')
                img = np.array(img)[:, :, :3]
            elif mode == 'RGB':
                img = img.convert('RGB')
                img = np.array(img)
            elif mode == 'L':
                img = np.array(img.convert('L'))
                img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
            else:
                return self.__getitem__(idx + 1)

            img = Image.fromarray(img)
            img_clip = self.image_transform(img, resolution=self.clip_image_size, mean=self.clip_mean,
                                            std=self.clip_std)
            img = self.image_transform(img, resolution=self.image_size)

            text = data['prompt'].replace('\n', '')
            if not self.is_captioning:
                text = remove_prefix(text)

            # Tokenization with truncation
            if random.random() < self.cond_dropout_prob and not self.is_captioning:
                text_tokens = self.text_tokenizer('',
                                                  add_special_tokens=False,
                                                  truncation=True,
                                                  max_length=self.max_text_len
                                                  ).input_ids
            else:
                text_tokens = self.text_tokenizer(text,
                                                  add_special_tokens=False,
                                                  truncation=True,
                                                  max_length=self.max_text_len
                                                  ).input_ids

            if self.is_captioning:
                # System prompt is not supported in this dataset
                text_tokens, text_labels, modality_positions, text_mask, image_mask = \
                    format_sequence_und(text_tokens, self.bos_id, self.eos_id,
                                        self.boi_id, self.eoi_id, self.pad_id, self.img_pad_id,
                                        self.num_image_tokens, self.max_seq_len)
            else:
                text_tokens, text_labels, modality_positions, text_mask, image_mask = \
                    format_sequence_gen_qwen2_5(text_tokens, self.system_tokens, self.bos_id, self.eos_id,
                                                self.boi_id, self.eoi_id, self.pad_id, self.img_pad_id,
                                                self.num_image_tokens, self.max_seq_len, self.system_token_len)

            sample = {
                'text_tokens': text_tokens, 'text_labels': text_labels, 'images': img,
                'modality_positions': modality_positions, 'text_masks': text_mask,
                'image_masks': image_mask, 'images_clip': img_clip, 'texts': text, 'data_type': self.data_type
            }

            return sample

        except Exception as e:
            print(e)
            return self.__getitem__(idx + 1)

    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batched = collections.defaultdict(list)
        for data in batch:
            for k, v in data.items():
                batched[k].append(v)
        for k, v in batched.items():
            if k in ('texts', 'data_type'):
                batched[k] = [_[0] for _ in v]
            else:
                batched[k] = torch.stack(v, dim=0)
        return batched


def create_imagetext_dataloader(train_shards_path_or_url: str,
                                batch_size: int,
                                text_tokenizer: Any,
                                accelerator: Optional[Any] = None,
                                image_size: int = 384,
                                latent_width: int = 24,
                                latent_height: int = 24,
                                cond_dropout_prob: float = 0.1,
                                max_seq_len: int = 768,
                                num_image_tokens: int = 576,
                                num_workers: int = 64,
                                is_captioning: bool = False,
                                min_res: List[int] = [512, 512],
                                shuffle: bool = True,
                                random_und_or_gen: float = 0.0,
                                drop_last: bool = True,
                                showo_token_ids: Optional[Dict[str, int]] = None,
                                system: Tuple[str, str, str] = ("", "", "")) -> DataLoader:

    dataset = ImageTextDataset(train_shards_path_or_url,
                               text_tokenizer=text_tokenizer,
                               max_seq_len=max_seq_len,
                               image_size=image_size,
                               latent_height=latent_height,
                               latent_width=latent_width,
                               cond_dropout_prob=cond_dropout_prob,
                               num_image_tokens=num_image_tokens,
                               is_captioning=is_captioning,
                               min_res=min_res,
                               random_und_or_gen=random_und_or_gen,
                               showo_token_ids=showo_token_ids,
                               system=system)

    if accelerator is not None and accelerator.num_processes > 1:
        sampler = DistributedSampler(dataset,
                                     num_replicas=accelerator.num_processes,
                                     rank=accelerator.process_index,
                                     shuffle=shuffle,
                                     drop_last=drop_last)
        shuffle = False
    else:
        sampler = None

    dataloader = DataLoader(dataset, batch_size=batch_size,
                            sampler=sampler, collate_fn=dataset.collate_fn,
                            shuffle=shuffle, num_workers=num_workers,
                            drop_last=drop_last)
    return dataloader


def example():

    from models.misc import get_text_tokenizer
    text_tokenizer, showo_token_ids = get_text_tokenizer(
        "Qwen/Qwen2.5-7B-Instruct",
        add_showo_tokens=True,
        return_showo_token_ids=True,
        llm_name="qwen2_5"
    )

    loader = create_imagetext_dataloader(
        '/mnt/bn/vgfm2/test_mlx/xavier/code/0923/show-o-next-v3/datasets/openimages_data.jsonl',
        text_tokenizer=text_tokenizer,
        batch_size=2,
        max_seq_len=1024,
        image_size=432,
        latent_height=27,
        latent_width=27,
        cond_dropout_prob=0.1,
        num_image_tokens=729,
        is_captioning=False,
        min_res=[256, 256],
        random_und_or_gen=0.0,
        num_workers=0,
        showo_token_ids=showo_token_ids,
        system=("system\nYou are a helpful assistant.<|im_end|>",
                "\n<|im_start|>user\n Generate a high-quality image based on the text prompt: ",
                "\n<|im_start|>assistant\n")
    )

    loader2 = create_imagetext_dataloader(
        '/mnt/bn/vgfm2/test_mlx/xavier/code/0923/show-o-next-v3/datasets/openimages_data.jsonl',
        text_tokenizer=text_tokenizer,
        batch_size=2,
        max_seq_len=1024,
        image_size=432,
        latent_height=27,
        latent_width=27,
        cond_dropout_prob=0.1,
        num_image_tokens=729,
        is_captioning=True,
        min_res=[256, 256],
        random_und_or_gen=0.0,
        num_workers=0,
        showo_token_ids=showo_token_ids,
        system=("system\nYou are a helpful assistant.<|im_end|>",
                "\n<|im_start|>user\n Generate a high-quality image based on the text prompt: ",
                "\n<|im_start|>assistant\n")
    )

    from datasets.mixed_dataloader import MixedDataLoader

    mixed_loader = MixedDataLoader(
        loader_list=[loader, loader2],
        samp_probs=[0.8, 0.2],
        accumulation=1,
        mode='concat_min_size'
    )

    for i, data in enumerate(mixed_loader):
        print()
        print(data['text_tokens'].shape, data['images'].shape,
              data['modality_positions'].shape, data['text_labels'].shape, data['text_masks'].shape,
              data['image_masks'].shape)
        print(text_tokenizer.batch_decode(data['text_tokens'])[-1])
        data['text_labels'][data['text_labels'] == -100] = 100
        print(text_tokenizer.batch_decode(data['text_labels'])[-1])
        import ipdb
        ipdb.set_trace()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-readers", type=int, default=1)
    parser.add_argument("--data-path", type=str, default="")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    example()
