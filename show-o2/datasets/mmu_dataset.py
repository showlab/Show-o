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

import collections
from typing import Any, Callable

import torch
from torchvision.datasets.folder import default_loader
from datasets.utils import image_transform, resize_and_pad_image, to_tensor_and_normalize
import os
import json
import torch.utils.data as data
import copy
from PIL import Image
IGNORE_INDEX = -100

class MMUDataset(data.Dataset):
    def __init__(
            self,
            root: str,
            text_tokenizer,
            max_seq_len=1024,
            source_max_len=512,
            target_max_len=512,
            image_size=768,
            latent_height=24,
            latent_width=24,
            num_image_tokens=576,
            cond_dropout_prob=0.1,
            loader: Callable[[str], Any] = default_loader,
            is_clip_encoder=False,
            annotation_path="",
            default_system_prompt="system\nYou are a helpful assistant.<|im_end|>",
            stage='pre-training',
            clip_processor=None,
            showo_token_ids=None
    ):

        self.text_tokenizer = text_tokenizer
        self.pad_id = self.text_tokenizer.pad_token_id
        self.bos_id = showo_token_ids['bos_id']
        self.eos_id = showo_token_ids['eos_id']
        self.boi_id = showo_token_ids['boi_id']
        self.eoi_id = showo_token_ids['eoi_id']
        self.img_pad_id = showo_token_ids['img_pad_id']
        self.img_id = showo_token_ids['img_id']
        self.data_type = "mmu"
        self.max_seq_len = max_seq_len
        self.source_max_len = source_max_len
        self.target_max_len = target_max_len
        self.image_size = image_size
        self.num_image_tokens = num_image_tokens
        self.h = latent_height
        self.w = latent_width
        self.cond_dropout_prob = cond_dropout_prob
        # 4 for bos, eos, boi, and eoi tokens
        self.max_text_len = max_seq_len - num_image_tokens - 4
        self.transform = image_transform
        if is_clip_encoder:
            self.mean = (0.48145466, 0.4578275, 0.40821073)
            self.std = (0.26862954, 0.26130258, 0.27577711)
        else:
            self.mean = (0.5, 0.5, 0.5)
            self.std = (0.5, 0.5, 0.5)
        self.stage = stage
        self.clip_processor = clip_processor
        if self.stage.startswith('pre-training'):
            self.default_system_prompt = None
        else:
            self.default_system_prompt = default_system_prompt

        self.root = root
        with open(annotation_path, 'r') as f:
            self.samples = json.load(f)

        self.loader = loader

        print(f"LLaVA dataset loaded. {self.__len__()} images!")

    def __len__(self):
        return len(self.samples)

    def format_multi_sequence_und_qwen2_5(self, sources, targets, ignore_question=True):

        text_tokens = []
        text_labels = []
        modality_positions = []

        if not self.stage.startswith('pre-training'):
            default_system_prompt = self.text_tokenizer(
                self.default_system_prompt,
                max_length=100,
                truncation=True,
                add_special_tokens=False,
            )['input_ids']
            role_a = self.text_tokenizer("\n<|im_start|>user\n", add_special_tokens=False)['input_ids']
            role_b = self.text_tokenizer("\n<|im_start|>assistant\n", add_special_tokens=False)['input_ids']

        cur_len = 1 # <|begin_of_text|>
        for source_ids, target_ids in zip(sources, targets):
            if not self.stage.startswith('pre-training'):
                source_ids = role_a + source_ids + [self.eos_id] + role_b
            # NOTE: only support one image
            if cur_len == 1 and not self.stage.startswith('pre-training'):
                source_ids = default_system_prompt + source_ids
            if self.img_id in source_ids:
                image_id_index = source_ids.index(self.img_id)
                source_ids = source_ids[:image_id_index] + \
                             [self.boi_id] + [self.img_pad_id] * self.num_image_tokens + [self.eoi_id] \
                             + source_ids[image_id_index + 1:]

                # +1 for one <|im_start|> token
                modality_positions.append((cur_len + image_id_index + 1, self.num_image_tokens))

            text_tokens.extend(source_ids + target_ids)
            # ignore question
            if ignore_question:
                text_labels.extend(
                    [IGNORE_INDEX for _ in range(len(source_ids))] + copy.deepcopy(target_ids)
                )
            else:
                text_labels.extend(copy.deepcopy(source_ids + target_ids))

            cur_len = len(text_tokens) + 1

        text_labels = [IGNORE_INDEX] + text_labels
        text_tokens = [self.bos_id] + text_tokens
        text_labels = text_labels + [IGNORE_INDEX] * (self.max_seq_len - len(text_labels))
        text_tokens = text_tokens + [self.pad_id] * (self.max_seq_len - len(text_tokens))

        text_tokens = torch.tensor(text_tokens)[:self.max_seq_len]
        text_labels = torch.tensor(text_labels)[:self.max_seq_len]

        if len(modality_positions) == 0:
            modality_positions = [(0, 0)]
        modality_positions = torch.tensor(modality_positions)

        text_mask = torch.where((text_tokens != self.img_pad_id) & (text_tokens != self.pad_id),
                                torch.ones_like(text_tokens), torch.zeros_like(text_tokens))
        image_mask = torch.where(text_tokens == self.img_pad_id,
                                 torch.ones_like(text_tokens), torch.zeros_like(text_tokens))

        return text_tokens, text_labels, modality_positions, text_mask, image_mask

    def __getitem__(self, idx):
        try:

            if 'image' in self.samples[idx]:
                image = self.loader(os.path.join(self.root, self.samples[idx]['image'])).convert('RGB')
            else:
                # print('No image found in this conversation.')
                image = Image.new('RGB', (self.image_size, self.image_size))
            if self.clip_processor is not None:
                image = self.clip_processor(images=[image], return_tensors="pt")['pixel_values'].squeeze(0)
            else:
                # image = self.transform(image, resolution=self.image_size, mean=self.mean, std=self.std)
                # resize and pad, not centercrop
                image = resize_and_pad_image(image, target_resolution=(self.image_size, self.image_size))
                img_clip = copy.deepcopy(image).resize((384, 384))
                img_clip = to_tensor_and_normalize(img_clip, mean=self.mean, std=self.std)
                image = to_tensor_and_normalize(image, mean=self.mean, std=self.std)

            conversation = []
            for conv in self.samples[idx]['conversations']:
                if conv['from'] == 'human':
                    if conv['value'].endswith('\n<image>'):
                        conv['value'] = '<image>\n' + conv['value'][:-len('\n<image>')]

                    # remove question from the first stage
                    if self.stage.startswith('pre-training'):
                        if '<image>' in conv['value']:
                            conv['value'] = '<image>'

                    conversation.append({"role": "user", "content": conv['value']})
                else:
                    conversation.append({"role": "assistant", "content": conv['value']})

            sources = [f"{conv['content']}" for conv in conversation if conv["role"] == "user"]
            targets = [f"{conv['content']}{self.text_tokenizer.eos_token}" for conv in conversation if
                       conv["role"] == "assistant"]

            # Tokenize
            # import ipdb
            # ipdb.set_trace()
            sources = [self.text_tokenizer(
                source,
                max_length=self.source_max_len,
                truncation=True,
                add_special_tokens=False
            ).input_ids for source in sources]

            targets = [self.text_tokenizer(
                target,
                max_length=self.target_max_len,
                truncation=True,
                add_special_tokens=False
            ).input_ids for target in targets]

            text_tokens, text_labels, modality_positions, text_mask, image_mask = \
                self.format_multi_sequence_und_qwen2_5(sources, targets)

            ret = {'text_tokens': text_tokens, 'text_labels': text_labels, 'images': image,
                   'modality_positions': modality_positions, 'text_masks': text_mask, 'data_type': self.data_type,
                   'image_masks': image_mask, 'texts': self.text_tokenizer.batch_decode(text_tokens),
                   'images_clip': img_clip}

            return ret

        except Exception as e:
            print(e)
            return self.__getitem__(idx+1)

    def collate_fn(self, batch):
        batched = collections.defaultdict(list)
        for data in batch:
            for k, v in data.items():
                batched[k].append(v)
        for k, v in batched.items():
            if k not in ('texts', 'data_type'):
                batched[k] = torch.stack(v, dim=0)

        return batched

if __name__ == '__main__':

    from torch.utils.data import DataLoader
    from models.misc import get_text_tokenizer

    text_tokenizer, showo_token_ids = get_text_tokenizer(
        # "meta-llama/Meta-Llama-3-8B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct",
        add_showo_tokens=True,
        return_showo_token_ids=True,
        # llm_name="llama3"
        llm_name="qwen2_5"
    )

    dataset_mmu = MMUDataset(
        root= "/mnt/bn/vgfm2/test_dit/pretraining_data",
        # root="/mnt/bn/vgfm2/",
        # root= "/mnt/bn/mllm-all-datasets/datasets/LLaVA-OneVision-Data-SI/images",
        image_size=384,
        num_image_tokens=729,
        max_seq_len=896,
        annotation_path="/mnt/bn/vgfm2/test_dit/blip_laion_cc_sbu_558k.json",
        # annotation_path="/mnt/bn/vgfm2/test_mlx/xavier/data/LLaVA-OneVision/stage1-5/OneVision_stage_1_5_all.json",
        # annotation_path="/mnt/bn/vgfm2/test_mlx/xavier/data/LLaVA-OneVision/stage1-5/OneVision_stage_1_5_all_with_densefusion.json",
        # annotation_path="/mnt/bn/vgfm2/test_mlx/xavier/data/LLaVA-OneVision/stage1-5/OneVision_stage_1_5_all_with_densefusion_mgm_sharegpt4v.json",
        # annotation_path="/mnt/bn/vgfm2/test_mlx/xavier/data/LLaVA-OneVision/stage1-5/Densefusion.json",
        # annotation_path="/mnt/bn/mllm-all-datasets/datasets/LLaVA-OneVision-Data-SI/llavaov-si-img-2.6m-v1.json",
        text_tokenizer=text_tokenizer,
        stage='pre-training',
        # stage='pre-training-1-5',
        # stage='tuning',
        showo_token_ids=showo_token_ids
    )
    train_dataloader_t2i = DataLoader(dataset_mmu, batch_size=16, collate_fn=dataset_mmu.collate_fn,
                                      shuffle=True, num_workers=16, drop_last=True)
    # print(len(train_dataloader_t2i))
    # import ipdb
    # ipdb.set_trace()
    from tqdm import tqdm
    for i, batch in enumerate(tqdm(train_dataloader_t2i)):
        # print(text_tokenizer.batch_decode(batch['text_tokens'])[1])
        # batch['text_labels'][batch['text_labels'] == -100] = 100
        # print(text_tokenizer.batch_decode(batch['text_labels'])[1])
        # import ipdb
        # ipdb.set_trace()
        aa = text_tokenizer.batch_decode(batch['text_tokens'])
        for j in range(len(aa)):
            if not aa[j].endswith("[PAD]"):
                import ipdb
                ipdb.set_trace()
        # continue
        # if (batch['modality_positions'][:, :, 1] == 576).sum() + (
        #         batch['modality_positions'][:, :, 1] == 0).sum() == 64 and batch['text_tokens'].shape[1] == 896:
        #     continue
        # else:
        #     import ipdb
        #     ipdb.set_trace()
        # print(f'[{i}/{len(train_dataloader_t2i)}]')
        # print(batch['text_tokens'])
        # print(batch['images'].shape)
        # import ipdb
        # ipdb.set_trace()
