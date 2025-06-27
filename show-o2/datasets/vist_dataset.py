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
import json
import os
import random
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from datasets.utils import image_transform, format_interleaved_sequence
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader


class VISTDataset(Dataset):
    """Dataset for Visual Story Telling (VIST) with interleaved image-text pairs."""

    def __init__(
            self,
            root: str,
            anno_path: str,
            text_tokenizer: Any,
            max_seq_len: int = 3840,
            image_size: int = 384,
            latent_height: int = 24,
            latent_width: int = 24,
            num_image_tokens: int = 576,
            cond_dropout_prob: float = 0.1,
            max_num_pairs: int = 5,
            loader: Callable[[str], Any] = default_loader,
            showo_token_ids: Optional[Dict[str, int]] = None,
            system: Tuple[str, str, str] = ("", "", ""),
            min_res: Optional[Tuple[int, int]] = None,
    ) -> None:
        """Initializes the VIST dataset.

        Args:
            root: Root directory of images.
            text_tokenizer: Tokenizer for text processing.
            max_seq_len: Maximum sequence length.
            image_size: Size to which images are resized.
            latent_height: Height of latent representation.
            latent_width: Width of latent representation.
            num_image_tokens: Number of tokens representing an image.
            cond_dropout_prob: Probability of conditioning dropout.
            max_num_pairs: Maximum number of image-text pairs per sample.
            loader: Function to load an image given its path.
            anno_path: Path to the annotation JSON file.
            showo_token_ids: Dictionary of special token IDs.
            system: Tuple of system prompt strings.
            min_res: Minimum resolution (height, width) for images.
        """
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
        self.data_type = "interleaved_data"
        self.transform = image_transform
        self.max_num_pairs = max_num_pairs

        self.root = root
        self.anno_path = anno_path
        self.samples: List[Dict[str, Any]] = []
        self.loader = loader

        with open(self.anno_path) as file:
            self.samples = json.load(file)

        print(f"VST dataset loaded. {len(self.samples)} samples!")

        self.flag_tokens = self.text_tokenizer(
            "Mixed-modality generation (VIST).", add_special_tokens=False
        ).input_ids
        self.system_tokens = self.text_tokenizer(system, add_special_tokens=False).input_ids
        self.system_token_len = sum(len(tokens) for tokens in self.system_tokens)

        if len(self.system_tokens[0]) == 0:
            # 4 for bos, eos, boi, and eoi tokens
            self.max_text_len = (
                                        max_seq_len
                                        - len(self.flag_tokens)
                                        - (num_image_tokens + 2) * max_num_pairs
                                        - 2
                                ) // max_num_pairs
        else:
            # 4 for bos, eos, boi, and eoi tokens
            # 1 for eos after text token (a bit tricky)
            # see more details in def format_sequence_gen_qwen2_5(...)
            self.max_text_len = (
                                        max_seq_len
                                        - (num_image_tokens + 2) * max_num_pairs
                                        - 2
                                        - self.system_token_len
                                        - 1
                                ) // max_num_pairs

        self.min_res = min_res if min_res is not None else (256, 256)

    def _get_interleaved_data(
            self, anno: Dict[str, Any]
    ) -> Tuple[List[Optional[torch.Tensor]], List[Optional[List[int]]], List[str]]:
        """Extracts interleaved image-text data from annotation.

        Args:
            anno: Annotation dictionary for a sample.

        Returns:
            Tuple of image list, tokenized text list, and raw texts.
        """
        if len(anno['images']) > self.max_num_pairs:
            start_fid = random.randint(0, len(anno['images']) - self.max_num_pairs - 1)
        else:
            start_fid = 0

        image_paths = anno['images'][start_fid: start_fid + self.max_num_pairs]
        texts = anno['captions'][start_fid: start_fid + self.max_num_pairs]

        image_list: List[Optional[torch.Tensor]] = []
        text_token_list: List[Optional[List[int]]] = []

        for path, text in zip(image_paths, texts):
            full_path = os.path.join(self.root, path)
            if not full_path.endswith('.jpg'):
                full_path += '.jpg'
            image = self.loader(full_path)
            image = self.transform(image, resolution=self.image_size)
            image_list.append(image)

            text_tokens = self.text_tokenizer(
                text,
                add_special_tokens=False,
                truncation=True,
                max_length=self.max_text_len,
            ).input_ids
            text_token_list.append(text_tokens)

        # Add flag token to the first text token list
        text_token_list[0] = self.flag_tokens + text_token_list[0]

        # Pad lists if fewer than max_num_pairs
        if len(image_list) != self.max_num_pairs:
            image_list += [None] * (self.max_num_pairs - len(image_list))
            text_token_list += [None] * (self.max_num_pairs - len(text_token_list))
            texts += [''] * (self.max_num_pairs - len(texts))

        return image_list, text_token_list, texts

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        try:
            anno = self.samples[idx]
            if len(anno['images']) == 0:
                return self.__getitem__(idx + 1)

            image_list, text_token_list, texts = self._get_interleaved_data(anno)
            (
                text_tokens,
                text_labels,
                modality_positions,
                text_mask,
                image_mask,
            ) = format_interleaved_sequence(
                image_list,
                text_token_list,
                self.bos_id,
                self.eos_id,
                self.boi_id,
                self.eoi_id,
                self.pad_id,
                self.img_pad_id,
                self.num_image_tokens,
                self.max_seq_len,
                self.max_num_pairs,
            )

            # Ignore flag tokens in the label (first one is bos token)
            text_labels[1: len(self.flag_tokens) + 1] = -100

            temp: List[torch.Tensor] = []
            for img in image_list:
                if img is not None:
                    temp.append(img)
                else:
                    temp.append(torch.zeros((3, self.image_size, self.image_size)))

            image = torch.stack(temp, dim=0)
            return {
                'text_tokens': text_tokens,
                'text_labels': text_labels,
                'images': image,
                'modality_positions': modality_positions,
                'text_masks': text_mask,
                'image_masks': image_mask,
                'texts': texts,
                'data_type': self.data_type,
            }

        except Exception:  # pylint: disable=broad-except
            return self.__getitem__(idx + 1)

    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate function to batch data."""
        batched = collections.defaultdict(list)
        for data in batch:
            for key, value in data.items():
                batched[key].append(value)
        for key, value in batched.items():
            if key not in ('texts', 'data_type'):
                batched[key] = torch.stack(value, dim=0)
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

    dataset = VISTDataset(
        "/mnt/bn/vgfm2/test_mlx/xavier/data/visual_story_telling/train_images/images/train",
        anno_path="/mnt/bn/vgfm2/test_mlx/xavier/data/visual_story_telling/vst_train_annotations.json",
        text_tokenizer=text_tokenizer,
        showo_token_ids=showo_token_ids,
        image_size=512,
        max_seq_len=5120,
        num_image_tokens=1024,
        latent_height=32,
        latent_width=32,
        max_num_pairs=4
    )
    train_dataloader_t2i = DataLoader(dataset, batch_size=12, collate_fn=dataset.collate_fn,
                                      shuffle=False, num_workers=12)

    from datasets.mixed_dataloader import MixedDataLoader

    mixed_loader = MixedDataLoader(
        loader_list=[train_dataloader_t2i],
        samp_probs=[1.0],
        accumulation=1,
        mode='max_size_cycle'
    )
    from tqdm import tqdm

    for i, data in tqdm(enumerate(train_dataloader_t2i)):
        # print()
        # print(data['data_type'], data['text_tokens'].shape, data['images'].shape)
        # print(data['text_tokens'][0])
        # print(data['modality_positions'][0])
        import ipdb
        ipdb.set_trace()
        # continue
