# coding=utf-8
# Copyright 2024 NUS Show Lab.
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
import random

import torch
from parquet.parquet_dataset import CruiseParquetDataset


class RefinedWebDataset(CruiseParquetDataset):
    def __init__(self,
                data_path,
                rank: int = 0,
                world_size: int = 1,
                shuffle=True,
                repeat=True,
                buffer_size=1000,
                max_length=8000,
                num_workers=1,
                **kwargs
                ):
        super().__init__(data_path, rank, world_size, shuffle, repeat, verbose=False, buffer_size=buffer_size, meta_data_path=None, state_path=None, num_workers=num_workers)
        self.max_length = max_length

    def __iter__(self):
        for example in self.generate():
            try:
                data, current_worker_hash, data_idx, seed = example
                text = data['content'].replace('\n', '')
                if len(text) > self.max_length:
                    start_index = random.randint(0, len(text) - self.max_length - 1)
                    selected_text = text[start_index:start_index + self.max_length]
                else:
                    selected_text = text
                ret = {'input_ids': selected_text}
                yield ret

            except Exception as e:
                # print('internal dataset iter error', e)
                continue

    def collate_fn(self, batch):
        batched = collections.defaultdict(list)
        for data in batch:
            for k, v in data.items():
                batched[k].append(v)
        for k, v in batched.items():
            if k not in ('key', 'input_ids', 'similarity'):
                batched[k] = torch.stack(v, dim=0)

        return batched

if __name__ == '__main__':

    dataset = RefinedWebDataset('/mnt/bn/vgfm2/test_mlx/xavier/data/falcon-refinedweb/data/*.parquet', num_workers=10)
    from torch.utils.data import DataLoader
    train_dataloader = DataLoader(dataset, batch_size=10,
                                  sampler=None, collate_fn=dataset.collate_fn,
                                  num_workers=10)
                                  # num_workers=0)
    for i, batch in enumerate(train_dataloader):
        print(len(batch['input_ids'][0]))
        import ipdb; ipdb.set_trace()
