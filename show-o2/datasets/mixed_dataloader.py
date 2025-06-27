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

from torch.utils.data import DataLoader
from typing import List, Literal
import bisect
import random
import collections
import torch
from itertools import chain

def weighted_random_sample_fast(elements, probabilities):
    cum_probs = []
    current = 0.0
    for p in probabilities:
        current += p
        cum_probs.append(current)
    return elements[bisect.bisect_left(cum_probs, random.random())]


class MixedDataLoader:
    def __init__(self,
                 loader_list: List[DataLoader],
                 samp_probs: List[float] = [0.1],
                 accumulation: int = 1,
                 mode: Literal['max_size_cycle', 'min_size'] = 'max_size_cycle',
                 n_iters_per_sequential_iter: int = 1,
                 ):
        self.accumulation = accumulation
        self.samp_probs = samp_probs
        self.mode = mode
        self.n_iters_per_sequential_iter = n_iters_per_sequential_iter
        self.current_loader_idx = 0

        self.loader_list = loader_list
        self.iter_list = [iter(loader) for loader in self.loader_list]

    def __iter__(self):
        self.iter_list = [iter(loader) for loader in self.loader_list]
        self.exhausted = False
        return self

    def _max_size_cycle(self):
        batched = []
        for _ in range(self.accumulation):
            idx = weighted_random_sample_fast([i for i in range(len(self.iter_list))], self.samp_probs)
            try:
                batch = next(self.iter_list[idx])
            except StopIteration:
                self.iter_list[idx] = iter(self.loader_list[idx])
                batch = next(self.iter_list[idx])
            batched.append(batch)

        return self.collate_fn(batched)

    def _min_size(self):
        if self.exhausted:
            raise StopIteration

        try:
            batched = []
            for _ in range(self.accumulation):
                idx = weighted_random_sample_fast([i for i in range(len(self.iter_list))], self.samp_probs)
                batch = next(self.iter_list[idx])
                batched.append(batch)
            return self.collate_fn(batched)
        except StopIteration:
            self.exhausted = True
            raise

    def _concat_max_size_cycle(self):
        batched = []
        for idx in range(len(self.iter_list)):
            try:
                batch = next(self.iter_list[idx])
            except StopIteration:
                self.iter_list[idx] = iter(self.loader_list[idx])
                batch = next(self.iter_list[idx])
            batched.append(batch)

        return self.collate_fn(batched)

    def _concat_min_size(self):
        if self.exhausted:
            raise StopIteration

        try:
            batched = []
            for idx in range(len(self.iter_list)):
                batch = next(self.iter_list[idx])
                batched.append(batch)
            return self.collate_fn(batched)
        except StopIteration:
            self.exhausted = True
            raise

    def _sequential_max_size_cycle(self):

        batched = []
        loaders_used = 0

        while loaders_used < self.n_iters_per_sequential_iter:
            try:
                batch = next(self.iter_list[self.current_loader_idx])
                batched.append(batch)
            except StopIteration:
                self.iter_list[self.current_loader_idx] = iter(self.loader_list[self.current_loader_idx])
                batch = next(self.iter_list[self.current_loader_idx])
                batched.append(batch)

            self.current_loader_idx += 1
            loaders_used += 1

            if self.current_loader_idx >= len(self.loader_list):
                self.current_loader_idx = 0

        return self.collate_fn(batched)

    def __next__(self):
        if self.mode == "max_size_cycle":
            return self._max_size_cycle()
        elif self.mode == "min_size":
            return self._min_size()
        elif self.mode == "concat_max_size_cycle":
            return self._concat_max_size_cycle()
        elif self.mode == "concat_min_size":
            return self._concat_min_size()
        elif self.mode == "sequential_max_size_cycle":
            return self._sequential_max_size_cycle()
        else:
            raise NotImplementedError

    def collate_fn(self, batch):
        batched = collections.defaultdict(list)
        for data in batch:
            for k, v in data.items():
                batched[k].append(v)
        for k, v in batched.items():
            if k in ('texts', 'data_type'):
                batched[k] = list(chain.from_iterable(v))
            else:
                batched[k] = torch.concat(v, dim=0)
        return batched
