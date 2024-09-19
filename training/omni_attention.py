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

import torch
torch.set_default_device('cuda')
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
flex_attention = torch.compile(flex_attention, dynamic=False)

# Class for Omni-Attention Mechanism based on FlexAttention (torch >= 2.5)
class OmniAttentionMechanism(torch.nn.Module):

    # def __init__(self, batch_size_t2i, batch_size_lm, batch_size_mmu, S, image_begin_ends=[(128 + 1, 128 + 1 + 258)], device='cuda'):
    def __init__(self, batch_size_t2i, batch_size_lm, batch_size_mmu, S, image_begin_ends=[(512, 1024)], device='cuda'):
        super().__init__()

        self.batch_size_t2i = batch_size_t2i
        self.batch_size_lm = batch_size_lm
        self.batch_size_mmu = batch_size_mmu

        self.image_begin_ends = image_begin_ends
        self.full_starts = torch.arange(S, device=device)
        self.full_ends = torch.arange(S, device=device)
        for image_begin, image_end in image_begin_ends:
            self.full_starts[image_begin:image_end] = image_begin
            self.full_ends[image_begin:image_end] = image_end

    def causal_mask(self, b, h, q_idx, kv_idx):
        return q_idx >= kv_idx

    def t2i_mask(self, b, h, q_idx, kv_idx):
        # causal mask that excludes padding regions
        causal_mask = ~(kv_idx < self.pad_ends[b, kv_idx]) & (q_idx >= kv_idx)
        full_mask = (kv_idx < self.full_ends[q_idx]) & (kv_idx >= self.full_starts[q_idx])
        # to avoid the NaN issue
        eye_mask = (q_idx == kv_idx)
        return eye_mask ^ (causal_mask | full_mask)

    # TODO: check the boundary.
    def mmu_mask(self, b, h, q_idx, kv_idx, eoi_index=258):
        return (q_idx >= kv_idx) | (kv_idx <= eoi_index)

    # TODO: check the boundary.
    def mmu_vit_mask(self, b, h, q_idx, kv_idx, system_prompt_len=28, num_clip_vit_feat=576):
        index = 1 + system_prompt_len + 1 + num_clip_vit_feat
        return (q_idx >= kv_idx) | (kv_idx >= (1 + system_prompt_len + 1) & kv_idx < index)

    # TODO: A bit cumbersome. Very slow. Should be improved.
    def mixed_mask(self, b, h, q_idx, kv_idx, num_clip_vit_feat=576):
        # causal mask that excludes padding regions
        causal_mask = ~(kv_idx < self.pad_ends[b, kv_idx]) & (q_idx >= kv_idx)
        full_mask = (kv_idx < self.full_ends[q_idx]) & (kv_idx >= self.full_starts[q_idx])
        # to avoid the NaN issue
        eye_mask = (q_idx == kv_idx)
        t2i_mask = eye_mask ^ (causal_mask | full_mask)

        lm_mask = q_idx >= kv_idx
        mmu_mask = (q_idx >= kv_idx) | (kv_idx <= num_clip_vit_feat + 3)

        return (((b < self.batch_size_t2i) & t2i_mask)
                ^ ((b >= self.batch_size_t2i) & (b < (self.batch_size_t2i + self.batch_size_lm)) & lm_mask)
                ^ ((b >= (self.batch_size_t2i + self.batch_size_lm)) & mmu_mask))

    def create_block_mask(self, sequence, pad_begin_ends=[(0, 256), (0, 300), (0, 400), (0, 0)], type="t2i"):
        B, S = sequence.shape
        self.pad_starts = torch.arange(S, device='cuda').repeat(B, 1)
        self.pad_ends = torch.arange(S, device='cuda').repeat(B, 1)

        cnt = 0
        for pb, pe in pad_begin_ends:
            self.pad_starts[cnt, pb:pe] = pb
            self.pad_ends[cnt, pb:pe] = pe
            cnt += 1

        if type == "t2i":
            block_mask = create_block_mask(self.t2i_mask, B=B, H=None, Q_LEN=S, KV_LEN=S, _compile=True)
        elif type == "mmu":
            block_mask = create_block_mask(self.mmu_mask, B=B, H=None, Q_LEN=S, KV_LEN=S, _compile=True)
        elif type == "causal":
            block_mask = create_block_mask(self.causal_mask, B=B, H=None, Q_LEN=S, KV_LEN=S, _compile=True)
        elif type == "mixed-t2i-lm-mmu":
            block_mask = create_block_mask(self.mixed_mask, B=B, H=None, Q_LEN=S, KV_LEN=S, _compile=True)
        else:
            raise ValueError("Unknown type")

        return block_mask

    def test(self):
        attn_mask = torch.zeros(4, 1, 21, 21)
        for b in range(4):
            for h in range(1):
                for q_idx in range(21):
                    for kv_idx in range(21):
                        attn_mask[b, h, q_idx, kv_idx] = self.t2i_mask(b, h, q_idx, kv_idx)
        import ipdb
        ipdb.set_trace()
        print()

if __name__ == '__main__':

    from triton.testing import do_bench

    B = 12
    S = 1024  # must be the multiple of 128
    H = 8
    D = 64
    q, k, v = [torch.randn(B, H, S, D, dtype=torch.float16) for _ in range(3)]

    OAM = OmniAttentionMechanism(4, 4, 4, S)
    sequence = torch.randn((B, S), device='cuda')
    block_mask = OAM.create_block_mask(sequence, type='t2i')
    print(block_mask)

    flex_attn = lambda: flex_attention(q, k, v, block_mask=block_mask)
    print("t2i flexattention: ", do_bench(flex_attn))

    sequence = torch.randn((B, S), device='cuda')
    block_mask = OAM.create_block_mask(sequence, type='causal')
    print(block_mask)

    flex_attn = lambda: flex_attention(q, k, v, block_mask=block_mask)
    print("lm flexattention: ", do_bench(flex_attn))

    sequence = torch.randn((B, S), device='cuda')
    block_mask = OAM.create_block_mask(sequence, type='mmu')
    print(block_mask)

    flex_attn = lambda: flex_attention(q, k, v, block_mask=block_mask)
    print("mmu flexattention: ", do_bench(flex_attn))


