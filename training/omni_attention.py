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
    def __init__(self, batch_size_t2i, batch_size_lm, batch_size_mmu, S, t2i_image_begin_end=[(128, 1152)], mmu_end=1027, right_padding=[(1024, 1280)], device='cuda'):
    # def __init__(self, batch_size_t2i, batch_size_lm, batch_size_mmu, S, t2i_image_begin_end=[(15, 20)], mmu_end=1027, right_padding=[(1024, 1280)], device='cuda'):
        super().__init__()

        self.batch_size_t2i = batch_size_t2i
        self.batch_size_lm = batch_size_lm
        self.batch_size_mmu = batch_size_mmu

        self.t2i_image_begin_end = t2i_image_begin_end
        self.t2i_full_begin = torch.arange(S, device=device)
        self.t2i_full_end = torch.arange(S, device=device)
        for image_begin, image_end in t2i_image_begin_end:
            self.t2i_full_begin[image_begin:image_end] = image_begin
            self.t2i_full_end[image_begin:image_end] = image_end

        self.mmu_end = mmu_end

        # if we add padding on the right most of sequence
        # self.right_pad_begins = torch.arange(S, device=device)
        # self.right_pad_ends = torch.arange(S, device=device)
        # for image_begin, image_end in right_padding:
        #     self.right_pad_begins[image_begin:image_end] = image_begin
        #     self.right_pad_ends[image_begin:image_end] = image_end

    def causal_mask(self, b, h, q_idx, kv_idx):
        # right_pad_mask = ~((kv_idx < self.right_pad_ends[q_idx]) & (kv_idx >= self.right_pad_begins[q_idx]))
        return (q_idx >= kv_idx) #& right_pad_mask

    def t2i_mask(self, b, h, q_idx, kv_idx):
        """
        (batch_size, seq_len)
        t2i sequence = [
                    [pad][pad][t2i][sot][text][text][eot][soi][image][image][eoi]
                    [pad][t2i][sot][text][text][text][eot][soi][image][image][eoi]
        ]
        left padding for the text
        #right padding for the requirement of flexattention (len is the multiple of 128)
        """
        # causal mask that excludes padding regions
        # eye_mask = (q_idx == kv_idx) to avoid the NaN issue
        causal_mask = ~((kv_idx < self.pad_ends[b, kv_idx])) & ((q_idx >= kv_idx)) | (q_idx == kv_idx)
        full_mask = (kv_idx < self.t2i_full_end[q_idx]) & (kv_idx >= self.t2i_full_begin[q_idx])
        # remove right padding attention (becuase we add some padding at the end of the sqeuence to meet the len of flexattention)
        # right_pad_mask = ~((kv_idx < self.right_pad_ends[q_idx]) & (kv_idx >= self.right_pad_begins[q_idx]))

        return (causal_mask | full_mask) #& right_pad_mask

    # TODO: check the boundary.
    def mmu_mask(self, b, h, q_idx, kv_idx):
        # right_pad_mask = ~((kv_idx < self.right_pad_ends[q_idx]) & (kv_idx >= self.right_pad_begins[q_idx]))
        return (q_idx >= kv_idx) | (kv_idx < self.mmu_end) #& right_pad_mask

    # TODO: check the boundary.
    def mmu_vit_mask(self, b, h, q_idx, kv_idx, system_prompt_len=28, num_clip_vit_feat=576):
        index = 1 + system_prompt_len + 1 + num_clip_vit_feat
        return (q_idx >= kv_idx) | ((kv_idx >= (1 + system_prompt_len + 1)) & (kv_idx < index))

    def mixed_mask(self, b, h, q_idx, kv_idx, num_clip_vit_feat=576):
        # causal mask that excludes padding regions
        # to avoid the NaN issue
        # eye_mask = (q_idx == kv_idx)
        causal_mask = ~(kv_idx < self.pad_ends[b, kv_idx]) & (q_idx >= kv_idx) | (q_idx == kv_idx)
        full_mask = (kv_idx < self.t2i_full_end[q_idx]) & (kv_idx >= self.t2i_full_begin[q_idx])
        # right_pad_mask = ~((kv_idx < self.right_pad_ends[q_idx]) & (kv_idx >= self.right_pad_begins[q_idx]))
        t2i_mask = (causal_mask | full_mask) #& right_pad_mask

        lm_mask = (q_idx >= kv_idx) #& right_pad_mask
        # mmu_mask = (q_idx >= kv_idx) | (kv_idx <= num_clip_vit_feat + 3) #& right_pad_mask
        mmu_mask = (q_idx >= kv_idx) | (kv_idx < self.mmu_end)

        return (((b < self.batch_size_t2i) & t2i_mask)
                ^ ((b >= self.batch_size_t2i) & (b < (self.batch_size_t2i + self.batch_size_lm)) & lm_mask)
                ^ ((b >= (self.batch_size_t2i + self.batch_size_lm)) & mmu_mask))

    def create_block_mask(self, sequence, pad_begin_ends=[(0, 80), (0, 100), (0, 110), (0, 0)], type="t2i"):
    # def create_block_mask(self, sequence, pad_begin_ends=[(0, 10), (0, 5), (0, 2), (0, 0)], type="t2i"):
        B, S = sequence.shape
        self.pad_begins = torch.arange(S, device='cuda').repeat(B, 1)
        self.pad_ends = torch.arange(S, device='cuda').repeat(B, 1)

        cnt = 0
        for pb, pe in pad_begin_ends:
            self.pad_begins[cnt, pb:pe] = pb
            self.pad_ends[cnt, pb:pe] = pe
            cnt += 1

        if type == "t2i":
            block_mask = create_block_mask(self.t2i_mask, B=B, H=None, Q_LEN=S, KV_LEN=S, _compile=True)
        elif type == "mmu":
            block_mask = create_block_mask(self.mmu_mask, B=B, H=None, Q_LEN=S, KV_LEN=S, _compile=True)
        elif type == "mmu_vit":
            block_mask = create_block_mask(self.mmu_vit_mask, B=B, H=None, Q_LEN=S, KV_LEN=S, _compile=True)
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
        # import ipdb
        # ipdb.set_trace()
        # print()
        return attn_mask

def create_attention_mask_for_mmu_vit(
        sequence,
        return_inverse_mask=False,
        system_prompt_len=0
):
    N, L = sequence.shape
    causal_mask = torch.tril(torch.ones((N, 1, L, L), dtype=torch.bool)).to(sequence.device)
    index = 1 + system_prompt_len + 1 + 576

    causal_mask[:, :, :, :index] = 1

    causal_mask[0:4, :, :, :] = 1

    if return_inverse_mask:
        inverted_mask = 1.0 - causal_mask.type(torch.int64)
        inverted_mask = inverted_mask.masked_fill(
            inverted_mask.to(torch.bool), torch.iinfo(torch.int64).min
        )
        return inverted_mask.to(dtype=torch.bool)
    else:
        return causal_mask

if __name__ == '__main__':

    from triton.testing import do_bench

    B = 12
    S = 1152  # must be the multiple of 128
    H = 8
    D = 64
    q, k, v = [torch.randn(B, H, S, D, dtype=torch.float16) for _ in range(3)]

    OAM = OmniAttentionMechanism(4, 4, 4, S)

    sequence = torch.randn((B, S), device='cuda')
    block_mask = OAM.create_block_mask(sequence, type='t2i')
    print(block_mask)

    flex_attn = lambda: flex_attention(q, k, v, block_mask=block_mask)
    print("t2i flexattention: ", do_bench(flex_attn))

    mask = OAM.test()
    import ipdb

    ipdb.set_trace()

    sequence = torch.randn((B, S), device='cuda')
    block_mask = OAM.create_block_mask(sequence, type='causal')
    print(block_mask)

    flex_attn = lambda: flex_attention(q, k, v, block_mask=block_mask)
    print("lm flexattention: ", do_bench(flex_attn))

    sequence = torch.randn((B, S), device='cuda')
    import time
    s = time.time()
    block_mask = OAM.create_block_mask(sequence, type='mmu')
    print(block_mask)
    print(time.time() - s, 'create mmu mask')

    flex_attn = lambda: flex_attention(q, k, v, block_mask=block_mask)
    print("mmu flexattention: ", do_bench(flex_attn))

    sequence = torch.randn((B, S), device='cuda')
    block_mask = OAM.create_block_mask(sequence, type='mmu_vit')
    print(block_mask.shape)

    flex_attn = lambda: flex_attention(q, k, v, block_mask=block_mask)
    print("mmu vit flexattention: ", do_bench(flex_attn))

    sequence = torch.randn((B, S), device='cuda')
    import time
    s = time.time()
    block_mask = OAM.create_block_mask(sequence, type='mixed-t2i-lm-mmu')
    print(block_mask.shape)
    print(time.time()-s, 'create mixed mask')

    flex_attn = lambda: flex_attention(q, k, v, block_mask=block_mask)
    print("mixed-t2i-lm-mmu flexattention: ", do_bench(flex_attn))



    import torch.nn.functional as F
    from torch.backends.cuda import sdp_kernel, SDPBackend

    # Helpful arg mapper
    backend_map = {
        SDPBackend.MATH: {"enable_math": True, "enable_flash": False, "enable_mem_efficient": False},
        SDPBackend.FLASH_ATTENTION: {"enable_math": False, "enable_flash": True, "enable_mem_efficient": False},
        SDPBackend.EFFICIENT_ATTENTION: {
            "enable_math": False, "enable_flash": False, "enable_mem_efficient": True}
    }

    with sdp_kernel(**backend_map[SDPBackend.FLASH_ATTENTION]):
    # with sdp_kernel(**backend_map[SDPBackend.EFFICIENT_ATTENTION]):
    # with sdp_kernel(**backend_map[SDPBackend.MATH]):
        q, k, v = [torch.randn(B, H, S, D, dtype=torch.float16) for _ in range(3)]
        sequence = torch.randn(B, S)
        s = time.time()
        mask = create_attention_mask_for_mmu_vit(sequence)
        print(time.time() - s, 'create mmu vit mask')
        xformer_attn = lambda: F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        print("xformer: ", do_bench(xformer_attn))





