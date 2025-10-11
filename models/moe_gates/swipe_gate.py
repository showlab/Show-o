r"""
Balanced gate using SWIPE algorithm
"""
import math
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from .naive_gate import NaiveGate

from .utils import count_by_gate

# Note: SwipeGate requires CUDA extension (swipe_once kernel)
# This gate is not fully functional without the CUDA extension


class SwipeGate(NaiveGate):
    def __init__(self, d_model, num_expert, world_size, top_k=2, gate_bias=True):
        super().__init__(d_model, num_expert, world_size, top_k, gate_bias)

    def swipe_once(self, idx, capacity, bias):
        # This requires CUDA extension which is not available
        # Falling back to identity (no swapping)
        raise NotImplementedError(
            "SwipeGate requires CUDA extension (fmoe_cuda.swipe_once) which is not available. "
            "Please use NaiveGate, NoisyGate, SwitchGate, or GShardGate instead."
        )
        # with torch.no_grad():
        #     # Original CUDA kernel call:
        #     # idx_new, capacity = fmoe_native.swipe_once(idx, capacity,
        #     #         self.num_expert, self.world_size, bias)
        #     # idx_new = idx_new.to(idx.device)
        # return idx, capacity


    def forward(self, inp):
        score = self.gate(inp)
        orig_score, orig_idx = torch.topk(score, k=self.top_k, dim=-1)

        if not self.training:
            topk_val = F.softmax(orig_score, dim=-1)
            return orig_idx, topk_val

        capacity = torch.scalar_tensor(inp.shape[0] * self.top_k,
                dtype=torch.long)

        topk_idxs = []
        topk_vals = []
        idx_x = torch.arange(inp.shape[0], device=inp.device)
        for k in range(self.top_k):
            idx, capacity = self.swipe_once(orig_idx[:, k], capacity,
                    k % self.num_expert)
            topk_vals.append(score[idx_x, idx])
            topk_idxs.append(idx)
        topk_idx = torch.stack(topk_idxs).transpose(0, 1)
        topk_val = torch.stack(topk_vals).transpose(0, 1)
        topk_val = F.softmax(topk_val, dim=-1)
        return topk_idx, topk_val
