import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from naive_gate import NaiveGate
from utils import limit_by_capacity


class SwitchGate(NaiveGate):
    r"""
    A switch gate implementation
    """

    def __init__(self, d_model, num_expert, world_size, top_k=1,
            switch_eps=.1, capacity=(20, 20), gate_bias=True):
        assert top_k == 1, 'top_k should be 1 in switch'
        super().__init__(d_model, num_expert, world_size, top_k=1, gate_bias=gate_bias)
        self.switch_eps = switch_eps
        self.capacity = capacity

    def forward(self, inp):
        r"""
        The switch firstly conduct softmax and then calculates the top-1
        """
        score = self.gate(inp)

        if self.training:
            # random uniform number from [1-eps, 1+eps]
            noise = torch.rand_like(score)
            noise = noise * 2 * self.switch_eps + 1.0 - self.switch_eps
            score += noise

        # fp32 softmax for numerical stability
        score = F.softmax(score.float(), dim=-1)

        top1_score, top1_idx = torch.topk(
            score, k=1, dim=-1, largest=True
        )  # [.. x top_k]
        top1_score = top1_score.to(dtype=inp.dtype)

        # cap_rate = self.capacity[0 if self.training else 1]
        # capacity = math.ceil(cap_rate * inp.shape[0] / self.num_expert)
        capacity = self.capacity[0 if self.training else 1]
        _new_lec, _new_gec, top1_idx = limit_by_capacity(
                top1_idx, self.num_expert, self.world_size, capacity)

        valid_idx = top1_idx[top1_idx > -1]
        fraction_expert = torch.scatter_add(
                torch.zeros(self.tot_expert, device=valid_idx.device),
                0,
                valid_idx,
                torch.ones_like(valid_idx, dtype=torch.float),
            ) / valid_idx.numel()
        prob_expert = score.sum(dim=0) / valid_idx.numel()
        loss = (fraction_expert * prob_expert).sum() * self.tot_expert
        self.set_loss(loss)
        return top1_idx, top1_score 
