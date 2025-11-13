import math
import torch
import torch.nn.functional as F
from .naive_gate import NaiveGate
from .utils import limit_by_capacity
from typing import Optional

import torch

def prune_gate_by_capacity(topk_idx: torch.Tensor,
                           capacity: torch.Tensor,
                           num_expert: int,
                           world_size: int) -> torch.Tensor:
    if topk_idx.dim() != 2:
        raise ValueError("topk_idx must be a 2D tensor of shape (S, top_k)")

    device = topk_idx.device
    top_k = topk_idx.size(1)
    tot_expert = capacity.numel()
    pruned = topk_idx.clone()
    flat = pruned.view(-1)
    valid_mask = flat >= 0
    if valid_mask.sum() == 0:
        return pruned

    counts = torch.zeros(tot_expert, dtype=torch.int32, device=device)
    cap = capacity.to(device=device, dtype=torch.int32)

    flat_np = flat
    idx_positions = torch.nonzero(valid_mask, as_tuple=False).squeeze(1)
    for pos in idx_positions:
        e = int(flat_np[pos].item())  # global expert id
        if e < 0 or e >= tot_expert:
            flat_np[pos] = -1
            continue
        if counts[e] < cap[e]:
            counts[e] += 1
        else:
            flat_np[pos] = -1

    pruned = flat_np.view_as(pruned)
    return pruned


class GShardGate(NaiveGate):
    def __init__(self, d_model, num_expert, world_size,
            top_k=2, capacity=(1.2, 2.4), random_routing=True, gate_bias=True, use_gumbel=False):
        assert top_k == 2, 'topk should be 2 in gshard'
        super().__init__(d_model, num_expert, world_size, top_k=2, gate_bias=gate_bias)
        self.capacity = capacity
        self.random_routing = random_routing

    def forward(
        self, 
        x, 
        temperature: float | None = None,
        return_all_scores: bool = False,
        bias: Optional[torch.Tensor] = None,
        ):

        total_bias = bias.to(device=x.device) if bias is not None else None
        naive_outs = super().forward(x, return_all_scores=True, bias=total_bias)
        topk_idx, topk_val, gate_score = naive_outs

        S = gate_score.shape[0]
        top1_idx = topk_idx.view((-1, self.top_k))[:, 0]
        c_e = torch.scatter_add(
            torch.zeros(self.tot_expert, device=top1_idx.device),
            0,
            top1_idx,
            torch.ones_like(top1_idx, dtype=torch.float),
        ) / S
        
        m_e = torch.mean(F.softmax(gate_score, dim=1), dim=0)
        gshard_loss = torch.mean(c_e * m_e) * (self.num_expert ** 2)
        target_load = 1.0 / self.tot_expert
        variance_penalty = torch.mean((c_e - target_load) ** 2) * (self.num_expert ** 2)
        loss = gshard_loss + variance_penalty
        self.set_loss(loss)

        cap_rate = self.capacity[0 if self.training else 1]
        capacity = math.ceil(cap_rate * x.shape[0])
        capacity = capacity * self.top_k // (self.world_size * self.num_expert)
        capacity = torch.ones(self.num_expert * self.world_size,
                dtype=torch.int32, device=topk_idx.device) * capacity
        topk_idx = prune_gate_by_capacity(topk_idx, capacity,
                self.num_expert, self.world_size)

        if self.random_routing:
            rand_routing_prob = torch.rand(gate_score.size(0), device=x.device)
            mask = (2 * topk_val[:, 1] < rand_routing_prob)
            topk_idx[:, 1].masked_fill_(mask, -1)

        return topk_idx, topk_val