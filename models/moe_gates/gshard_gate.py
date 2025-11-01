import math
import torch
import torch.nn.functional as F

from .naive_gate import NaiveGate
from .utils import limit_by_capacity, prune_gate_by_capacity


class GShardGate(NaiveGate):
    def __init__(self, d_model, num_expert, world_size,
            topk=2, capacity=(20, 20), random_routing=False, gate_bias=True, top_k=2):
        assert topk == 2, 'topk should be 2 in gshard'
        super().__init__(d_model, num_expert, world_size, gate_bias=gate_bias, top_k=top_k)
        self.capacity = capacity
        self.random_routing = random_routing

    def forward(self, x, temperature: float | None = None, return_all_scores: bool = False):
        naive_outs = super().forward(x, return_all_scores=True)
        topk_idx, topk_val, gate_logits = naive_outs

        S = gate_logits.shape[0]
        top1_idx = topk_idx.view((-1, self.top_k))[:, 0]
        c_e = torch.scatter_add(
                torch.zeros(self.tot_expert, device=top1_idx.device),
                0,
                top1_idx,
                torch.ones_like(top1_idx, dtype=torch.float),
                ) / S
        m_e = torch.mean(F.softmax(gate_logits, dim=1), dim=0)
        loss = torch.mean(c_e * m_e) * (self.num_expert ** 2)
        self.set_loss(loss)

        if temperature is not None:
            tau = max(float(temperature), 1e-6)
            probs = F.gumbel_softmax(gate_logits, tau=tau, hard=False, dim=1)  # [B, E_tot]
            topk_prob, topk_idx = probs.topk(self.top_k, dim=1)
            topk_val = topk_prob

        cap_rate = self.capacity[0 if self.training else 1]
        capacity = math.ceil(cap_rate * x.shape[0])
        capacity = capacity * self.top_k // (self.world_size * self.num_expert)
        capacity = torch.ones(self.num_expert * self.world_size,
                dtype=torch.int32, device=topk_idx.device) * capacity
        # capacity = self.capacity[0 if self.training else 1]
        topk_idx = prune_gate_by_capacity(topk_idx, capacity,
                self.num_expert, self.world_size)

        if self.random_routing:
            rand_routing_prob = torch.rand(gate_logits.size(0), device=x.device)
            mask = (2 * topk_val[:, 1] < rand_routing_prob)
            topk_idx[:, 1].masked_fill_(mask, -1)

        if temperature is not None:
            valid = (topk_idx >= 0).float()
            topk_val = topk_val * valid
            denom = topk_val.sum(dim=1, keepdim=True).clamp_min(1e-8)
            topk_val = topk_val / denom

        if return_all_scores:
            return topk_idx, topk_val, gate_logits
        return topk_idx, topk_val
