r"""
Distributed Capacity gate, extended from GShard gate.
Instead of setting capacity based on local batch size and expert count,
the global load of each experts are calculated, and then the experts make
decisions of capacities on each worker.
"""
import math
import torch
import torch.nn.functional as F
from naive_gate import NaiveGate
from utils import limit_by_capacity


class DCGate(NaiveGate):
    def __init__(self, d_model, num_expert, world_size,
            top_k=2, capacity=(20, 20), random_routing=False, gate_bias=True):
        assert top_k == 2, 'topk should be 2 in gshard'
        super().__init__(d_model, num_expert, world_size, top_k=2, gate_bias=gate_bias)
        self.capacity = capacity
        self.random_routing = random_routing

    def forward(self, x):
        naive_outs = super().forward(x, return_all_scores=True)
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
        loss = torch.mean(c_e * m_e) * (self.num_expert ** 2)
        self.set_loss(loss)

        cap_rate = self.capacity[0 if self.training else 1]
        capacity = math.ceil(cap_rate * x.shape[0])
        _new_lec, _new_gec, topk_idx = limit_by_capacity(
                topk_idx, self.num_expert, self.world_size, capacity)

        if self.random_routing:
            rand_routing_prob = torch.rand(gate_score.size(0), device=x.device)
            mask = (2 * topk_val[:, 1] < rand_routing_prob)
            topk_idx[:, 1].masked_fill_(mask, -1)

        return topk_idx, topk_val
