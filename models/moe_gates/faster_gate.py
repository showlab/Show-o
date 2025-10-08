r"""
The example topology-aware gate for two-layer tree-like topology, proposed by
the PPoPP'22 paper, FasterMoE.  Limited number of tokens are sent across the
upper-level slow connection, and other ones are re-directed to experts in the
local network.

The number of GPUs to form such a local network is defined by an environment
variable `FMOE_TOPO_GPUS_PER_NODE`, and it is by default `8`.

The fraction of tokens that are allowed to be sent across nodes is defined by
an environement variable `FMOE_TOPO_OUTGOING_FRACTION`, and it is by default
`0.14`. Users are supposed to set the proper value in their own environemnt,
guided by some performance model, to achieve maximum throughput.
"""

from naive_gate import NaiveGate

import os
import sys
import torch
import torch.nn.functional as F
from utils import limit_by_capacity, count_by_gate

# Note: FasterGate is for advanced distributed training with topology-aware routing
# Some features may require CUDA extensions


nw_per_node = 8
try:
    nw_per_node = int(os.environ["FMOE_TOPO_GPUS_PER_NODE"])
except Exception:
    pass


class FasterGate(NaiveGate):
    def __init__(
        self, d_model, num_expert, world_size, node_rank=1, gate_bias=True, top_k=2
    ):
        assert top_k == 2, "top_k should be 2 in faster"
        super().__init__(d_model, num_expert, world_size, top_k=2, gate_bias=gate_bias)
        self.ne_per_node = nw_per_node * num_expert
        self.ogn_ratio = 0.14
        try:
            self.ogn_ratio = float(os.environ["FMOE_TOPO_OUTGOING_FRACTION"])
        except Exception:
            pass
        self.node_rank = node_rank

        mask = [1] * world_size * num_expert
        for i in range(num_expert * world_size):
            if i // self.ne_per_node == self.node_rank:
                mask[i] = 0
        self.mask = torch.Tensor(mask).bool()
        self.policy_fn = None
        print("node rank {} mask {}".format(node_rank, mask))

    def forward(self, inp):
        if self.mask.device != inp.device:
            self.mask = self.mask.to(inp.device)

        gate_score = self.gate(inp)
        lim_mask = self.mask

        top2_val, top2_idx = torch.topk(gate_score, k=2, dim=-1)
        S = gate_score.shape[0]
        top_k = 2

        with torch.no_grad():
            top1_idx = top2_idx.view((-1, top_k))[:, 0]
            top1_val = top2_val.view((-1, top_k))[:, 0]
        c_e = (
            torch.scatter_add(
                torch.zeros(self.tot_expert, device=top1_idx.device),
                0,
                top1_idx,
                torch.ones_like(top1_idx, dtype=torch.float),
            )
            / S
        )
        m_e = torch.mean(F.softmax(gate_score, dim=1), dim=0)
        loss = torch.mean(c_e * m_e) * (self.num_expert**2)
        self.set_loss(loss)

        with torch.no_grad():
            if self.policy_fn is None:
                stored_models = torch.zeros(
                    self.num_expert * self.world_size, dtype=torch.bool
                )
            else:
                # TODO: Fix this after expert shadowing is ported
                _, lec, aec, gec, agec = count_by_gate(
                    top2_idx, self.num_expert, self.world_size, require_pos=False
                )
                stored_models = self.policy_fn(
                    aec, agec, self.num_expert, self.world_size, inp.shape[-1], True
                )
            lim_mask = lim_mask & ~stored_models.view(-1).to(lim_mask.device)

            ogn_mask = lim_mask[top1_idx]
            ogn_thres = int(inp.shape[0] * self.ogn_ratio)

        if ogn_mask.sum().item() < ogn_thres:
            topk_val, topk_idx = torch.topk(gate_score, k=self.top_k)
            topk_val = F.softmax(topk_val, dim=-1)
            return topk_idx, topk_val

        with torch.no_grad():
            top1_val[~ogn_mask] = float("-inf")
            _, top_ogn = torch.topk(top1_val.view(-1), k=ogn_thres)
            cand = gate_score.clone()
            cand[:, lim_mask] = float("-inf")
            _, topk_idx = torch.topk(cand, k=self.top_k)
            topk_idx[top_ogn, 1] = top1_idx.view(-1)[top_ogn]

        idx_x = torch.arange(inp.shape[0], device=inp.device).repeat_interleave(2)
        topk_val = gate_score[idx_x, topk_idx.view(-1)].view(-1, self.top_k)

        topk_val = F.softmax(topk_val, dim=-1)

        return topk_idx, topk_val


def gen_faster_gate(rank):
    def _gen(d_model, num_expert, world_size, top_k=2):
        assert top_k == 2
        return FasterGate(d_model, num_expert, world_size, rank // nw_per_node)

    return _gen
