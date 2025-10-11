r"""
Zero gate that direct all input to gate 0
"""
from .base_gate import BaseGate

import torch
import torch.nn as nn
import torch.nn.functional as F


class ZeroGate(BaseGate):
    r"""
    Guide all input samples to gate 0.
    """

    def __init__(self, _1, num_expert, world_size, top_k=2):
        super().__init__(num_expert, world_size)
        self.top_k = top_k

    def forward(self, inp):
        r"""
        All output to expert 1
        """
        idx = torch.zeros(
            inp.shape[0] * self.top_k, dtype=torch.int64, device=inp.device
        )
        gate_score = (
            torch.ones(inp.shape[0] * self.top_k, device=inp.device) / self.top_k
        )
        return idx, gate_score.reshape(-1, 1, self.top_k)
