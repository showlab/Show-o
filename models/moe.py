import torch
import torch.nn as nn

from .phi import PhiConfig, PhiModel, PhiMLP
from .moe_gates import GShardGate


class MoE(nn.Module):
    def __init__(self, num_experts, hidden_size, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.top_k = top_k

        self.experts = nn.ModuleList([PhiModel(PhiConfig(hidden_size=self.hidden_size)) for _ in range(self.num_experts)])

        self.alpha = nn.Parameter(torch.randn(num_experts))

        self.gate = GShardGate(self.hidden_size, self.num_experts, 1, top_k=self.top_k)

    def forward(self, hidden_states):
        batch_size = hidden_states.shape[0]
        gate_idx, gate_score = self.gate(hidden_states)
        # gate_idx: (batch, top_k)
        # gate_score: (batch, top_k)
        
        output = torch.zeros(batch_size, self.hidden_size, device=hidden_states.device)
        
        for k in range(self.top_k):
            expert_indices = gate_idx[:, k]  # (batch,)
            weights = gate_score[:, k]  # (batch,)
            
            for expert_id in range(self.num_experts):
                mask = expert_indices == expert_id
                
                if mask.any():
                    expert_input = hidden_states[mask]
                    expert_output = self.experts[expert_id](expert_input)
                    weight = weights[mask] * self.alpha[expert_id]
                    output[mask] += expert_output * weight.unsqueeze(-1)
        
        return output
        