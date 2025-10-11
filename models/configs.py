from transformers.utils.hub import PushToHubMixin
from transformers.models.phi.configuration_phi import PhiConfig


class MLPConfig(PushToHubMixin):
    def __init__(self, hidden_size):
        self.hidden_size = hidden_size


class MoEConfig(MLPConfig):
    def __init__(self, hidden_size, num_experts, top_k):
        super().__init__(hidden_size)
        self.num_experts = num_experts
        self.top_k = top_k


# class PhiMoEConfig(PhiConfig):
#     def __init__(self, num_experts, top_k, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.moe = MoEConfig(num_experts, top_k)
