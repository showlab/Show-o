import torch
import torch.nn.functional as F


class SoftTopKRouter:
    @staticmethod
    def route(logits: torch.Tensor, top_k: int, temperature: float) -> tuple[torch.Tensor, torch.Tensor]:
        eps = 1e-8
        tau = max(float(temperature), 1e-6)

        y_soft = F.gumbel_softmax(logits, tau=tau, hard=False, dim=-1)

        vals, idxs = y_soft.topk(top_k, dim=-1)
        y_hard_k = torch.zeros_like(y_soft)
        y_hard_k.scatter_(1, idxs, 1.0)

        y_soft_k = y_soft * y_hard_k
        denom = y_soft_k.sum(dim=1, keepdim=True).clamp_min(eps)
        y_soft_k = y_soft_k / denom

        y_st = y_hard_k - y_soft_k.detach() + y_soft_k
        return y_st, idxs





