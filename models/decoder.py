import math
from typing import List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from packaging import version
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    get_torch_version,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)

from hydra.utils import instantiate
from .moe_gates import GShardGate
from .phi import PhiModel, PhiMLP, PHI_ATTENTION_CLASSES
from transformers.models.phi.configuration_phi import PhiConfig

if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "microsoft/phi-1"
_CONFIG_FOR_DOC = "PhiConfig"


class FFN(nn.Module):
    pass


class FFNDefault(FFN):
    def __init__(self, config: PhiConfig):
        super().__init__()
        self.dropout = nn.Dropout(config.resid_pdrop)
        self.mlp = PhiMLP(config)
        self.input_layernorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )

    def forward(self, x: torch.Tensor):
        x = self.input_layernorm(x)
        x = self.mlp(x)
        return x


class MoE(FFN):
    def __init__(self, config: PhiConfig):
        super().__init__()
        self.gate = instantiate(config.moe.gate)

        self.num_experts = config.moe.num_experts
        self.hidden_size = config.hidden_size
        self.top_k = config.moe.top_k

        self.experts = nn.ModuleList(
            [FFNDefault(config) for _ in range(self.num_experts)]
        )

        self.alpha = nn.Parameter(torch.randn(self.num_experts))

    def forward(self, hidden_states):
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Reshape to (batch_size * seq_len, hidden_size) for gate processing
        hidden_states_flat = hidden_states.view(-1, hidden_size)
        
        gate_idx, gate_score = self.gate(hidden_states_flat)
        # gate_idx: (batch_size * seq_len, top_k)
        # gate_score: (batch_size * seq_len, top_k)

        output_flat = torch.zeros(batch_size * seq_len, hidden_size, device=hidden_states.device)

        for k in range(self.top_k):
            expert_indices = gate_idx[:, k]  # (batch_size * seq_len,)
            weights = gate_score[:, k]  # (batch_size * seq_len,)

            for expert_id in range(self.num_experts):
                mask = expert_indices == expert_id

                if mask.any():
                    expert_input = hidden_states_flat[mask]
                    expert_output = self.experts[expert_id](expert_input)
                    weight = weights[mask] * self.alpha[expert_id]
                    output_flat[mask] += expert_output * weight.unsqueeze(-1)
        
        # Reshape back to (batch_size, seq_len, hidden_size)
        output = output_flat.view(batch_size, seq_len, hidden_size)
        return output


class PhiDecoderLayer(nn.Module):
    def __init__(self, config: PhiConfig, layer_idx: int, ffn: FFN = None):
        super().__init__()
        self.self_attn = PHI_ATTENTION_CLASSES[config._attn_implementation](
            config, layer_idx=layer_idx
        )
        self.ffn: FFN = ffn or instantiate(config.ffn)
        self.input_layernorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        """
        Args:
            hidden_states (`torch.FloatTensor`):
                input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
                Indices of positions of each input sequence tokens in the position embeddings. Selected in the range
                `[0, config.n_positions - 1]`. [What are position IDs?](../glossary#position-ids)
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_outputs, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        attn_outputs = self.resid_dropout(attn_outputs)
        hidden_states = attn_outputs + residual
        hidden_states = hidden_states + self.ffn(hidden_states)
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)

        return outputs


if __name__ == "__main__":
    hidden_size = 512
    num_experts = 8
    ...
