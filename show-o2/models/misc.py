import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .modules import modulate, RMSNorm

from timm.layers.helpers import to_2tuple
from transformers import AutoTokenizer


def next_token_prediction(logits, labels, vocab_szie):
    return F.cross_entropy(logits[:, :-1].contiguous().view(-1, vocab_szie), labels[:, 1:].contiguous().view(-1),
                           ignore_index=-100)


def velocity_prediction(latents, labels, mask=None):
    if mask is not None:
        loss = F.mse_loss(latents, labels, reduction='none')[mask.bool()]
        return loss.mean()
    else:
        return F.mse_loss(latents, labels)


def prepare_gen_input(prompts, text_tokenizer, num_image_tokens, bos_id, eos_id, boi_id, eoi_id, pad_id, img_pad_id,
                      max_text_len, device):
    batch_text_tokens = []
    batch_modality_positions = []
    batch_text_tokens_null = []
    batch_modality_positions_null = []
    for prompt in prompts:
        text_tokens = text_tokenizer(prompt, add_special_tokens=False)['input_ids'][:(max_text_len)]

        modality_positions = torch.tensor([len(text_tokens) + 1 + 1, num_image_tokens]).unsqueeze(0)
        text_tokens = [bos_id] + text_tokens + [boi_id] + [img_pad_id] * num_image_tokens + \
                      [eoi_id] + [eos_id] + [pad_id] * (max_text_len - len(text_tokens))

        batch_text_tokens.append(torch.tensor(text_tokens))
        batch_modality_positions.append(modality_positions)

        text_tokens_null = []
        modality_positions_null = torch.tensor([len(text_tokens_null) + 1 + 1, num_image_tokens]).unsqueeze(0)
        text_tokens_null = [bos_id] + text_tokens_null + [boi_id] + [img_pad_id] * num_image_tokens + \
                           [eoi_id] + [eos_id] + [pad_id] * (max_text_len - len(text_tokens_null))

        batch_text_tokens_null.append(torch.tensor(text_tokens_null))
        batch_modality_positions_null.append(modality_positions_null)

    batch_text_tokens = torch.stack(batch_text_tokens, dim=0).to(device)
    batch_modality_positions = torch.stack(batch_modality_positions, dim=0).to(device)

    batch_text_tokens_null = torch.stack(batch_text_tokens_null, dim=0).to(device)
    batch_modality_positions_null = torch.stack(batch_modality_positions_null, dim=0).to(device)

    return batch_text_tokens, batch_text_tokens_null, batch_modality_positions, batch_modality_positions_null


def prepare_mixed_modal_gen_input(prompts, nulls, text_tokenizer, num_image_tokens, bos_id, boi_id, eoi_id, pad_id, img_pad_id, device):
    batch_text_tokens = []
    batch_modality_positions = []
    batch_text_tokens_null = []
    batch_modality_positions_null = []
    for prompt, null in zip(prompts, nulls):
        text_tokens = text_tokenizer(prompt, add_special_tokens=False).input_ids
        modality_positions = torch.tensor([len(text_tokens) + 1 + 1, num_image_tokens]).unsqueeze(0)
        text_tokens = [bos_id] + text_tokens + [boi_id] + [img_pad_id] * num_image_tokens + [eoi_id]

        text_tokens_null = text_tokenizer(null, add_special_tokens=False).input_ids
        modality_positions_null = torch.tensor([len(text_tokens_null) + 1 + 1, num_image_tokens]).unsqueeze(0)
        text_tokens_null = [bos_id] + text_tokens_null + [boi_id] + [img_pad_id] * num_image_tokens + [eoi_id]

        len_a = len(text_tokens)
        len_b = len(text_tokens_null)

        max_len = max(len_a, len_b)

        if max_len % 128 != 0:
            max_len = (max_len // 128 + 1) * 128

        num_pads_a = max_len - len_a
        num_pads_b = max_len - len_b

        text_tokens = text_tokens + [pad_id] * num_pads_a
        text_tokens_null = text_tokens_null + [pad_id] * num_pads_b

        batch_text_tokens.append(torch.tensor(text_tokens))
        batch_modality_positions.append(modality_positions)

        batch_text_tokens_null.append(torch.tensor(text_tokens_null))
        batch_modality_positions_null.append(modality_positions_null)

    batch_text_tokens = torch.stack(batch_text_tokens, dim=0).to(device)
    batch_modality_positions = torch.stack(batch_modality_positions, dim=0).to(device)

    batch_text_tokens_null = torch.stack(batch_text_tokens_null, dim=0).to(device)
    batch_modality_positions_null = torch.stack(batch_modality_positions_null, dim=0).to(device)

    return batch_text_tokens, batch_text_tokens_null, batch_modality_positions, batch_modality_positions_null


# def prepare_mixed_modal_gen_input(prompt, text_tokenizer, num_image_tokens, boi_id, eoi_id, img_pad_id, pad_id, device):
#     text_tokens = text_tokenizer(prompt, add_special_tokens=False).input_ids
#
#     modality_positions = torch.Tensor([[len(text_tokens), num_image_tokens]]).long().unsqueeze(0)
#     text_tokens = text_tokens + [img_pad_id] * num_image_tokens + [eoi_id]
#
#     modality_positions_null = torch.Tensor([[2, num_image_tokens]]).long().unsqueeze(0)
#     text_tokens_null = [text_tokens[0]] + [boi_id] + [img_pad_id] * num_image_tokens + [eoi_id]
#
#     len_a = len(text_tokens)
#     len_b = len(text_tokens_null)
#     num_pads_a = max(len_a, len_b) - len_a
#     num_pads_b = max(len_a, len_b) - len_b
#
#     text_tokens += [pad_id] * num_pads_a
#     text_tokens_null += [pad_id] * num_pads_b
#
#     text_tokens = torch.tensor(text_tokens).unsqueeze(0)
#     text_tokens_null = torch.tensor(text_tokens_null).unsqueeze(0)
#
#     return text_tokens.to(device), text_tokens_null.to(device), \
#            modality_positions.to(device), modality_positions_null.to(device)


class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding"""

    def __init__(
            self,
            patch_size=16,
            in_chans=3,
            embed_dim=768,
            kernel_size=None,
            padding=0,
            norm_layer=None,
            flatten=True,
            bias=True,
    ):
        super().__init__()
        kernel_size = kernel_size or patch_size
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.flatten = flatten
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=patch_size, bias=bias
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t, dtype):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size).to(dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, adaln_input):
        shift, scale = self.adaLN_modulation(adaln_input).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class UpdatedVisionTransformer(nn.Module):
    def __init__(self, model, del_last_layer=True):
        super().__init__()
        self.model = model
        if del_last_layer:
            del self.model.transformer.resblocks[-1]

    def forward(self, x: torch.Tensor):
        x = self.model.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.model.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype,
                                                                            device=x.device), x],
                      dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.model.positional_embedding.to(x.dtype)
        x = self.model.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.model.transformer(x)
        x = x.permute(1, 0, 2)[:, 1:]  # LND -> NLD

        return x


class CLIPVisionEncoder(nn.Module):
    def __init__(self, model, del_last_layer=False):
        super().__init__()
        self.model = model
        if del_last_layer:
            del self.model.transformer.resblocks[-1]

    def forward(self, x: torch.Tensor):
        x = self.model.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.model.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype,
                                                                            device=x.device), x],
                      dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.model.positional_embedding.to(x.dtype)
        x = self.model.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.model.transformer(x)
        x = x.permute(1, 0, 2)[:, 1:]  # LND -> NLD

        return x


class SigLipVisionEncoder(nn.Module):
    def __init__(self, model, del_last_layer=True):
        """
        A wrapper for extracting features from the penultimate layer of a vision transformer model.

        Args:
            model: The pre-trained model (e.g., CLIP or SigLIP).
            del_last_layer (bool): Whether to delete the last layer of the vision transformer.
        """
        super().__init__()
        self.model = model

        # Remove the text model (if not needed)
        if hasattr(self.model, "text_model"):
            del self.model.text_model

        # Remove the last layer of the vision transformer
        if del_last_layer and hasattr(self.model.vision_model, "encoder"):
            del self.model.vision_model.encoder.layers[-1]

        # Replace the classification head (if it exists) with an identity layer
        if hasattr(self.model.vision_model, "head"):
            self.model.vision_model.head = nn.Identity()
        if hasattr(self.model.vision_model, "post_layernorm"):
            self.model.vision_model.post_layernorm = nn.Identity()

    def forward(self, x):
        """
        Forward pass to extract features from the penultimate layer.

        Args:
            x: Input image tensor (pixel values).

        Returns:
            Tensor: Features from the penultimate layer.
        """
        return self.model.get_image_features(pixel_values=x)

from transformers.utils import torch_int
def interpolate_pos_encoding(dim: int, position_embedding: torch.nn.Embedding, height: int, width: int,
                             patch_size: int) -> torch.Tensor:
    """
    This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher resolution
    images. This method is also adapted to support torch.jit tracing and no class embeddings.

    Adapted from:
    - https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174-L194, and
    - https://github.com/facebookresearch/dinov2/blob/e1277af2ba9496fbadf7aec6eba56e8d882d1e35/dinov2/models/vision_transformer.py#L179-L211
    """

    num_positions = position_embedding.weight.shape[0]
    patch_pos_embed = position_embedding.weight.unsqueeze(0)

    new_height = height // patch_size
    new_width = width // patch_size

    sqrt_num_positions = torch_int(num_positions**0.5)
    patch_pos_embed = patch_pos_embed.reshape(1, sqrt_num_positions, sqrt_num_positions, dim)
    patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)

    patch_pos_embed = nn.functional.interpolate(
        patch_pos_embed,
        size=(new_height, new_width),
        mode="bicubic",
        align_corners=False,
    )

    patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
    return patch_pos_embed


def get_text_tokenizer(model_path, add_showo_tokens=True, return_showo_token_ids=False, llm_name="qwen2_5"):
    text_tokenizer = AutoTokenizer.from_pretrained(model_path)
    text_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    if add_showo_tokens:
        if llm_name == "llama3":
            text_tokenizer.add_tokens('<|img_start|>')
            text_tokenizer.add_tokens('<|img_end|>')
            text_tokenizer.add_tokens('<|image_pad|>')
            text_tokenizer.add_tokens('<|video_pad|>')
            text_tokenizer.add_tokens('<|vid_start|>')
            text_tokenizer.add_tokens('<|vid_end|>')
            text_tokenizer.add_tokens('<image>')
        elif llm_name == "qwen2_5":
            text_tokenizer.add_tokens('<image>')
            text_tokenizer.add_tokens('<|vid_start|>')
            text_tokenizer.add_tokens('<|vid_end|>')
        else:
            raise NotImplementedError

    if return_showo_token_ids:
        if llm_name == "llama3":
            showo_token_ids = {
                "bos_id": text_tokenizer.get_vocab()["<|begin_of_text|>"],
                "eos_id": text_tokenizer.eos_token_id,
                "boi_id": text_tokenizer.get_vocab()["<|img_start|>"],
                "eoi_id": text_tokenizer.get_vocab()["<|img_end|>"],
                "bov_id": text_tokenizer.get_vocab()["<|vid_start|>"],
                "eov_id": text_tokenizer.get_vocab()["<|vid_end|>"],
                "img_pad_id": text_tokenizer.get_vocab()["<|image_pad|>"],
                "vid_pad_id": text_tokenizer.get_vocab()["<|video_pad|>"],
                "img_id": text_tokenizer.get_vocab()["<image>"],
            }
        elif llm_name == "qwen2_5":
            showo_token_ids = {
                "bos_id": text_tokenizer.get_vocab()["<|im_start|>"],
                "eos_id": text_tokenizer.eos_token_id,
                "boi_id": text_tokenizer.get_vocab()["<|vision_start|>"],
                "eoi_id": text_tokenizer.get_vocab()["<|vision_end|>"],
                "bov_id": text_tokenizer.get_vocab()["<|vid_start|>"],
                "eov_id": text_tokenizer.get_vocab()["<|vid_end|>"],
                "img_pad_id": text_tokenizer.get_vocab()["<|image_pad|>"],
                "vid_pad_id": text_tokenizer.get_vocab()["<|video_pad|>"],
                "img_id": text_tokenizer.get_vocab()["<image>"],
            }
        else:
            raise NotImplementedError

        return text_tokenizer, showo_token_ids

    return text_tokenizer


def get_weight_type(config):
    if config.training.mixed_precision == 'bf16':
        weight_type = torch.bfloat16
    elif config.training.mixed_precision == 'float16':
        weight_type = torch.float16
    else:
        weight_type = torch.float32
    return weight_type
