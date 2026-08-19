"""Geometry-aware building blocks for the DRSA-Adapter.

Coordinates affect attention geometry only. They are never added to image
token values or residual hidden states.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class SpatialAttentionConfig:
    hidden_size: int = 1536
    relation_dim: int = 256
    num_heads: int = 8
    bias_hidden_dim: int = 128
    rope_base: float = 10_000.0
    attention_dropout: float = 0.0

    def validate(self) -> None:
        if self.hidden_size <= 0 or self.relation_dim <= 0:
            raise ValueError("hidden_size and relation_dim must be positive")
        if self.num_heads <= 0 or self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        head_dim = self.hidden_size // self.num_heads
        if head_dim % 4 != 0:
            raise ValueError("head_dim must be divisible by four for axial 2D RoPE")
        if self.bias_hidden_dim <= 0:
            raise ValueError("bias_hidden_dim must be positive")
        if self.rope_base <= 1.0:
            raise ValueError("rope_base must be greater than one")
        if not 0.0 <= self.attention_dropout < 1.0:
            raise ValueError("attention_dropout must be in [0, 1)")


def _rotate_half(value: torch.Tensor) -> torch.Tensor:
    first, second = value.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class AxialRoPE2D(nn.Module):
    """Apply independent rotary phases to horizontal and vertical Q/K bands."""

    def __init__(self, head_dim: int, base: float = 10_000.0):
        super().__init__()
        if head_dim % 4 != 0:
            raise ValueError("head_dim must be divisible by four")
        if base <= 1.0:
            raise ValueError("base must be greater than one")
        self.head_dim = head_dim
        self.axis_dim = head_dim // 2
        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(0, self.axis_dim, 2, dtype=torch.float32)
                / self.axis_dim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _axis_cos_sin(
        self,
        positions: torch.Tensor,
        *,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        frequencies = torch.outer(positions.float(), self.inv_freq.float())
        phases = torch.cat((frequencies, frequencies), dim=-1)
        return phases.cos().to(dtype), phases.sin().to(dtype)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        *,
        grid_height: int,
        grid_width: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if query.shape != key.shape or query.ndim != 4:
            raise ValueError("query and key must share shape [batch, heads, tokens, dim]")
        if query.shape[-1] != self.head_dim:
            raise ValueError("query head dimension does not match the RoPE configuration")
        token_count = grid_height * grid_width
        if grid_height <= 0 or grid_width <= 0 or query.shape[-2] != token_count:
            raise ValueError("grid_height * grid_width must equal the token count")

        y_positions = torch.arange(grid_height, device=query.device).repeat_interleave(grid_width)
        x_positions = torch.arange(grid_width, device=query.device).repeat(grid_height)
        cos_x, sin_x = self._axis_cos_sin(x_positions, dtype=query.dtype)
        cos_y, sin_y = self._axis_cos_sin(y_positions, dtype=query.dtype)
        cos_x = cos_x[None, None, :, :]
        sin_x = sin_x[None, None, :, :]
        cos_y = cos_y[None, None, :, :]
        sin_y = sin_y[None, None, :, :]

        query_x, query_y = query.split(self.axis_dim, dim=-1)
        key_x, key_y = key.split(self.axis_dim, dim=-1)
        query_x = query_x * cos_x + _rotate_half(query_x) * sin_x
        key_x = key_x * cos_x + _rotate_half(key_x) * sin_x
        query_y = query_y * cos_y + _rotate_half(query_y) * sin_y
        key_y = key_y * cos_y + _rotate_half(key_y) * sin_y
        return torch.cat((query_x, query_y), dim=-1), torch.cat((key_x, key_y), dim=-1)


class RelationConditionedRelativeBias(nn.Module):
    """Continuous relative geometry whose coefficients come from relation code."""

    feature_dim = 12

    def __init__(self, relation_dim: int, num_heads: int, hidden_dim: int):
        super().__init__()
        self.relation_dim = relation_dim
        self.num_heads = num_heads
        self.coefficient_mlp = nn.Sequential(
            nn.LayerNorm(relation_dim),
            nn.Linear(relation_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_heads * self.feature_dim),
        )
        nn.init.zeros_(self.coefficient_mlp[-1].weight)
        nn.init.zeros_(self.coefficient_mlp[-1].bias)

    @staticmethod
    def relative_features(
        grid_height: int,
        grid_width: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if grid_height <= 0 or grid_width <= 0:
            raise ValueError("grid dimensions must be positive")
        y, x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, grid_height, device=device),
            torch.linspace(-1.0, 1.0, grid_width, device=device),
            indexing="ij",
        )
        coordinates = torch.stack((x.reshape(-1), y.reshape(-1)), dim=-1)
        delta = coordinates[:, None, :] - coordinates[None, :, :]
        dx, dy = delta.unbind(dim=-1)
        abs_x, abs_y = dx.abs(), dy.abs()
        radius = torch.sqrt(dx.square() + dy.square() + 1e-12)
        features = torch.stack(
            (
                dx,
                dy,
                abs_x,
                abs_y,
                dx * dy,
                dx.square(),
                dy.square(),
                radius,
                torch.log1p(abs_x),
                torch.log1p(abs_y),
                torch.sign(dx),
                torch.sign(dy),
            ),
            dim=-1,
        )
        return features.to(dtype=dtype)

    def forward(
        self,
        relation_code: torch.Tensor,
        *,
        grid_height: int,
        grid_width: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if relation_code.ndim != 2 or relation_code.shape[-1] != self.relation_dim:
            raise ValueError("relation_code must have shape [batch, relation_dim]")
        features = self.relative_features(
            grid_height,
            grid_width,
            device=relation_code.device,
            dtype=relation_code.dtype,
        )
        coefficients = self.coefficient_mlp(relation_code)
        coefficients = coefficients.view(-1, self.num_heads, self.feature_dim)
        bias = torch.einsum("bhf,ijf->bhij", coefficients, features)
        return bias.to(dtype=dtype)


class RelationConditionedSpatialAttention(nn.Module):
    """Zero-initialized residual attention for DRSA understanding/generation routes."""

    def __init__(self, config: SpatialAttentionConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.head_dim = config.hidden_size // config.num_heads
        self.norm = nn.LayerNorm(config.hidden_size)
        self.qkv = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=False)
        self.rope = AxialRoPE2D(self.head_dim, config.rope_base)
        self.relative_bias = RelationConditionedRelativeBias(
            config.relation_dim,
            config.num_heads,
            config.bias_hidden_dim,
        )
        self.output = nn.Linear(config.hidden_size, config.hidden_size)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        image_tokens: torch.Tensor,
        relation_code: torch.Tensor,
        *,
        grid_height: int,
        grid_width: int,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if image_tokens.ndim != 3 or image_tokens.shape[-1] != self.config.hidden_size:
            raise ValueError("image_tokens must have shape [batch, tokens, hidden_size]")
        if relation_code.shape[0] != image_tokens.shape[0]:
            raise ValueError("image_tokens and relation_code batch sizes must match")
        batch, token_count, _ = image_tokens.shape
        if token_count != grid_height * grid_width:
            raise ValueError("grid shape must match image token count")

        qkv = self.qkv(self.norm(image_tokens))
        query, key, value = qkv.chunk(3, dim=-1)
        query = query.view(batch, token_count, self.config.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch, token_count, self.config.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch, token_count, self.config.num_heads, self.head_dim).transpose(1, 2)
        query, key = self.rope(
            query,
            key,
            grid_height=grid_height,
            grid_width=grid_width,
        )
        logits = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        logits = logits + self.relative_bias(
            relation_code,
            grid_height=grid_height,
            grid_width=grid_width,
            dtype=logits.dtype,
        )
        attention = F.softmax(logits.float(), dim=-1).to(logits.dtype)
        attention = F.dropout(
            attention,
            p=self.config.attention_dropout,
            training=self.training,
        )
        update = torch.matmul(attention, value)
        update = update.transpose(1, 2).reshape(batch, token_count, self.config.hidden_size)
        output = image_tokens + self.output(update)
        if return_attention:
            return output, attention
        return output


class GenerationSpatialAdapter(nn.Module):
    """Prompt-aware generation adapter without a separate relation encoder.

    Qwen hidden states already contain the generation prompt. A learned static
    spatial code keeps the 2D relative-bias capacity without introducing an
    independently trained text relation bottleneck.
    """

    def __init__(self, hidden_size: int, spatial_dim: int = 604, num_heads: int = 8,
                 bias_hidden_dim: int = 128):
        super().__init__()
        config = SpatialAttentionConfig(
            hidden_size=hidden_size,
            relation_dim=spatial_dim,
            num_heads=num_heads,
            bias_hidden_dim=bias_hidden_dim,
        )
        self.spatial = RelationConditionedSpatialAttention(config)
        self.spatial_code = nn.Parameter(torch.zeros(1, spatial_dim))

    def forward(self, hidden_states, modality_positions, *, grid_height, grid_width,
                add_time_embeds):
        if hidden_states.ndim != 3 or modality_positions.ndim != 3:
            raise ValueError("hidden_states must be [B,L,H] and modality_positions [B,M,2]")
        batch, modalities, _ = modality_positions.shape
        expected_tokens = grid_height * grid_width
        codes = self.spatial_code.to(dtype=hidden_states.dtype).expand(batch * modalities, -1)
        rows = []
        for i, modality_batch in enumerate(modality_positions.detach().cpu().tolist()):
            row = hidden_states[i:i + 1]
            pieces = []
            cursor = 0
            for j, (offset, length) in enumerate(modality_batch):
                offset, length = int(offset), int(length)
                token_start = offset + 1 if add_time_embeds else offset
                token_count = length - 1 if add_time_embeds else length
                if token_count <= 0:
                    continue
                if token_count != expected_tokens:
                    raise ValueError("image token count does not match the adapter grid")
                token_end = token_start + token_count
                pieces.append(row[:, cursor:token_start])
                current = self.spatial(
                    row[:, token_start:token_end],
                    codes[i * modalities + j:i * modalities + j + 1],
                    grid_height=grid_height,
                    grid_width=grid_width,
                )
                pieces.append(current)
                cursor = token_end
            pieces.append(row[:, cursor:])
            rows.append(torch.cat(pieces, dim=1))
        return torch.cat(rows, dim=0)


class DualRouteSpatialAlignmentAdapter(nn.Module):
    """Separate U/G residual adapters used at Show-o2 route boundaries."""

    def __init__(self, hidden_size: int, relation_dim: int = 256, num_heads: int = 8,
                 bias_hidden_dim: int = 128):
        super().__init__()
        config = SpatialAttentionConfig(hidden_size=hidden_size, relation_dim=relation_dim,
                                        num_heads=num_heads, bias_hidden_dim=bias_hidden_dim)
        self.relation_dim = relation_dim
        self.understanding = RelationConditionedSpatialAttention(config)
        self.generation = RelationConditionedSpatialAttention(config)
        self.relation_encoder = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, relation_dim),
            nn.SiLU(),
            nn.Linear(relation_dim, relation_dim),
        )
        # Legacy checkpoints may contain a second serial G adapter. New training
        # must use generation only; this optional branch remains load-only so
        # historical Mask/Flow/K4 checkpoints can still be evaluated exactly.
        self._spatial_config = config
        self.generation_spatial: RelationConditionedSpatialAttention | None = None

    def enable_generation_spatial(self) -> RelationConditionedSpatialAttention:
        """Create the legacy second G route for historical checkpoint loading."""
        if self.generation_spatial is None:
            reference = next(self.generation.parameters())
            spatial = RelationConditionedSpatialAttention(self._spatial_config)
            spatial.to(device=reference.device, dtype=reference.dtype)
            self.generation_spatial = spatial
        return self.generation_spatial

    def encode_relation(self, text_features, attention_mask=None):
        if text_features.ndim != 3:
            raise ValueError("text_features must have shape [B, tokens, hidden_size]")
        if attention_mask is None:
            pooled = text_features.mean(dim=1)
        else:
            weights = attention_mask.to(device=text_features.device, dtype=text_features.dtype)
            if weights.ndim != 2 or weights.shape != text_features.shape[:2]:
                raise ValueError("attention_mask must have shape [B, tokens]")
            denominator = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
            pooled = (text_features * weights.unsqueeze(-1)).sum(dim=1) / denominator
        return self.relation_encoder(pooled)

    def _flat_codes(self, relation_codes, batch, modalities, device, dtype):
        if relation_codes is None:
            return None
        if not torch.is_tensor(relation_codes) or relation_codes.shape[-1] != self.relation_dim:
            raise ValueError("drsa_relation_codes must be a tensor with the configured last dimension")
        if relation_codes.ndim == 2 and relation_codes.shape[0] == batch * modalities:
            flat = relation_codes
        elif relation_codes.ndim == 2 and relation_codes.shape[0] == batch:
            flat = relation_codes[:, None, :].expand(batch, modalities, -1).reshape(batch * modalities, -1)
        elif relation_codes.ndim == 3 and relation_codes.shape[:2] == (batch, modalities):
            flat = relation_codes.reshape(batch * modalities, -1)
        else:
            raise ValueError("relation codes must have shape [B,D], [B*M,D], or [B,M,D]")
        return flat.to(device=device, dtype=dtype)

    def apply_understanding(self, image_tokens, relation_codes, *, grid_height, grid_width):
        if relation_codes is None:
            return image_tokens
        if relation_codes.ndim == 3:
            flat = relation_codes.reshape(-1, self.relation_dim)
            if flat.shape[0] != image_tokens.shape[0]:
                raise ValueError("3D relation codes must flatten to the image batch")
            flat = flat.to(device=image_tokens.device, dtype=image_tokens.dtype)
        else:
            flat = self._flat_codes(relation_codes, image_tokens.shape[0], 1,
                                    image_tokens.device, image_tokens.dtype)
        return self.understanding(image_tokens, flat, grid_height=grid_height, grid_width=grid_width)

    def apply_generation(self, hidden_states, modality_positions, relation_codes, *,
                         grid_height, grid_width, add_time_embeds):
        if relation_codes is None:
            return hidden_states
        if hidden_states.ndim != 3 or modality_positions.ndim != 3:
            raise ValueError("hidden_states must be [B,L,H] and modality_positions [B,M,2]")
        batch, modalities, _ = modality_positions.shape
        flat = self._flat_codes(relation_codes, batch, modalities,
                                hidden_states.device, hidden_states.dtype)
        expected_tokens = grid_height * grid_width
        rows = []
        for i, modality_batch in enumerate(modality_positions.detach().cpu().tolist()):
            row = hidden_states[i:i + 1]
            pieces = []
            cursor = 0
            for j, (offset, length) in enumerate(modality_batch):
                offset, length = int(offset), int(length)
                token_start = offset + 1 if add_time_embeds else offset
                token_count = length - 1 if add_time_embeds else length
                if token_count <= 0:
                    continue
                if token_count != expected_tokens:
                    raise ValueError("image token count does not match the adapter grid")
                token_end = token_start + token_count
                pieces.append(row[:, cursor:token_start])
                current = row[:, token_start:token_end]
                current = self.generation(
                    current, flat[i * modalities + j:i * modalities + j + 1],
                    grid_height=grid_height, grid_width=grid_width)
                if self.generation_spatial is not None:
                    current = self.generation_spatial(
                        current,
                        flat[i * modalities + j:i * modalities + j + 1],
                        grid_height=grid_height,
                        grid_width=grid_width,
                    )
                pieces.append(current)
                cursor = token_end
            pieces.append(row[:, cursor:])
            rows.append(torch.cat(pieces, dim=1))
        return torch.cat(rows, dim=0)
