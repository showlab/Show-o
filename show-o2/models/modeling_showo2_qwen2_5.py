# coding=utf-8
# Copyright 2025 NUS Show Lab.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers import AutoConfig
from torch.nn.attention.flex_attention import BlockMask
from .misc import velocity_prediction, next_token_prediction, interpolate_pos_encoding
from .modeling_siglip import SiglipModel
from .modeling_utils import ConfigMixin, ModelMixin, register_to_config
from .modules import DiffusionHeadConfig
from .modules import ModulatedAttentionBlock, RMSNorm, PatchEmbed, TimestepEmbedder, FinalLayer
from .qwen2 import Qwen2ForCausalLM
from .spatial_alignment_adapter import (
    DualRouteSpatialAlignmentAdapter,
    GenerationSpatialAdapter,
)


class Showo2Qwen2_5(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
            self,
            llm_vocab_size=None,
            llm_model_path='',
            load_from_showo=False,
            image_latent_dim=16,
            image_latent_height=16,
            image_latent_width=16,
            video_latent_height=16,
            video_latent_width=16,
            patch_size=2,
            hidden_size=2048,
            clip_latent_dim=1152,
            num_diffusion_layers=10,
            add_time_embeds=True,
            add_qk_norm=False,
            drsa_enabled=False,
            drsa_relation_dim=256,
            drsa_num_heads=8,
            drsa_bias_hidden_dim=128,
            generation_adapter_enabled=False,
            generation_adapter_spatial_dim=604,
            generation_adapter_num_heads=8,
            generation_adapter_bias_hidden_dim=128,
            clip_pretrained_model_path="google/siglip-so400m-patch14-384",
            **kwargs,
    ):
        super().__init__()

        llm_config = AutoConfig.from_pretrained(llm_model_path)
        if load_from_showo:
            self.showo = Qwen2ForCausalLM(llm_config)
        else:
            self.showo = Qwen2ForCausalLM.from_pretrained(llm_model_path, attn_implementation='sdpa')
        self.showo.resize_token_embeddings(llm_vocab_size)

        # patch embedding layer for semantic layers
        self.image_embedder_und = PatchEmbed(
            patch_size=patch_size,
            in_chans=image_latent_dim,
            embed_dim=clip_latent_dim,
        )

        # projector
        self.image_embedder_gen = PatchEmbed(
            patch_size=patch_size,
            in_chans=image_latent_dim,
            embed_dim=hidden_size,
        )

        # initialize semantic layers from siglip
        siglip_model = SiglipModel.from_pretrained(clip_pretrained_model_path)
        self.position_embedding = siglip_model.vision_model.embeddings.position_embedding
        self.und_trans = siglip_model.vision_model.encoder
        del self.und_trans.layers[-1]
        self.register_buffer("image_position_ids",
                             torch.arange(image_latent_height * image_latent_width).expand((1, -1)),
                             persistent=False)

        self.fusion_proj = nn.Sequential(
            RMSNorm(clip_latent_dim + hidden_size),
            nn.Linear(clip_latent_dim + hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.drsa_enabled = bool(drsa_enabled)
        self.drsa_relation_dim = int(drsa_relation_dim)
        self.drsa_adapter = (
            DualRouteSpatialAlignmentAdapter(
                hidden_size=hidden_size,
                relation_dim=drsa_relation_dim,
                num_heads=drsa_num_heads,
                bias_hidden_dim=drsa_bias_hidden_dim,
            ) if self.drsa_enabled else None
        )
        self.generation_adapter_enabled = bool(generation_adapter_enabled)
        self.generation_adapter_active = True
        self.generation_adapter = (
            GenerationSpatialAdapter(
                hidden_size=hidden_size,
                spatial_dim=generation_adapter_spatial_dim,
                num_heads=generation_adapter_num_heads,
                bias_hidden_dim=generation_adapter_bias_hidden_dim,
            ) if self.generation_adapter_enabled else None
        )

        # adjust for diffusion head
        self.diffusion_head_config = DiffusionHeadConfig()
        self.time_embed = TimestepEmbedder(self.diffusion_head_config.hidden_size)
        if hidden_size != self.diffusion_head_config.hidden_size:
            self.diff_proj = nn.Sequential(
                nn.Linear(hidden_size, self.diffusion_head_config.hidden_size),
                nn.GELU(),
                nn.Linear(self.diffusion_head_config.hidden_size, self.diffusion_head_config.hidden_size)
            )
            self.time_embed_proj = nn.Linear(self.diffusion_head_config.hidden_size, hidden_size)
        self.diffusion_head_a = nn.ModuleList(
            [ModulatedAttentionBlock(self.diffusion_head_config, layer_idx) for layer_idx in
             range(num_diffusion_layers)]
        )
        self.diffusion_head_b = FinalLayer(self.diffusion_head_config.hidden_size, patch_size, image_latent_dim)

        self.reset_parameters()

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = True

    def reset_parameters(self):

        # Initialize image embedders
        w1 = self.image_embedder_und.proj.weight.data
        nn.init.xavier_uniform_(w1.view([w1.shape[0], -1]))
        nn.init.constant_(self.image_embedder_und.proj.bias, 0)

        w2 = self.image_embedder_gen.proj.weight.data
        nn.init.xavier_uniform_(w2.view([w2.shape[0], -1]))
        nn.init.constant_(self.image_embedder_gen.proj.bias, 0)

        # Initialize transformer layers for understanding encoding and diffusion head
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        _basic_init(self.und_trans)
        _basic_init(self.fusion_proj)
        _basic_init(self.diffusion_head_a)

        # Initialize timestep embedding MLP
        nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embed.mlp[2].weight, std=0.02)

        # Zero-out output layers
        nn.init.constant_(self.diffusion_head_b.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.diffusion_head_b.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.diffusion_head_b.linear.weight, 0)
        nn.init.constant_(self.diffusion_head_b.linear.bias, 0)

    def unpatchify(self, x, h, w, T=0):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.config.image_latent_dim
        p = self.image_embedder_gen.patch_size[0]
        if T == 0:
            x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
            imgs = x.reshape(shape=(x.shape[0], h * p * w * p, c))
        else:
            x = x.reshape(shape=(x.shape[0], T, h, w, p, p, c))
            imgs = x.reshape(shape=(x.shape[0], T, h * p * w * p, c))
        return imgs

    def forward_und_only(
            self,
            text_tokens=None,
            image_latents=None,
            t=None,
            attention_mask=None,
            text_masks=None,
            image_masks=None,
            text_labels=None,
            image_labels=None,
            modality_positions=None,
            output_hidden_states=True,
            max_seq_len=None,
            device='cuda:0',
            use_generation_adapter_for_understanding=False,
            **kwargs,
    ):
        T = 0
        input_embeds = self.showo.model.embed_tokens(text_tokens)
        dtype = input_embeds.dtype
        if len(image_latents.shape) != 4:
            b, c, T, h, w = image_latents.shape
        else:
            b, c, h, w = image_latents.shape

        if T == 0:
            image_embeds_und = self.image_embedder_und(image_latents.to(dtype))
            image_embeds_gen = self.image_embedder_gen(image_latents.to(dtype))
        else:
            # (B, C, T, H, W) --> (BT, C, H, W)
            image_latents = rearrange(image_latents, 'b c t h w -> (b t) c h w')
            # (BT, C, H, W) --> (BT, L=H/p*W/p, D)
            image_embeds_und = self.image_embedder_und(image_latents.to(dtype))
            image_embeds_und = image_embeds_und.reshape(b, T, -1, self.config.clip_latent_dim)
            image_embeds_und = rearrange(image_embeds_und, 'b t l d -> (b t) l d')

            image_embeds_gen = self.image_embedder_gen(image_latents.to(dtype))
            image_embeds_gen = image_embeds_gen.reshape(b, T, -1, self.config.hidden_size)
            image_embeds_gen = rearrange(image_embeds_gen, 'b t l d -> b (t l) d')

        # go through semantic layers
        p = self.config.patch_size
        h_, w_ = h // p, w // p
        # specific for fixed resolution of 432x432
        if self.position_embedding.weight.shape[0] == self.image_position_ids.shape[-1]:
            image_embeds_und = image_embeds_und + self.position_embedding(self.image_position_ids)
            image_embeds_und = self.und_trans(image_embeds_und)['last_hidden_state']
        # interpolate position embeddings for dynamic resolution
        else:
            image_embeds_und = image_embeds_und + interpolate_pos_encoding(
                self.config.clip_latent_dim,
                self.position_embedding,
                h_,
                w_,
                1,
            )
            image_embeds_und = self.und_trans(image_embeds_und)['last_hidden_state']
        if T != 0:
            image_embeds_und = image_embeds_und.reshape(b, T, image_embeds_und.shape[1], -1)
            image_embeds_und = rearrange(image_embeds_und, 'b t l d -> b (t l) d')

        # spatial (-temporal) fusion
        image_embeds = self.fusion_proj(torch.cat([image_embeds_und, image_embeds_gen], dim=-1))
        drsa_relation_codes = kwargs.get("drsa_relation_codes")
        if T == 0 and self.drsa_adapter is not None and drsa_relation_codes is not None:
            image_embeds = self.drsa_adapter.apply_understanding(
                image_embeds,
                drsa_relation_codes,
                grid_height=h_,
                grid_width=w_,
            )

        time_embeds = self.time_embed(t, dtype)
        if hasattr(self, 'time_embed_proj'):
            time_embeds_proj = self.time_embed_proj(time_embeds)
        else:
            time_embeds_proj = time_embeds

        for i, modality_batch in enumerate(modality_positions):
            for j, (offset, length) in enumerate(modality_batch):
                if self.config.add_time_embeds:
                    input_embeds[i, offset] = time_embeds_proj[i * modality_positions.size(1) + j]
                    # length - 1 because we add 1 to the num_image_tokens when add_time_embeds=True
                    # it's necessary to include :length-1, as sometimes we may skip some idle images when length=0
                    input_embeds[i, offset + 1:offset + 1 + length - 1] = \
                        image_embeds[i * modality_positions.size(1) + j, :max(length - 1, 0)]
                else:
                    input_embeds[i, offset:offset + length] = image_embeds[i * modality_positions.size(1) + j, :length]

        if use_generation_adapter_for_understanding:
            if T != 0:
                raise ValueError(
                    "the generation-adapter understanding bridge supports images only"
                )
            if self.generation_adapter is None:
                raise RuntimeError(
                    "the generation-adapter understanding bridge requires "
                    "generation_adapter_enabled=True"
                )
            if not self.generation_adapter_active:
                raise RuntimeError(
                    "the generation-adapter understanding bridge requires an active adapter"
                )
            input_embeds = self.generation_adapter(
                input_embeds,
                modality_positions,
                grid_height=h_,
                grid_width=w_,
                add_time_embeds=bool(self.config.add_time_embeds),
            )

        outputs = self.showo(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            # position_ids=position_ids,
            output_hidden_states=output_hidden_states
        )

        logits, last_hidden_states = outputs['logits'], outputs['hidden_states'][-1]

        if text_labels is not None:
            loss_ntp = next_token_prediction(logits, text_labels, self.config.llm_vocab_size)
            return logits, loss_ntp
        else:
            return logits

    def forward(
            self,
            text_tokens=None,
            image_latents=None,
            t=None,
            attention_mask=None,
            text_masks=None,
            image_masks=None,
            text_labels=None,
            image_labels=None,
            modality_positions=None,
            first_frame_as_cond=False,
            only_denoise_last_image=False,
            guidance_scale=0.0,
            output_attentions=False,
            output_diffusion_attentions=False,
            output_hidden_states=True,
            max_seq_len=None,
            device='cuda:0',
            **kwargs,
    ):
        T = 0
        if image_latents is None:
            # text-only
            logits = self.showo(input_ids=text_tokens, attention_mask=attention_mask)
            return logits
        else:
            # multimoidal understanding and generatiopn
            input_embeds = self.showo.model.embed_tokens(text_tokens)
            dtype = input_embeds.dtype
            if len(image_latents.shape) != 4:
                b, c, T, h, w = image_latents.shape
            else:
                b, c, h, w = image_latents.shape

            # go through dual-path extraction
            if T == 0:
                image_embeds_und = self.image_embedder_und(image_latents.to(dtype))
                image_embeds_gen = self.image_embedder_gen(image_latents.to(dtype))
            else:
                # (B, C, T, H, W) --> (BT, C, H, W)
                image_latents = rearrange(image_latents, 'b c t h w -> (b t) c h w')
                # (BT, C, H, W) --> (BT, L=H/p*W/p, D)
                image_embeds_und = self.image_embedder_und(image_latents.to(dtype))
                image_embeds_und = image_embeds_und.reshape(b, T, -1, self.config.clip_latent_dim)
                image_embeds_und = rearrange(image_embeds_und, 'b t l d -> (b t) l d')

                image_embeds_gen = self.image_embedder_gen(image_latents.to(dtype))
                image_embeds_gen = image_embeds_gen.reshape(b, T, -1, self.config.hidden_size)
                image_embeds_gen = rearrange(image_embeds_gen, 'b t l d -> b (t l) d')

            # go through semantic layers
            p = self.config.patch_size
            h_, w_ = h // p, w // p
            # specific for fixed resolution of 432x432
            if self.position_embedding.weight.shape[0] == self.image_position_ids.shape[-1]:
                image_embeds_und = image_embeds_und + self.position_embedding(self.image_position_ids)
                image_embeds_und = self.und_trans(image_embeds_und)['last_hidden_state']
            # interpolate position embeddings for dynamic resolution
            else:
                image_embeds_und = image_embeds_und + interpolate_pos_encoding(
                    self.config.clip_latent_dim,
                    self.position_embedding,
                    h_,
                    w_,
                    1,
                )
                image_embeds_und = self.und_trans(image_embeds_und)['last_hidden_state']
            if T != 0:
                image_embeds_und = image_embeds_und.reshape(b, T, image_embeds_und.shape[1], -1)
                image_embeds_und = rearrange(image_embeds_und, 'b t l d -> b (t l) d')

            # spatial (-temporal) fusion
            image_embeds = self.fusion_proj(torch.cat([image_embeds_und, image_embeds_gen], dim=-1))
            drsa_relation_codes = kwargs.get("drsa_relation_codes")
            if T == 0 and self.drsa_adapter is not None and drsa_relation_codes is not None:
                image_embeds = self.drsa_adapter.apply_understanding(
                    image_embeds,
                    drsa_relation_codes,
                    grid_height=h_,
                    grid_width=w_,
                )

            if image_labels is not None:
                if T == 0:
                    image_labels = rearrange(image_labels, 'b c h w -> b (h w) c')
                    image_labels = image_labels.reshape(shape=(b, h_, w_, p, p, c))
                    image_labels = image_labels.reshape(shape=(b, h_ * w_, p * p * c))
                else:
                    # (B, C, T, H/p, W/p)
                    image_labels = rearrange(image_labels, 'b c t h w -> b (t h w) c')
                    image_labels = image_labels.reshape(shape=(b, T, h_, w_, p, p, c))
                    image_labels = image_labels.reshape(shape=(b, T * h_ * w_, p * p * c))

            time_embeds = self.time_embed(t, dtype)
            if hasattr(self, 'time_embed_proj'):
                time_embeds_proj = self.time_embed_proj(time_embeds)
            else:
                time_embeds_proj = time_embeds

            # structure text and image embeddings into sequences
            if image_labels is not None:
                new_image_labels = torch.zeros([b, max_seq_len, p * p * c], device=device, dtype=dtype)
                image_masks = image_masks[:, :, None].repeat(1, 1, p * p * c)

            for i, modality_batch in enumerate(modality_positions):
                for j, (offset, length) in enumerate(modality_batch):
                    if self.config.add_time_embeds:
                        input_embeds[i, offset] = time_embeds_proj[i * modality_positions.size(1) + j]
                        # length - 1 because we add 1 to the num_image_tokens when add_time_embeds=True
                        # it's necessary to include :length-1, as sometimes we may skip some idle images when length=0
                        input_embeds[i, offset + 1:offset + 1 + length - 1] = image_embeds[
                                                                              i * modality_positions.size(1) + j,
                                                                              :max(length - 1, 0)]
                        if image_labels is not None:
                            # mask the position of time embedding
                            image_masks[i, offset] = 0
                            # it's necessary to include :length-1, as sometimes we may skip some idle images when length=0
                            new_image_labels[i, offset + 1:offset + 1 + length - 1] = image_labels[
                                                                                      i * modality_positions.size(
                                                                                          1) + j, :max(length - 1, 0)]
                    else:
                        input_embeds[i, offset:offset + length] = image_embeds[i * modality_positions.size(1) + j,
                                                                  :length]
                        if image_labels is not None:
                            new_image_labels[i, offset:offset + length] = image_labels[
                                                                          i * modality_positions.size(1) + j, :length]

            outputs = self.showo(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
                # position_ids=position_ids,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states
            )

            logits, last_hidden_states = outputs['logits'], outputs['hidden_states'][-1]
            if T == 0 and self.generation_adapter is not None and self.generation_adapter_active:
                last_hidden_states = self.generation_adapter(
                    last_hidden_states,
                    modality_positions,
                    grid_height=h_,
                    grid_width=w_,
                    add_time_embeds=bool(self.config.add_time_embeds),
                )
            elif T == 0 and self.drsa_adapter is not None and drsa_relation_codes is not None:
                last_hidden_states = self.drsa_adapter.apply_generation(
                    last_hidden_states,
                    modality_positions,
                    drsa_relation_codes,
                    grid_height=h_,
                    grid_width=w_,
                    add_time_embeds=bool(self.config.add_time_embeds),
                )

            # diffusion head to predict vector fields
            if hasattr(self, 'diff_proj'):
                last_hidden_states = self.diff_proj(last_hidden_states)
            position_ids = torch.arange(last_hidden_states.shape[1], device=last_hidden_states.device).unsqueeze(0)
            diffusion_attentions = [] if output_diffusion_attentions else None
            for layer in self.diffusion_head_a:
                layer_outputs = layer(hidden_states=last_hidden_states,
                                      adaln_input=time_embeds,
                                      attention_mask=attention_mask,
                                      position_ids=position_ids,
                                      modality_positions=modality_positions,
                                      output_attentions=output_diffusion_attentions,
                                      )
                last_hidden_states = layer_outputs[0]
                if output_diffusion_attentions:
                    diffusion_attentions.append(layer_outputs[1])
            v_pred = self.diffusion_head_b(last_hidden_states, time_embeds, modality_positions)

            # [:v_pred.shape[0]] is the valid image labels (special case for interleaved data training)
            if text_labels is not None and image_labels is not None:
                loss_ntp = next_token_prediction(logits, text_labels, self.config.llm_vocab_size)
                loss_flow = velocity_prediction(v_pred, new_image_labels[:v_pred.shape[0]], image_masks)
                return logits, loss_ntp, loss_flow

            elif image_labels is not None:
                loss_flow = velocity_prediction(v_pred, new_image_labels[:v_pred.shape[0]], image_masks)
                return logits, loss_flow

            elif text_labels is not None:
                loss_ntp = next_token_prediction(logits, text_labels, self.config.llm_vocab_size)
                return logits, loss_ntp

            else:
                v_pred_ = []
                num_imgs = 0
                for i, modality_batch in enumerate(modality_positions):
                    for j, (offset, length) in enumerate(modality_batch):
                        if length == 0:
                            break
                        else:
                            v_pred_.append(v_pred[i, offset:offset + length])
                            num_imgs += 1
                v_pred_ = torch.stack(v_pred_)

                # remove the time embedding
                if self.config.add_time_embeds:
                    v_pred_ = v_pred_[:, 1:, :]

                # unpatchify
                v_pred_ = self.unpatchify(v_pred_, h_, w_, T=T)

                if T == 0:
                    v_pred_ = rearrange(v_pred_, 'i j k -> i k j')
                    v_pred_ = v_pred_.reshape(num_imgs, c, h, w)
                else:
                    v_pred_ = rearrange(v_pred_, 'b t l c -> b c t l')
                    v_pred_ = v_pred_.reshape(num_imgs, c, T, h, w)

                # specific for image-to-video generation
                if first_frame_as_cond:
                    # zero the v-prediction for the first frame
                    v_pred_ = torch.cat([
                        torch.zeros_like(v_pred_)[:, :, :1],
                        v_pred_[:, :, 1:]
                    ], dim=2)

                # specific for mixed-modality generation
                if only_denoise_last_image:
                    if guidance_scale > 0:
                        v_pred_cond, v_pred_uncond = torch.chunk(v_pred_, 2)

                        v_pred_cond = torch.cat([
                            torch.zeros_like(v_pred_cond)[:-1, :, :],
                            v_pred_cond[-1:, :, :]
                        ], dim=0)

                        v_pred_uncond = torch.cat([
                            torch.zeros_like(v_pred_uncond)[:-1, :, :],
                            v_pred_uncond[-1:, :, :]
                        ], dim=0)

                        v_pred_ = torch.cat([v_pred_cond, v_pred_uncond], dim=0)
                    else:
                        v_pred_ = torch.cat([
                            torch.zeros_like(v_pred_)[:-1, :, :],
                            v_pred_[-1:, :, :]
                        ], dim=0)

                if output_attentions and output_diffusion_attentions:
                    return logits, v_pred_, outputs['attentions'], tuple(diffusion_attentions)
                if output_diffusion_attentions:
                    return logits, v_pred_, tuple(diffusion_attentions)
                if output_attentions:
                    return logits, v_pred_, outputs['attentions']
                return logits, v_pred_

    @torch.no_grad()
    def t2i_generate(
            self,
            image_latents=None,
            t=None,
            text_tokens=None,
            attention_mask=None,
            modality_positions=None,
            first_frame_as_cond=False,
            only_denoise_last_image=False,
            max_seq_len=None,
            guidance_scale=0.0,
            **kwargs,
    ):
        kwargs.pop("output_hidden_states", None)
        if guidance_scale > 0.0:
            if t.shape[-1] != text_tokens.shape[0]:
                t_cond, t_uncond = torch.chunk(t, 2)
                t_cond[:-1] = 1.0
                t_uncond[:-1] = 1.0
                t = torch.cat([t_cond, t_uncond])
            _, v = self(text_tokens,
                        image_latents=image_latents,
                        t=t,
                        attention_mask=attention_mask,
                        modality_positions=modality_positions,
                        first_frame_as_cond=first_frame_as_cond,
                        only_denoise_last_image=only_denoise_last_image,
                        guidance_scale=guidance_scale,
                        output_hidden_states=True,
                        max_seq_len=max_seq_len,
                        **kwargs)
            v_cond, v_uncond = torch.chunk(v, 2)
            v = v_uncond + guidance_scale * (v_cond - v_uncond)
            return torch.cat([v, v], dim=0)

        else:
            if t.shape[-1] != text_tokens.shape[0]:
                t[:-1] = 1.0
            _, v = self(text_tokens,
                        image_latents=image_latents,
                        t=t,
                        attention_mask=attention_mask,
                        modality_positions=modality_positions,
                        first_frame_as_cond=first_frame_as_cond,
                        only_denoise_last_image=only_denoise_last_image,
                        guidance_scale=guidance_scale,
                        output_hidden_states=True,
                        max_seq_len=max_seq_len,
                        **kwargs)
            return v

    def t2i_generate_with_attentions(
            self,
            image_latents=None,
            t=None,
            text_tokens=None,
            attention_mask=None,
            modality_positions=None,
            first_frame_as_cond=False,
            only_denoise_last_image=False,
            max_seq_len=None,
            guidance_scale=0.0,
            attention_source="qwen",
            **kwargs,
    ):
        """Predict T2I velocity while retaining differentiable spatial attentions."""
        kwargs.pop("output_hidden_states", None)
        if attention_source not in {"qwen", "diffusion"}:
            raise ValueError(
                f"attention_source must be 'qwen' or 'diffusion', got {attention_source!r}"
            )
        output_qwen_attentions = attention_source == "qwen"
        output_diffusion_attentions = attention_source == "diffusion"
        t = t.clone()
        if guidance_scale > 0.0:
            if t.shape[-1] != text_tokens.shape[0]:
                t_cond, t_uncond = torch.chunk(t, 2)
                t_cond[:-1] = 1.0
                t_uncond[:-1] = 1.0
                t = torch.cat([t_cond, t_uncond])
            _, v, attentions = self(
                text_tokens,
                image_latents=image_latents,
                t=t,
                attention_mask=attention_mask,
                modality_positions=modality_positions,
                first_frame_as_cond=first_frame_as_cond,
                only_denoise_last_image=only_denoise_last_image,
                guidance_scale=guidance_scale,
                output_attentions=output_qwen_attentions,
                output_diffusion_attentions=output_diffusion_attentions,
                output_hidden_states=True,
                max_seq_len=max_seq_len,
                **kwargs,
            )
            v_cond, v_uncond = torch.chunk(v, 2)
            v = v_uncond + guidance_scale * (v_cond - v_uncond)
            return torch.cat([v, v], dim=0), attentions

        if t.shape[-1] != text_tokens.shape[0]:
            t[:-1] = 1.0
        _, v, attentions = self(
            text_tokens,
            image_latents=image_latents,
            t=t,
            attention_mask=attention_mask,
            modality_positions=modality_positions,
            first_frame_as_cond=first_frame_as_cond,
            only_denoise_last_image=only_denoise_last_image,
            guidance_scale=guidance_scale,
            output_attentions=output_qwen_attentions,
            output_diffusion_attentions=output_diffusion_attentions,
            output_hidden_states=True,
            max_seq_len=max_seq_len,
            **kwargs,
        )
        return v, attentions

    def t2i_generate_with_all_attentions(
            self,
            image_latents=None,
            t=None,
            text_tokens=None,
            attention_mask=None,
            modality_positions=None,
            first_frame_as_cond=False,
            only_denoise_last_image=False,
            max_seq_len=None,
            guidance_scale=0.0,
            **kwargs,
    ):
        """Probe-only T2I forward returning Qwen and diffusion-head attentions."""
        kwargs.pop("output_hidden_states", None)
        if guidance_scale != 0.0:
            raise ValueError(
                "t2i_generate_with_all_attentions is a conditional probe API; "
                "run it with guidance_scale=0"
            )
        t = t.clone()
        if t.shape[-1] != text_tokens.shape[0]:
            t[:-1] = 1.0
        _, velocity, qwen_attentions, diffusion_attentions = self(
            text_tokens,
            image_latents=image_latents,
            t=t,
            attention_mask=attention_mask,
            modality_positions=modality_positions,
            first_frame_as_cond=first_frame_as_cond,
            only_denoise_last_image=only_denoise_last_image,
            guidance_scale=guidance_scale,
            output_attentions=True,
            output_diffusion_attentions=True,
            output_hidden_states=True,
            max_seq_len=max_seq_len,
            **kwargs,
        )
        return velocity, qwen_attentions, diffusion_attentions

    @torch.no_grad()
    def mmu_generate(
            self,
            input_embeds=None,
            attention_mask=None,
            max_new_tokens=100,
            temperature=1.0,
            top_k=None,
            do_sample=True,
            eos_token=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        if do_sample and temperature <= 0:
            raise ValueError("temperature must be positive when do_sample=True")
        if top_k is not None and top_k < 0:
            raise ValueError("top_k must be non-negative or None")
        device = input_embeds.device
        dtype = input_embeds.dtype
        result = []
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            # idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            # logits, _ = self(idx_cond)
            logits = self.showo(inputs_embeds=input_embeds, attention_mask=attention_mask.to(dtype))['logits']

            L = attention_mask.shape[-1]
            attention_mask = attention_mask.squeeze()
            attention_mask_a = torch.hstack(
                [
                    attention_mask,  # L, L
                    torch.zeros((L, 1)).to(device) + torch.finfo(logits.dtype).min,
                ]
            )
            attention_mask_b = torch.vstack(
                [
                    attention_mask_a,  # L, L+1
                    torch.hstack([attention_mask[-1, :], torch.tensor([0]).to(device)]).unsqueeze(0),
                ]
            )
            attention_mask = attention_mask_b

            logits = logits[:, -1, :]
            if do_sample:
                logits = logits / temperature
                if top_k is not None and top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            result.append(idx_next[0][0])
            # append sampled index to the running sequence and continue
            idx_next_embeds = self.showo.model.embed_tokens(idx_next)
            input_embeds = torch.cat([input_embeds, idx_next_embeds], dim=1).to(dtype)

            if eos_token is not None and idx_next.cpu() == eos_token:
                break

        return result

    @torch.no_grad()
    def lm_generate(
            self,
            input_ids=None,
            attention_mask=None,
            tokenizer=None,
            max_new_tokens=100,
            boi_token=None,
            temperature=1.0,
            top_k=None,
            top_p=None,
            device=None,
    ):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        generated_tokens = input_ids
        output_tokens = []
        for _ in range(max_new_tokens):
            # Generate the next token
            outputs = self.showo(
                input_ids=torch.tensor([generated_tokens]).to(device),
                attention_mask=attention_mask,
                return_dict=True,
            )
            next_token_logits = outputs.logits[:, -1, :]  # Get logits for the last token

            # Apply temperature scaling
            next_token_logits = next_token_logits / temperature

            # Apply top-k sampling
            if top_k is not None:
                top_k_values, _ = torch.topk(next_token_logits, top_k)
                min_top_k_value = top_k_values[:, -1].unsqueeze(-1)
                next_token_logits = torch.where(
                    next_token_logits < min_top_k_value,
                    torch.full_like(next_token_logits, float("-inf")),
                    next_token_logits,
                )

            # Apply top-p (nucleus) sampling
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = 0
                next_token_logits[sorted_indices[sorted_indices_to_remove]] = float("-inf")

            # Convert logits to probabilities
            probs = F.softmax(next_token_logits, dim=-1)

            # Sample the next token
            next_token = torch.multinomial(probs, num_samples=1).item()

            # Append the next token to the sequence
            generated_tokens.append(next_token)
            output_tokens.append(next_token)

            # Check if the `eos_token_id` is generated
            if next_token == tokenizer.eos_token_id or next_token == boi_token:  # EOS token ID
                break

            # Decode the generated tokens
        generated_text = tokenizer.decode(output_tokens, skip_special_tokens=False)

        return generated_text

    @torch.no_grad()
    def mm_generate(
            self,
            input_ids=None,
            image_latents=None,
            t=None,
            modality_positions=None,
            attention_mask=None,
            tokenizer=None,
            max_new_tokens=100,
            boi_token=None,
            temperature=1.0,
            top_k=None,
            top_p=None,
            device=None,
    ):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        generated_tokens = input_ids
        output_tokens = []
        if attention_mask is not None and type(attention_mask) == BlockMask:
            raise NotImplementedError

        for _ in range(max_new_tokens):
            # Generate the next token
            logits = self.forward_und_only(
                text_tokens=torch.tensor([generated_tokens]).to(device),
                image_latents=image_latents,
                t=t,
                attention_mask=attention_mask,
                modality_positions=modality_positions,
            )

            next_token_logits = logits[:, -1, :]  # Get logits for the last token

            # Apply temperature scaling
            next_token_logits = next_token_logits / temperature

            # Apply top-k sampling
            if top_k is not None:
                top_k_values, _ = torch.topk(next_token_logits, top_k)
                min_top_k_value = top_k_values[:, -1].unsqueeze(-1)
                next_token_logits = torch.where(
                    next_token_logits < min_top_k_value,
                    torch.full_like(next_token_logits, float("-inf")),
                    next_token_logits,
                )

            # Apply top-p (nucleus) sampling
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = 0
                next_token_logits[sorted_indices[sorted_indices_to_remove]] = float("-inf")

            # Convert logits to probabilities
            probs = F.softmax(next_token_logits, dim=-1)

            # Sample the next token
            next_token = torch.multinomial(probs, num_samples=1).item()

            # Append the next token to the sequence
            generated_tokens.append(next_token)
            output_tokens.append(next_token)

            # Check if the `eos_token_id` is generated
            if next_token == tokenizer.eos_token_id or next_token == boi_token:  # EOS token ID
                break

            L = attention_mask.shape[-1]
            attention_mask = attention_mask.squeeze()
            attention_mask_a = torch.hstack(
                [
                    attention_mask,  # L, L
                    torch.zeros((L, 1)).to(device) + torch.finfo(next_token_logits.dtype).min,
                ]
            )
            attention_mask_b = torch.vstack(
                [
                    attention_mask_a,  # L, L + 1
                    torch.hstack([attention_mask[-1, :], torch.tensor([0]).to(device)]).unsqueeze(0),
                ]
            )
            attention_mask = attention_mask_b.to(image_latents.dtype)

            # Decode the generated tokens
        generated_text = tokenizer.decode(output_tokens, skip_special_tokens=False)

        return generated_text
