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

from .modeling_siglip import SiglipModel
from .modeling_utils import ConfigMixin, ModelMixin, register_to_config
from .modules import PatchEmbed


class ShowoSemanticLayers(ModelMixin, ConfigMixin):
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
            patch_size=2,
            hidden_size=2048,
            clip_latent_dim=1024,
            add_time_embeds=True,
            add_qk_norm=False,
            **kwargs,
    ):
        super().__init__()

        self.image_embedder_und = PatchEmbed(
            patch_size=patch_size,
            in_chans=image_latent_dim,
            embed_dim=clip_latent_dim,
        )
        self.reset_parameters()

        # initialize semantic layers from siglip
        siglip_model = SiglipModel.from_pretrained("google/siglip-so400m-patch14-384")
        self.position_embedding = siglip_model.vision_model.embeddings.position_embedding
        self.und_trans = siglip_model.vision_model.encoder
        del self.und_trans.layers[-1]
        self.register_buffer("position_ids",
                             torch.arange(image_latent_height * image_latent_width).expand((1, -1)),
                             persistent=False)

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = True

    def reset_parameters(self):

        w1 = self.image_embedder_und.proj.weight.data
        nn.init.xavier_uniform_(w1.view([w1.shape[0], -1]))
        nn.init.constant_(self.image_embedder_und.proj.bias, 0)

    # to distill patch embedding layers
    def forward(
            self,
            image_latents=None,
            clip_features=None,
            device='cuda:0',
    ):
        image_embeds_und = self.image_embedder_und(image_latents)
        image_embeds_und = image_embeds_und + self.position_embedding(self.position_ids)

        if hasattr(self, 'und_trans'):
            image_embeds_und = self.und_trans(image_embeds_und)['last_hidden_state']

        if clip_features is not None:
            clip_features = clip_features.reshape(-1, self.config.clip_latent_dim)
            image_embeds_und = image_embeds_und.reshape(-1, self.config.clip_latent_dim)

            clip_features = F.normalize(clip_features, dim=-1)
            image_embeds_und = F.normalize(image_embeds_und, dim=-1)

            similarities = torch.sum(clip_features * image_embeds_und, dim=-1)
            clamped_sims = torch.clamp(similarities, 0.0001, 0.9999)
            distill_loss = -torch.log(clamped_sims).mean()

            # if self.config.self_sim_distill:
            # sim_mat_tgt = clip_features @ clip_features.T
            # sim_mat_src = image_embeds_und @ image_embeds_und.T
            # sim_mat_loss = F.mse_loss(sim_mat_src, sim_mat_tgt)

            return image_embeds_und, distill_loss.to(
                image_embeds_und.dtype)  # , sim_mat_loss.to(image_embeds_und.dtype)
        else:
            return image_embeds_und
