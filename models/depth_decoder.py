"""
Modified from https://github.com/CompVis/taming-transformers/blob/master/taming/modules/diffusionmodules/model.py
"""
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common_modules import *
from .misc import *

class DepthDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.ch = self.cfg.ch
        self.temb_ch = 0
        self.num_resolutions = len(self.cfg.ch_mult)
        self.num_res_blocks = self.cfg.num_res_blocks
        self.resolution = self.cfg.resolution
        self.in_ch = self.cfg.in_ch

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,) + tuple(self.cfg.ch_mult)
        block_in = self.cfg.ch * self.cfg.ch_mult[self.num_resolutions - 1]
        curr_res = self.resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, self.cfg.z_channels, curr_res, curr_res)
        print(
            "Working with z of shape {} = {} dimensions.".format(
                self.z_shape, np.prod(self.z_shape)
            )
        )

        # z to block_in
        self.conv_in = torch.nn.Conv2d(
            self.cfg.z_channels, block_in, kernel_size=3, stride=1, padding=1
        )

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=self.cfg.dropout,
        )
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=self.cfg.dropout,
        )

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = self.cfg.ch * self.cfg.ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=self.cfg.dropout,
                    )
                )
                block_in = block_out
                if curr_res in self.cfg.attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                if self.cfg.depth_to_space:
                    up.upsample = DepthToSpaceUpsample(block_in)
                else:
                    up.upsample = Upsample(block_in, True)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        if self.cfg.predict_depth:
            self._depth_norm_out = Normalize(block_in)
            self._depth_conv_out = torch.nn.Conv2d(
                block_in, 1, kernel_size=3, stride=1, padding=1
            )

        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in, 1, kernel_size=3, stride=1, padding=1)
        self.post_quant_conv = torch.nn.Conv2d(
            self.cfg.z_channels, self.cfg.z_channels, 1
        )
        for param in self.parameters():
            broadcast(param, src=0)

    def forward(self, z):
        # assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None
        z = self.post_quant_conv(z)

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        h = self.norm_out(h)
        h = nonlinearity(h)
        depth = self.conv_out(h)
        return depth
