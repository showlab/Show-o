import torch.nn as nn
import torch.nn.functional as F

from .common_modules import *
from .misc import *


class Dinov2Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks, dropout):
        super().__init__()
        self.conv2d_in = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            self.blocks.append(
                ResnetBlock(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    temb_channels=out_channels,
                    dropout=dropout,
                )
            )
        self.conv2d_out = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        for param in self.parameters():
            broadcast(param, src=0)

    def forward(self, inputs):
        out = self.conv2d_in(inputs)
        tmpt = None
        for block in self.blocks:
            out = block(out, tmpt)
        out = self.conv2d_out(out)
        return out
