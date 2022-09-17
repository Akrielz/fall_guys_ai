from typing import Tuple

from torch import nn
from torch.nn import functional as F

from model.conv_block_3d import ConvBlock3d


class ResidualBlock3d(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: Tuple[int, int, int] = (3, 3, 3),
            stride: Tuple[int, int, int] = (1, 1, 1),
            padding: Tuple[int, int, int] = (1, 1, 1),
            bias: bool = False
    ):
        super(ResidualBlock3d, self).__init__()
        self.conv1 = ConvBlock3d(
            in_channels,
            out_channels,
            stride=stride,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias
        )
        self.conv2 = ConvBlock3d(
            out_channels,
            out_channels,
            stride=(1, 1, 1),
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
            apply_relu=False
        )

        self.identity_reshape = ConvBlock3d(
            in_channels, out_channels, stride=stride, apply_relu=False, kernel_size=(1, 1, 1), padding=(0, 0, 0)
        ) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)

        out += self.identity_reshape(identity)
        out = F.relu(out)

        return out
