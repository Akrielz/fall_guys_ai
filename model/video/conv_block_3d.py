from typing import Tuple

import torch
from torch import nn


class ConvBlock3d(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            apply_relu: bool = True,
            kernel_size: Tuple[int, int, int] = (3, 3, 3),
            stride: Tuple[int, int, int] = (1, 1, 1),
            padding: Tuple[int, int, int] = (1, 1, 1),
            bias: bool = False
    ):
        super(ConvBlock3d, self).__init__()
        self.sequential = nn.Sequential(
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias
            ),
            nn.BatchNorm3d(out_channels),
            nn.ReLU() if apply_relu else nn.Identity()
        )

    def forward(self, x):
        out = self.sequential(x)
        return out


def main():
    x = torch.randn(1, 4, 32, 64, 64)  # [batch_size, num_channels, time, height, width]
    block = ConvBlock3d(in_channels=4, out_channels=64, stride=(1, 1, 1), kernel_size=(1, 3, 3), padding=(0, 1, 1))
    out = block(x)
    print(out.shape)


if __name__ == "__main__":
    main()