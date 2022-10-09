import torch
from einops import repeat
from torch import nn


class FakeModel(nn.Module):
    def __init__(self, output: torch.Tensor):
        super(FakeModel, self).__init__()
        self.output = output

    def forward(self, x, **Kwargs):
        batch_size = x.shape[0]

        output = repeat(self.output, "... -> b ...", b=batch_size)
        return output