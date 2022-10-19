from torch import nn
from torchvision.transforms import RandomHorizontalFlip, RandomRotation, ColorJitter, GaussianBlur, RandomAffine


class WeakAugmeneter(nn.Module):
    def __init__(
            self,
            background_color: float = 1.0,
            rotation_angles: int = 5

    ):
        super().__init__()

        self.net = nn.Sequential(
            RandomHorizontalFlip(),
            GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            RandomRotation(degrees=rotation_angles, fill=background_color),
            RandomAffine(degrees=0, translate=(0.05, 0.05), fill=background_color),
            ColorJitter(brightness=(0.9, 1.0)),
            ColorJitter(contrast=(0.9, 1.0)),
            ColorJitter(saturation=(0.9, 1.0)),
        )

    def forward(self, x):
        return self.net(x)
