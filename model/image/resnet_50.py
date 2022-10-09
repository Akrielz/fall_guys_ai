from typing import Optional

from torch import nn
from torchvision.models import resnet50


def build_resnet_50(
        in_channels: int = 4,
        num_classes: Optional[int] = None,
        **kwargs):
    # Create ResNet50 pretrained model from torchvision
    model = resnet50(**kwargs)

    if in_channels != 3:
        model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

    if num_classes is not None:
        model.fc = nn.Linear(2048, num_classes)

    return model
