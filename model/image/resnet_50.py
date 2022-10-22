from typing import Optional

import torch
from torch import nn
from torchvision.models import resnet50

from eda.keys_frequency import get_key_mapping
from model.utility.load_agent import load_agent


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


def load_resnet_50(
        agent_path,
        device: torch.device = torch.device('cpu'),
        in_channels: int = 4,
        num_classes: Optional[int] = None,
        previous_num_classes: Optional[int] = None,
        **kwargs
):
    if previous_num_classes is None:
        previous_num_classes = num_classes

    model = build_resnet_50(in_channels=in_channels, num_classes=previous_num_classes, **kwargs)
    model = load_agent(agent_path, model, device)

    if num_classes is not None and previous_num_classes != num_classes:
        model.fc = nn.Linear(2048, num_classes)

    return model
