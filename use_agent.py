from typing import Optional, Dict, Tuple

import numpy as np
import torch
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import nn
from torchvision.transforms import Resize
from vision_models_playground import models

from eda.keys_frequency import get_key_mapping, int_to_list
from image_utils.image_handler import get_screenshot_mss_api, threshold_frame, image_bgr_to_rgb
from key_utils.input_check import input_check
from key_utils.press_keys import keyboard_keys, press_keys
from model.image.resnet_50 import build_resnet_50
from model.utility.load_agent import load_agent
from record_data import print_current_time


def get_state(key_mapping: Dict, action: int):
    for keys, value in key_mapping.items():
        if value == action:
            return keys


def decode_actions(actions: torch.Tensor, key_mapping: Dict[int, int]):
    # choose argmax
    actions = actions.argmax(dim=1).cpu().numpy()

    # get key that has value
    actions = actions[0]

    state = get_state(key_mapping, actions)
    binary_representation = int_to_list(state)

    # convert binary to bool
    binary_representation = [bool(x) for x in binary_representation]

    len_binary = len(binary_representation)
    binary_representation = binary_representation + [False] * (7 - len_binary)

    return binary_representation


@torch.no_grad()
def get_agent_action(
        agent: nn.Module,
        frame: np.array,
        key_mapping: Dict[int, int],
        rescale_layer: Optional[nn.Sequential] = None,
        device: torch.device = "cpu",
):
    frame = process_frame(device, frame, rescale_layer)

    # Get actions
    actions = agent(frame)

    # Apply sigmoid
    actions = decode_actions(actions, key_mapping)

    return actions


def process_frame(device, frame, rescale_layer):
    # Get frames as tensor
    frame = torch.from_numpy(frame).to(device)
    frame = rearrange(frame, 'h w c -> 1 h w c')

    # Rescale frame
    if rescale_layer is not None:
        frame = rescale_layer(frame)
        frame = rearrange(frame, '1 h w c -> h w c')

    # Get numpy array
    frame = frame.cpu().numpy()

    # Get segmentation
    frame_segmented = threshold_frame(frame)

    # Move to torch
    frame_segmented = torch.from_numpy(frame_segmented).to(device)
    frame_segmented = rearrange(frame_segmented, "h w -> 1 1 h w")
    frame = torch.from_numpy(frame).to(device)
    frame = rearrange(frame, "h w c -> 1 c h w")

    # Concatenate
    frame = torch.cat([frame, frame_segmented], dim=1)

    # Convert images to [0, 1]
    frame = frame.float()
    frame = frame / 255.0

    return frame


def use_agent_image(
        agent_path: str,
        model: nn.Module,

        key_mapping: Dict[int, int],

        device: torch.device = "cpu",

        x1: int = 0,
        y1: int = 0,
        x2: int = 1920,
        y2: int = 1080,

        model_width: Optional[int] = None,
        model_height: Optional[int] = None,

        resize_image_size: Optional[Tuple[int, int]] = None,
):
    original_width = x2 - x1
    original_height = y2 - y1

    used_width = model_width if model_width is not None else original_width
    used_height = model_height if model_height is not None else original_height

    agent = load_agent(
        agent_path=agent_path,
        model=model,
        device=device,
    )
    agent.eval()

    rescale_layer = None
    if resize_image_size is not None:
        rescale_layer = nn.Sequential(
            Rearrange('b h w c -> b c h w'),
            Resize(resize_image_size),
            Rearrange('b c h w -> b h w c')
        )
        rescale_layer.to(device)

    print("Agent loaded")

    recording = False

    while True:
        keys = input_check()
        frame = get_screenshot_mss_api(x1, y1, x2, y2, used_width, used_height) if recording else None

        start_recording_key = keys[0]
        stop_recording_key = keys[1]

        if recording:
            actions = get_agent_action(agent, frame, key_mapping, rescale_layer, device)
            print(actions)
            press_keys(keyboard_keys, actions)

        if start_recording_key and not recording:
            print("Agent Started")
            print_current_time()
            recording = True

        elif stop_recording_key and recording:
            print("Agent Stopped")
            print_current_time()
            recording = False
            press_keys(keyboard_keys, [False] * (len(keyboard_keys) + 2))


if __name__ == "__main__":
    agent_path = "trained_agents/the_whirlygig/resnet50_pretrained/2022-10-20_21-23-14/model_last.pt"

    data_dir = 'data/the_whirlygig'
    key_mapping = get_key_mapping(data_dir)
    num_classes = len(key_mapping)
    in_channels = 4

    model = build_resnet_50(weights="IMAGENET1K_V2", in_channels=in_channels, num_classes=num_classes)

    use_agent_image(
        agent_path,
        model,
        key_mapping,
        torch.device('cuda'),
        model_width=384,
        model_height=216,
        resize_image_size=(224, 224)
    )
