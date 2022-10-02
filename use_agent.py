from typing import Optional

import numpy as np
import torch
from einops import rearrange
from torch import nn
from vision_models_playground import models

from image_utils.image_handler import get_screenshot_mss_api, threshold_frame, image_bgr_to_rgb
from key_utils.input_check import input_check
from key_utils.press_keys import keyboard_keys, press_keys
from record_data import print_current_time


def load_agent(
        agent_path: str,
        model: nn.Module,
        device: torch.device = "cpu",
):
    # Load weights
    weights = torch.load(agent_path)

    # Put received model on device
    model.to(device)

    # Load model
    model.load_state_dict(weights)
    # del weights

    # Put model in evaluation mode
    model.eval()

    return model


@torch.no_grad()
def get_agent_action(agent: nn.Module, frame: np.array, device: torch.device = "cpu"):
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
    frame = frame / 255

    # Get actions
    actions = agent(frame)

    # Apply sigmoid
    actions = torch.sigmoid(actions)
    actions = actions >= 0.3

    return actions[0]


def use_agent_image(
        agent_path: str,
        model: nn.Module,
        device: torch.device = "cpu",

        x1: int = 0,
        y1: int = 0,
        x2: int = 1920,
        y2: int = 1080,

        model_width: Optional[int] = None,
        model_height: Optional[int] = None,
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

    print("Agent loaded")

    recording = False

    while True:
        keys = input_check()
        frame = get_screenshot_mss_api(x1, y1, x2, y2, used_width, used_height) if recording else None

        start_recording_key = keys[0]
        stop_recording_key = keys[1]

        if recording:
            actions = get_agent_action(agent, frame, device)
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
    agent_path = "trained_agents/CvT13/2022-10-02_00-55-30.pt"
    # agent_path = "trained_agents/ResNet50/2022-09-27_02-12-36_best.pt"
    model = models.classifiers.build_cvt_13(num_classes=7, in_channels=4)
    # model = models.classifiers.build_resnet_50(num_classes=7, in_channels=4)
    use_agent_image(agent_path, model, torch.device('cuda'), model_width=384, model_height=216)
