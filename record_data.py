from typing import Optional

import numpy as np

from data_utils.data_handling import save_data_training
from image_utils.grab_screen import get_screenshot
from key_utils.get_keys import key_check


def record_data(
        x1: int = 0,
        y1: int = 0,
        x2: int = 1920,
        y2: int = 1080,
        directory: str = "data",
        save_width: Optional[int] = None,
        save_height: Optional[int] = None
):
    video = []
    targets = []

    recording = False

    while True:
        frame = get_screenshot(x1, y1, x2, y2, save_width, save_height) if recording else None
        key = key_check()

        if recording:
            video.append(frame)
            targets.append(key)

        if key == "1" and not recording:
            print("Recording")
            recording = True

        elif key == "2" and recording:
            print("Save record")
            recording = False

            video.pop()
            targets.pop()

            video = np.array(video)

            save_data_training(video, targets, directory)
            video, targets = [], []

        elif key == "3" and recording:
            print("Cancel record")
            recording = False
            video, targets = [], []


if __name__ == "__main__":
    record_data(save_width=192*5, save_height=108*5)
