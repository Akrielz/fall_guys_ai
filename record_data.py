import os
from typing import Optional

import cv2
import numpy as np

from data_utils.data_handler import save_data_training, get_temporary_name, save_data_general
from image_utils.grab_screen import get_screenshot
from key_utils.input_check import input_check
from key_utils.keyboard_handler import keyboard_key_check


def record_data_using_stream(
        x1: int = 0,
        y1: int = 0,
        x2: int = 1920,
        y2: int = 1080,
        directory: str = "data",
        save_width: Optional[int] = None,
        save_height: Optional[int] = None

):
    fps = 30

    recording = False
    file_name = None

    targets = []

    video_stream = None

    used_width = save_width if save_width is not None else x2 - x1
    used_height = save_height if save_height is not None else y2 - y1

    while True:
        frame = get_screenshot(x1, y1, x2, y2, save_width, save_height) if recording else None
        inputs = input_check()
        key = inputs[0]

        if recording:
            video_stream.write(frame)
            targets.append(inputs)

        if key == "1" and not recording:
            print("Started recording")
            recording = True

            file_name = get_temporary_name(directory, return_full_path=True)

            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            video_stream = cv2.VideoWriter(f"{file_name}.avi", fourcc, fps, (used_width, used_height))

        elif key == "2" and recording:
            print("Saving recording")
            recording = False

            video_stream.release()

            save_data_general(targets, f"{file_name}.keys")
            targets = []
            print("Recording saved successfully!")

        elif key == "3" and recording:
            print("Cancel recording")
            recording = False

            video_stream.release()
            os.remove(f"{file_name}.avi")

            targets = []


def record_data_using_ram(
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
        inputs = input_check()
        key = inputs[0]

        if recording:
            video.append(frame)
            targets.append(inputs)

        if key == "1" and not recording:
            print("Started recording")
            recording = True

        elif key == "2" and recording:
            print("Saving recording")
            recording = False

            video.pop()
            targets.pop()

            video = np.array(video)

            save_data_training(video, targets, directory)
            video, targets = [], []
            print("Recording saved successfully!")

        elif key == "3" and recording:
            print("Cancel recording")
            recording = False
            video, targets = [], []


if __name__ == "__main__":
    record_data_using_stream(save_width=192*4, save_height=108*4)
