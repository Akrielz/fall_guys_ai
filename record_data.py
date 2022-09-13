import os
import threading
from datetime import datetime
from typing import Optional, Tuple, List

import cv2
import numpy as np

from data_utils.data_handler import save_data_training, get_temporary_name, save_data_general
from image_utils.image_handler import get_screenshot
from key_utils.input_check import input_check
from key_utils.mouse_handler import mouse_position_check

got_frame = False

results_thread = [None, None]


def print_current_time():
    get_current_date_time = datetime.now().strftime("%Y %m %d %H:%M:%S")
    print(f"Current date and time: {get_current_date_time}\n")


def get_screenshot_thread_wrap(
        x1: int = 0,
        y1: int = 0,
        x2: int = 1920,
        y2: int = 1080,
        save_width: Optional[int] = None,
        save_height: Optional[int] = None,
):
    global got_frame
    global results_thread

    frame = get_screenshot(x1, y1, x2, y2, save_width, save_height)
    got_frame = True
    results_thread[1] = frame


def get_mouse_pos_thread_wrap():
    global got_frame
    global results_thread

    mouse_positions = []

    while True:
        mouse_position = mouse_position_check()
        mouse_positions.append(mouse_position)

        if got_frame:
            break

    results_thread[0] = mouse_positions


def track_mouse_movement(
        mouse_positions: List[Tuple[int, int]],
        width: int,
        height: int,
) -> np.ndarray:
    mouse_positions = np.array(mouse_positions).astype(np.float32)

    mouse_positions_mask_x = abs(mouse_positions[:, 0] - width/2) >= 1.02
    mouse_positions_mask_y = abs(mouse_positions[:, 1] - height/2) >= 1.02
    mouse_positions_mask = mouse_positions_mask_x | mouse_positions_mask_y

    if not mouse_positions_mask.any():
        return np.array([0.0, 0.0])

    mouse_positions = mouse_positions[mouse_positions_mask]

    mouse_positions[:, 0] = mouse_positions[:, 0] / width
    mouse_positions[:, 1] = mouse_positions[:, 1] / height

    mouse_positions[:, 0] = mouse_positions[:, 0] - 0.5  # [-0.5, 0.5]
    mouse_positions[:, 1] = mouse_positions[:, 1] - 0.5  # [-0.5, 0.5]

    return mouse_positions.sum(axis=0)


def track_mouse_movement_diff(
        mouse_positions: List[Tuple[int, int]],
) -> np.ndarray:
    mouse_positions = np.array(mouse_positions).astype(np.float32)
    return (mouse_positions[1:] - mouse_positions[:-1]).sum(axis=0)


def get_training_data_threads(
        original_width: int,
        original_height: int,
        recording: bool,
        x1: int = 0,
        y1: int = 0,
        x2: int = 1920,
        y2: int = 1080,
        save_width: Optional[int] = None,
        save_height: Optional[int] = None,
) -> Tuple[List[bool], Optional[np.ndarray], Optional[np.ndarray]]:
    if not recording:
        return input_check(), None, None

    global got_frame
    global results_thread

    got_frame = False

    thread_input = threading.Thread(target=get_mouse_pos_thread_wrap)
    thread_screenshot = threading.Thread(
        target=get_screenshot_thread_wrap, args=(x1, y1, x2, y2, save_width, save_height)
    )

    thread_input.start()
    thread_screenshot.start()

    thread_input.join()
    thread_screenshot.join()

    keys = input_check()
    mouse_positions = track_mouse_movement(results_thread[0], original_width, original_height)
    frame = results_thread[1]

    return keys, mouse_positions, frame


def get_training_data(
        original_width: int,
        original_height: int,
        recording: bool,
        x1: int = 0,
        y1: int = 0,
        x2: int = 1920,
        y2: int = 1080,
        save_width: Optional[int] = None,
        save_height: Optional[int] = None,
        is_first_person: bool = False,
) -> Tuple[List[bool], Optional[np.ndarray], Optional[np.ndarray]]:
    if is_first_person:
        return get_training_data_threads(
            original_width, original_height, recording, x1, y1, x2, y2, save_width, save_height
        )

    frame = get_screenshot(x1, y1, x2, y2, save_width, save_height) if recording else None
    keys = input_check()
    mouse_position = np.array(mouse_position_check()) if recording else None

    return keys, mouse_position, frame


def record_data_using_stream(
        x1: int = 0,
        y1: int = 0,
        x2: int = 1920,
        y2: int = 1080,
        directory: str = "data",
        save_width: Optional[int] = None,
        save_height: Optional[int] = None,
        is_first_person: bool = False,
):
    fps = 30

    recording = False
    file_name = None

    targets = []

    video_stream = None

    original_width = x2 - x1
    original_height = y2 - y1

    used_width = save_width if save_width is not None else original_width
    used_height = save_height if save_height is not None else original_height

    while True:
        keys, mouse_positions, frame = get_training_data(
            original_width, original_height,
            recording,
            x1, y1, x2, y2,
            save_width, save_height,
            is_first_person
        )

        inputs = (keys, mouse_positions)

        start_recording_key = keys[0]
        stop_recording_key = keys[1]
        cancel_recording_key = keys[2]

        if recording:
            video_stream.write(frame)
            targets.append(inputs)

        if start_recording_key and not recording:
            print("Started recording")
            print_current_time()
            recording = True

            file_name = get_temporary_name(directory, return_full_path=True)

            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            video_stream = cv2.VideoWriter(f"{file_name}.avi", fourcc, fps, (used_width, used_height))

        elif stop_recording_key and recording:
            print("Saving recording")
            print_current_time()

            recording = False

            video_stream.release()

            save_data_general(targets, f"{file_name}.keys")
            targets = []

            print("Recording saved successfully!")
            print_current_time()

        elif cancel_recording_key and recording:
            print("Cancel recording")
            print_current_time()

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
        save_height: Optional[int] = None,
        is_first_person: bool = False,
):
    video = []
    targets = []

    recording = False

    original_width = x2 - x1
    original_height = y2 - y1

    while True:
        keys, mouse_positions, frame = get_training_data(
            original_width, original_height,
            recording,
            x1, y1, x2, y2,
            save_width, save_height,
            is_first_person
        )

        inputs = (keys, mouse_positions)

        start_recording_key = keys[0]
        stop_recording_key = keys[1]
        cancel_recording_key = keys[2]

        if recording:
            video.append(frame)
            targets.append(inputs)

        if start_recording_key and not recording:
            print("Started recording")
            recording = True

        elif stop_recording_key and recording:
            print("Saving recording")
            recording = False

            video.pop()
            targets.pop()

            video = np.array(video)

            save_data_training(video, targets, directory)
            video, targets = [], []
            print("Recording saved successfully!")

        elif cancel_recording_key and recording:
            print("Cancel recording")
            recording = False
            video, targets = [], []


if __name__ == "__main__":
    record_data_using_stream(is_first_person=True)
