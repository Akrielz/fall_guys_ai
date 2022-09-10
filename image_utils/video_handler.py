import time
from typing import Optional, List

import cv2
import numpy as np


def video_bgr_to_rgb(video: np.array):
    return np.concatenate(
        [
            video[:, :, :, 2:3],
            video[:, :, :, 1:2],
            video[:, :, :, 0:1]
        ], -1
    )


def process_keyboard_key(key):
    if key == " ":
        return "space"

    if key == "+":
        return " "

    return key


def process_mouse_key(key):
    if key == "+":
        return " "

    return key


def process_sign(number):
    if number < 0:
        return "-"
    elif number > 0:
        return "+"
    else:
        return " "


def process_mouse_possition(mouse_pos):
    x = mouse_pos[0]
    y = mouse_pos[1]
    return f"{process_sign(x)}{process_sign(y)}"


def view_video(
        video: np.array,
        keys: Optional[List] = None,
        fps: int = 60,
        is_bgr: bool = False,
        font_size: int = 1,
):
    video = video.astype(np.uint8)

    if is_bgr:
        video = video_bgr_to_rgb(video)

    # <keyboard_key, mouse_key, (mouse_pos_x, mouse_pos_y)>
    for i, frame in enumerate(video):
        if keys is not None:
            display_keys(font_size, frame, keys[i])

        cv2.imshow('video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(1 / fps)

    cv2.destroyAllWindows()


def display_keys(font_size, frame, keys):
    process_funcs = [process_keyboard_key, process_mouse_key, process_mouse_possition]

    for i, key in enumerate(keys):
        y = frame.shape[0] - 40
        x = frame.shape[1] * (i + 1) * 2 // 7 - font_size * len(key) * 5 - 100

        cv2.putText(
            frame,
            process_funcs[i](key),
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_size,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )


def save_video(
        video: np.array,
        keys: Optional[List] = None,
        fps: int = 30,
        is_bgr: bool = False,
        font_size: int = 5,
        file_name: str = "video.avi",
):
    video = video.astype(np.uint8)

    if is_bgr:
        video = video_bgr_to_rgb(video)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(file_name, fourcc, fps, (video.shape[2], video.shape[1]))

    for i, frame in enumerate(video):
        if keys is not None:
            key = keys[i]
            y = frame.shape[0] - 40
            x = frame.shape[1] // 2 - font_size * len(key) * 5 // 2

            cv2.putText(frame, key, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), 1, cv2.LINE_AA)

        out.write(frame)

    out.release()


def load_video(file_name: str):
    cap = cv2.VideoCapture(file_name)
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frames.append(frame)

    cap.release()
    return np.array(frames)
