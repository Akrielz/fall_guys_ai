import time
from typing import Optional, List

import cv2
import numpy as np

from draw_utils.keyboard import draw_keyboard
from draw_utils.mouse import draw_mouse
from key_utils.keyboard_handler import inv_binary_key


def video_bgr_to_rgb(video: np.array):
    return np.concatenate(
        [
            video[:, :, :, 2:3],
            video[:, :, :, 1:2],
            video[:, :, :, 0:1]
        ], -1
    )


def process_sign(number):
    if number < 0:
        return "-"
    elif number > 0:
        return "+"
    else:
        return "*"


def process_mouse_possition(mouse_pos):
    x = process_sign(mouse_pos[0])
    y = process_sign(mouse_pos[1])
    return f"{x}{y}"


def display_keys(frame, keys):
    pressed_keys = keys[0]
    mouse_position = keys[1]

    mouse_keys = pressed_keys[-2:]
    keyboard_keys = pressed_keys[:-2]

    height = frame.shape[0]
    width = frame.shape[1]

    mouse_pos = (width * 4 // 5, height * 8 // 10)

    size = min(width, height) // 20

    frame = draw_mouse(
        frame,
        mouse_pos[0], mouse_pos[1],
        size=size,
        inner_color=(0, 0, 0), stroke_color=(249, 47, 138),
        thickness=1,
        left_click=mouse_keys[0],
        right_click=mouse_keys[1],
    )

    keyboard_pos = [width * 1 // 5, height * 8 // 10]
    keyboard_pos[1] += int(size * 0.5)

    keyboard_pressed = [val for val, pressed in zip(inv_binary_key, keyboard_keys) if pressed]

    frame = draw_keyboard(
        frame,
        keyboard_pos[0], keyboard_pos[1],
        size=int(size * (2.5 / 4)),
        inner_color=(0, 0, 0), stroke_color=(249, 47, 138),
        thickness=1,
        keys_pressed=keyboard_pressed
    )

    return frame


def view_video(
        video: np.array,
        keys: Optional[List] = None,
        fps: int = 60,
        is_bgr: bool = False,
):
    video = video.astype(np.uint8)

    if is_bgr:
        video = video_bgr_to_rgb(video)

    for i, frame in enumerate(video):
        if keys is not None:
            frame = display_keys(frame, keys[i])

        cv2.imshow('video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(1 / fps)

    cv2.destroyAllWindows()


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
            frame = display_keys(font_size, frame, keys[i])

        # Add drawing with a mouse
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
