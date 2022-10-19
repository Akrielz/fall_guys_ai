import pickle
import time
from typing import Optional, List

import cv2
import numpy as np
import torch
from einops.layers.torch import Rearrange
from torch import nn
from tqdm import tqdm

from draw_utils.keyboard import draw_keyboard
from draw_utils.mouse import draw_mouse
from image_utils.weak_image_augmentation import WeakAugmeneter
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
        display_height: Optional[int] = None,
        display_width: Optional[int] = None,
):
    if is_bgr:
        video = video_bgr_to_rgb(video)

    for i, frame in enumerate(video):
        if display_height is not None and display_width is not None:
            frame = cv2.resize(frame, (display_width, display_height))

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


def load_video_iterator(file_name: str, mask: Optional[np.array] = None):
    cap = cv2.VideoCapture(file_name)

    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if mask is not None and mask[i] == 0:
            i += 1
            continue

        i += 1
        yield frame

    cap.release()
    return


def load_video_batch_iterator(
        file_name: str,
        batch_size: int = 1,
        mask: Optional[np.array] = None
):
    frames = []

    for i, frame in enumerate(load_video_iterator(file_name)):
        if mask is not None and mask[i] == 0:
            continue

        frames.append(frame)

        if len(frames) == batch_size:
            yield np.array(frames)
            frames = []

    yield np.array(frames)


def load_video_len(file_name: str):
    cap = cv2.VideoCapture(file_name)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return length


def load_video_len_augmented(file_name: str):
    video_len = load_video_len(file_name)
    augmented_file_name = file_name.replace(".avi", ".aug")

    with open(augmented_file_name, "rb") as f:
        augmented_frames_data = pickle.load(f)

    augmented_len = augmented_frames_data[:, 1].sum()
    total_len = video_len + augmented_len
    return total_len


def augment_image(frame: np.array, augmenter: nn.Module):
    frame = torch.from_numpy(frame).float() / 255
    frame = augmenter(frame)
    frame = frame.numpy()
    frame *= 255
    frame = frame.astype(np.uint8)
    return frame


def load_images_augmented_iterator(
        file_name: str,
        permutation: Optional[np.array] = None,
        random_permutation: bool = False,
        return_augmented: bool = False,
):
    augmenter = nn.Sequential(
        Rearrange("h w c-> 1 c h w"),
        WeakAugmeneter(),
        Rearrange("1 c h w -> h w c"),
    ) if return_augmented else None

    # compute lens
    total_len = load_video_len_augmented(file_name)
    video_len = load_video_len(file_name)

    # read augmented data
    augmented_file_name = file_name.replace(".avi", ".aug")
    with open(augmented_file_name, "rb") as f:
        augmented_frames_data = pickle.load(f)

    original_reference_frames = [i for i in range(video_len)]
    augmented_reference_frames = [frame_index for frame_index, frame_len in augmented_frames_data for _ in range(frame_len)]

    # combine the two lists
    reference_frames = original_reference_frames + augmented_reference_frames

    if permutation is None:
        if random_permutation:
            permutation = np.random.permutation(total_len)
        else:
            permutation = np.arange(total_len)

    # open video
    cap = cv2.VideoCapture(file_name)

    for i in range(total_len):
        index = permutation[i]
        frame_reference = reference_frames[index]

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_reference)
        _, frame = cap.read()
        if index < video_len:
            yield frame, False
        else:
            if return_augmented:
                frame = augment_image(frame, augmenter), True
            yield frame, True


if __name__ == "__main__":
    images_iter = load_images_augmented_iterator("data/big_shots/train/tmp0inn4dtw.avi", random_permutation=True)

    start_time = time.time()
    for image in tqdm(images_iter):
        pass

