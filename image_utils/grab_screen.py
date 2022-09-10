from typing import Optional

import cv2
import numpy as np
from PIL import Image
from mss import mss


def get_screenshot(
        x1: int = 0,
        y1: int = 0,
        x2: int = 1920,
        y2: int = 1080,
        save_width: Optional[int] = None,
        save_height: Optional[int] = None
):
    width = x2 - x1
    height = y2 - y1

    bounding_box = {'top': y1, 'left': x1, 'width': width, 'height': height}

    sct = mss()  # BGRA

    img_sct = sct.grab(bounding_box)
    img_np = np.array(img_sct)  # BGRA
    img_np = img_np[:, :, :3]  # BGR

    if save_width is None and save_height is None:
        return img_np

    img_np = cv2.resize(img_np, dsize=(save_width, save_height), interpolation=cv2.INTER_CUBIC)
    return img_np


def image_rgb_to_bgr(img: np.array):
    return np.concatenate([img[:, :, 2:3], img[:, :, 1:2], img[:, :, 0:1]], -1)  # RGB


if __name__ == "__main__":
    img_np = get_screenshot(save_width=192, save_height=108)
    img_np = image_rgb_to_bgr(img_np)
    img_pil = Image.fromarray(img_np)
    img_pil.show()
