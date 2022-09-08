import time

import numpy as np
from mss import mss


def get_screenshot(x1: int = 0, y1: int = 0, x2: int = 1920, y2: int = 1080):
    width = x2 - x1
    height = y2 - y1

    bounding_box = {'top': y1, 'left': x1, 'width': width, 'height': height}

    sct = mss()  # BGR

    img_sct = sct.grab(bounding_box)
    img_np = np.array(img_sct)  # BRG

    return img_np


def video_brg_to_rgb(video: np.array):
    return np.concatenate([video[:, :, :, 2:3], video[:, :, :, 1:2], video[:, :, :, 0:1]], -1)


def image_rgb_to_brg(img: np.array):
    return np.concatenate([img[:, :, 2:3], img[:, :, 1:2], img[:, :, 0:1]], -1)  # RGB


if __name__ == "__main__":
    cnt = 0
    start = time.time()

    while cnt < 40:
        img_np = get_screenshot()
        cnt += 1

    stop = time.time()

    print(f"Time: {stop - start}. Captured: {cnt}")
