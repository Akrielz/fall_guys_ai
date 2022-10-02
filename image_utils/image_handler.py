from typing import Optional

import cv2
import numpy as np
import pyautogui
import win32con
import win32gui
import win32ui
from PIL import Image, ImageGrab
from mss import mss

sct = mss()


def get_screenshot(
        x1: int = 0,
        y1: int = 0,
        x2: int = 1920,
        y2: int = 1080,
        save_width: Optional[int] = None,
        save_height: Optional[int] = None,
        api='mss'
):
    if api == 'mss':
        return get_screenshot_mss_api(x1, y1, x2, y2, save_width, save_height)

    if api == 'win':
        return get_screenshot_win_api(x1, y1, x2, y2, save_width, save_height)

    if api == 'pil':
        return get_screenshot_pil_api(x1, y1, x2, y2, save_width, save_height)

    if api == 'pyautogui':
        return get_screenshot_pyautogui_api(x1, y1, x2, y2, save_width, save_height)

    else:
        raise ValueError(f'API {api} not supported')


def get_screenshot_pyautogui_api(
        x1: int = 0,
        y1: int = 0,
        x2: int = 1920,
        y2: int = 1080,
        save_width: Optional[int] = None,
        save_height: Optional[int] = None
):
    img = pyautogui.screenshot(region=(x1, y1, x2, y2))

    if save_width is None and save_height is None:
        return img

    img = img.resize((save_width, save_height), Image.ANTIALIAS)
    return np.array(img)


def get_screenshot_pil_api(
        x1: int = 0,
        y1: int = 0,
        x2: int = 1920,
        y2: int = 1080,
        save_width: Optional[int] = None,
        save_height: Optional[int] = None,
):
    img = ImageGrab.grab(bbox=(x1, y1, x2, y2))

    if save_width is None and save_height is None:
        return img

    img = img.resize((save_width, save_height), Image.ANTIALIAS)
    return np.array(img)


def get_screenshot_mss_api(
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

    img_sct = sct.grab(bounding_box)
    img_np = np.array(img_sct)
    img_np = img_np[:, :, :3]

    if save_width is None and save_height is None:
        return img_np

    img_np = cv2.resize(img_np, dsize=(save_width, save_height), interpolation=cv2.INTER_CUBIC)
    return img_np


def image_bgr_to_rgb(img: np.array):
    return np.concatenate([img[:, :, 2:3], img[:, :, 1:2], img[:, :, 0:1]], -1)  # RGB


def show_image(img: np.array):
    img_pil = Image.fromarray(img)
    img_pil.show()


def get_screenshot_win_api(
        x1: int = 0,
        y1: int = 0,
        x2: int = 1920,
        y2: int = 1080,
        save_width: Optional[int] = None,
        save_height: Optional[int] = None
):
    hwin = win32gui.GetDesktopWindow()

    width = x2 - x1
    height = y2 - y1

    hwindc = win32gui.GetWindowDC(hwin)
    srcdc = win32ui.CreateDCFromHandle(hwindc)
    memdc = srcdc.CreateCompatibleDC()
    bmp = win32ui.CreateBitmap()
    bmp.CreateCompatibleBitmap(srcdc, width, height)
    memdc.SelectObject(bmp)
    memdc.BitBlt((0, 0), (width, height), srcdc, (x1, y1), win32con.SRCCOPY)

    signedIntsArray = bmp.GetBitmapBits(True)
    img = np.fromstring(signedIntsArray, dtype='uint8')
    img.shape = (height, width, 4)

    srcdc.DeleteDC()
    memdc.DeleteDC()
    win32gui.ReleaseDC(hwin, hwindc)
    win32gui.DeleteObject(bmp.GetHandle())

    if save_width is None and save_height is None:
        return img

    img = cv2.resize(img, dsize=(save_width, save_height), interpolation=cv2.INTER_CUBIC)
    return img


def threshold_frame(frame: np.ndarray):
    black_white_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    segmented_frame = cv2.Canny(black_white_frame, threshold1=119, threshold2=250)

    return segmented_frame


if __name__ == "__main__":
    img = get_screenshot(0, 0, 1920, 1080, 1920, 1080, api='mss')
    img = threshold_frame(img)
    # img = image_bgr_to_rgb(img)
    show_image(img)