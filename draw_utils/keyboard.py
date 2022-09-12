from typing import Optional, List

import numpy as np

from draw_utils.general import draw_rectangle, draw_text
from image_utils.image_handler import show_image


keys_width = {
    'TAB': 1.8,
    'CAPS': 2.0,
    'LSHIFT': 2.3,
    'RSHIFT': 3.11,
    'LCTRL': 2.0,
    'RCTRL': 2.0,
    'ALT': 1.4,
    'SPACE': 7.2,
    'ENTER': 2.4,
    'BACK': 2.4,
    'WIN': 1.4,
    "\\": 1.6,
}


def keyboard_layoput():
    keys = [
        ['`', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '-', '=', 'Back'],
        ['TAB', 'q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p', '[', ']', '\\'],
        ['CAPS', 'a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l', ';', "'", "ENTER"],
        ['LSHIFT', 'z', 'x', 'c', 'v', 'b', 'n', 'm', ',', '.', '/', 'RSHIFT'],
        ['LCTRL', 'WIN', 'ALT', 'SPACE', 'ALT', 'RCTRL']
    ]

    keys = list(map(lambda x: list(map(lambda y: y.upper(), x)), keys))
    return keys


def draw_key(frame, x, y, width, height, inner_color, stroke_color, key, thickness=1, key_pressed=False):
    color_key = stroke_color if not key_pressed else inner_color
    color_back = inner_color if not key_pressed else stroke_color

    frame = draw_rectangle(frame, x, y, x + width, y + height, color_back, -1)
    frame = draw_rectangle(frame, x, y, x + width, y + height, stroke_color, thickness)

    size = min(width, height)

    font_size_approximation = size // 2
    middle_y = font_size_approximation // 2
    middle_x = font_size_approximation // 3

    x_offset = x + width // 2 - len(key) * middle_x + size // 10
    y_offset = y + middle_y + font_size_approximation - size // 10

    frame = draw_text(frame, key, x_offset, y_offset, color_key, thickness, size/70)
    return frame


def draw_keyboard(
        frame,
        x_orig,
        y_orig,
        size,
        inner_color,
        stroke_color,
        thickness=1,
        keys_pressed: Optional[List[str]] = None,
):
    if keys_pressed is None:
        keys_pressed = []

    y_top = y_orig - size * 2

    y = y_top
    x = x_orig

    keys = keyboard_layoput()
    for row in keys:
        for key in row:
            width = int(keys_width.get(key, 1) * size)

            key_pressed = key in keys_pressed

            frame = draw_key(frame, x, y, width, size, inner_color, stroke_color, key, thickness, key_pressed)

            x += width
        y += size
        x = x_orig

    return frame


if __name__ == "__main__":
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    frame = draw_keyboard(frame, 100, 200, 100, (255, 255, 255), (0, 0, 0), 4, keys_pressed=["W", "A", "S", "D"])
    show_image(frame)