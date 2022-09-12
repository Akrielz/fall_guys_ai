import numpy as np

from draw_utils.general import draw_arc, draw_rectangle, draw_line
from image_utils.image_handler import show_image


def compute_mouse_box(x, y, size):
    # compute left and right points
    left_x = x - size
    right_x = x + size

    # compute top and bottom points
    top_y = y + size
    bottom_y = y + int(size * 2.5)

    return left_x, top_y, right_x, bottom_y


def draw_mouse_left_click(frame, x, y, size, color):
    # draw arc from 180 to 360
    frame = draw_arc(frame, x, y, size, 180, 270, color, -1)

    # draw on the left side a filled rectangle
    frame = draw_rectangle(frame, x - size, y, x, y + size // 2, color, -1)

    return frame


def draw_mouse_right_click(frame, x, y, size, color):
    # draw arc from 180 to 360
    frame = draw_arc(frame, x, y, size, 270, 360, color, -1)

    # draw on the left side a filled rectangle
    frame = draw_rectangle(frame, x, y, x + size, y + size // 2, color, -1)

    return frame


def draw_mouse_inside(frame, x, y, size, color):
    # compute left and right points
    left_x = x - size
    right_x = x + size

    # draw arc from 180 to 360
    frame = draw_arc(frame, x, y, size, 180, 360, color, -1)
    # draw rectangle
    frame = draw_rectangle(frame, left_x, y, right_x, y + int(size * 1.5), color, -1)

    # draw a final arc at the bottom
    frame = draw_arc(frame, x, y + int(size * 1.5), size, 0, 180, color, -1)
    return frame


def draw_mouse_stoke(frame, x, y, size, color, thickness=1):
    # compute left and right points
    left_x = x - size
    right_x = x + size

    # draw arc from 180 to 360
    frame = draw_arc(frame, x, y, size, 180, 360, color, thickness)
    # draw lines at the 2 ends of the arc

    frame = draw_line(frame, left_x, y, left_x, y + int(size * 1.5), color, thickness)
    frame = draw_line(frame, right_x, y, right_x, y + int(size * 1.5), color, thickness)

    # draw line from the bottom of the arc to the center
    frame = draw_line(frame, x, y - size, x, y, color, thickness)

    # draw a final arc at the bottom
    frame = draw_arc(frame, x, y + int(size * 1.5), size, 0, 180, color, thickness)

    return frame


def draw_mouse_inner_arc(frame, x, y, size, inner_color, stroke_color, thickness=1):
    frame = draw_arc(frame, x, y + size // 4, size // 4, 180, 360, inner_color, -1)
    frame = draw_arc(frame, x, y + size // 4, size // 4, 180, 360, stroke_color, thickness)

    # draw rectangle under arc
    frame = draw_rectangle(frame, x - size // 4, y + size // 4, x + size // 4, y + size // 2, inner_color, -1)
    return frame


def draw_mouse(frame, x, y, size, inner_color, stroke_color, thickness=1, left_click=False, right_click=False):
    frame = draw_mouse_inside(frame, x, y, size, inner_color)
    frame = draw_mouse_stoke(frame, x, y, size, stroke_color, thickness)

    if left_click:
        frame = draw_mouse_left_click(frame, x, y, size, stroke_color)

    if right_click:
        frame = draw_mouse_right_click(frame, x, y, size, stroke_color)

    frame = draw_mouse_inner_arc(frame, x, y, size, inner_color, stroke_color, thickness)

    return frame


if __name__ == "__main__":
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    frame = draw_mouse(frame, 100, 100, 50, (0, 0, 0), (255, 255, 255), 3, left_click=True, right_click=True)
    show_image(frame)
