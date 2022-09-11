import win32api as wapi
import win32con

left_click_code = 0x01
right_click_code = 0x02

click_codes = [left_click_code, right_click_code]
click_names = ["left_click", "right_click"]
click_initials = ["L", "R"]

last_mouse_position = None


def mouse_position_check():
    return wapi.GetCursorPos()


def mouse_position_diff_check():
    global last_mouse_position

    current_position = mouse_position_check()

    if last_mouse_position is None:
        last_mouse_position = current_position
        return 0, 0

    x_diff = current_position[0] - last_mouse_position[0]
    y_diff = current_position[1] - last_mouse_position[1]
    last_mouse_position = current_position

    return x_diff, y_diff


def mouse_key_check():
    return [True if wapi.GetAsyncKeyState(key) else False for key in click_codes]


def move_mouse_relative(x, y):
    wapi.mouse_event(win32con.MOUSEEVENTF_MOVE, x, y, 0, 0)


def move_mouse_absolute(x, y):
    wapi.SetCursorPos((x, y))