import time

from key_utils.keyboard_handler import keyboard_key_check
from key_utils.mouse_handler import mouse_key_check, mouse_position_check, mouse_position_diff_check


def input_check():
    keyboard_key = keyboard_key_check()
    mouse_key = mouse_key_check()
    mouse_position = mouse_position_check()

    return keyboard_key, mouse_key, mouse_position


if __name__ == "__main__":
    while True:
        keyboard_key, mouse_key, mouse_position = input_check()
        print(keyboard_key, mouse_key, mouse_position)

        time.sleep(0.01)
