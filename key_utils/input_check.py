import time

from key_utils.keyboard_handler import keyboard_key_check
from key_utils.mouse_handler import mouse_key_check, mouse_position_check, mouse_position_diff_check


def input_check():
    all_keys = keyboard_key_check()
    all_keys.extend(mouse_key_check())
    return all_keys


if __name__ == "__main__":
    while True:
        all_keys = input_check(), mouse_position_check()
        print(all_keys)

        time.sleep(0.01)
