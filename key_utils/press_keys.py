from typing import List

import keyboard

from key_utils.mouse_handler import press_left_click, press_right_click, release_left_click, release_right_click

keyboard_keys = ["SPACE", "W", "A", "S", "D"]
mouse_press_functions = [press_left_click, press_right_click]
mouse_release_functions = [release_left_click, release_right_click]


def press_keyboard_keys(keys: List[str], actions: List[bool]):
    for i, (key, action) in enumerate(zip(keys, actions)):
        if action:
            keyboard.press(key)
        else:
            keyboard.release(key)

        if i == 0:
            keyboard.release(key)


def press_mouse_keys(actions: List[bool]):
    for action, press_function, release_function in zip(actions, mouse_press_functions, mouse_release_functions):
        if action:
            press_function()
        else:
            release_function()


def press_keys(keys: List[str], actions: List[bool]):
    assert len(keys) == len(actions) - 2
    press_keyboard_keys(keys, actions[:len(keys)])
    press_mouse_keys(actions[len(keys):])


if __name__ == "__main__":
    # sleep 2 seconds to allow the user to switch to the game window
    import time

    time.sleep(2)

    press_keys(keyboard_keys, [True, True, False, False, False, True, True])

    while True:
        pass