import random
import time

from key_utils.keyboard_handler import keyboard_key_check
import keyboard

delay_time = 0.1
keyboard.wait('B')
time.sleep(2)
keyboard.press('w')

while True:
    action = random.randint(0,0)

    if action == 0:
        print("Doing nothing....")
        keyboard.release("a")
        keyboard.release("d")
        keyboard.release("space")
        time.sleep(delay_time)

    elif action == 1:
        print("Going left....")
        keyboard.press("a")
        keyboard.release("d")
        keyboard.release("space")
        time.sleep(delay_time)

    elif action == 2:
        print("Going right....")
        keyboard.press("d")
        keyboard.release("a")
        keyboard.release("space")
        time.sleep(delay_time)

    elif action == 3:
        print("JUMP!")
        keyboard.press("space")
        keyboard.release("a")
        keyboard.release("d")
        time.sleep(delay_time)

    # End simulation by hitting h
    keys = keyboard_key_check()
    if keys == "H":
        break

keyboard.release('W')