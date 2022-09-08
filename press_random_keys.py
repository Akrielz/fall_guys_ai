import random
import time

from key_utils.get_keys import key_check
import pydirectinput
import keyboard
from key_utils.direct_keys import press_key, release_key, W, D, A

# Sleep time after actions
delay_time = 0.1

# Wait for me to push B to start.
keyboard.wait('B')
time.sleep(2)

# Hold down W no matter what!
keyboard.press('w')

# Randomly pick action then sleep.
# 0 do nothing release everything ( except W )
# 1 hold left
# 2 hold right
# 3 Press Jump

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
    keys = key_check()
    if keys == "H":
        break

keyboard.release('W')