import win32api as wapi


key_list = ["\b"]
for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ 123456789,.'APS$/\\":
    key_list.append(char)

important_keys = "123 "
single_keys = "WASD"
combo_keys = {
    "Q": "WA",
    "E": "WD",
    "Z": "SA",
    "C": "SD",
}

all_keys = "123 WASD"


def keyboard_key_check():
    pressed_keys = [key for key in all_keys if wapi.GetAsyncKeyState(ord(key))]

    for key in important_keys:
        if key in pressed_keys:
            return key

    for pseudo_key, combo in combo_keys.items():
        if all(key in pressed_keys for key in combo):
            return pseudo_key

    for key in single_keys:
        if key in pressed_keys:
            return key

    return "+"
