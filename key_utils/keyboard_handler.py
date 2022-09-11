import win32api as wapi


key_list = ["\b"]
for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ 123456789,.'APS$/\\":
    key_list.append(char)

important_keys = "123 "
single_keys = "WASD"

all_keys = "123 WASD"
all_key_representations = "123^WASD"


def keyboard_key_check():
    return [True if wapi.GetAsyncKeyState(ord(key)) else False for key in all_keys]
