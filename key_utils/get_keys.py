import win32api as wapi


key_list = ["\b"]
for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ 123456789,.'APS$/\\":
    key_list.append(char)


def key_check():
    keys = []
    for key in key_list:
        if wapi.GetAsyncKeyState(ord(key)):
            keys.append(key)

    for key in "123WASD ":
        if key in keys:
            return key

    return "+"
