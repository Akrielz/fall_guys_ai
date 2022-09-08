import os.path
import pickle
import tempfile

from image_utils.grab_screen import get_screenshot
from key_utils.get_keys import key_check


def record_data(x1: int = 0, y1: int = 0, x2: int = 1920, y2: int = 1080, directory="data"):
    imgs = []
    targets = []

    recording = False

    while True:
        img = get_screenshot(x1, y1, x2, y2) if recording else None
        key = key_check()

        if key == "1" and not recording:
            print("Recording")
            recording = True

        elif key == "2" and recording:
            print("Save record")
            recording = False

            with tempfile.NamedTemporaryFile() as tf:
                file_name = tf.name
                file_name = file_name[file_name.rfind("\\") + 1:]

            file_path = os.path.join(directory, file_name)

            with open(file_path, "wb") as f:
                pickle.dump((imgs, targets), f, protocol=pickle.HIGHEST_PROTOCOL)

            imgs, targets = [], []

        elif key == "3" and recording:
            print("Cancel record")
            recording = False
            imgs, targets = [], []

        if recording:
            imgs.append(img)
            targets.append(key)


if __name__ == "__main__":
    record_data()
