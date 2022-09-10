import os.path
import pickle
import tempfile
from typing import Optional

from image_utils.video_handling import save_video, load_video


def get_temporary_name():
    with tempfile.NamedTemporaryFile() as tf:
        file_name = tf.name
        file_name = file_name[file_name.rfind("\\") + 1:]

    return file_name


def save_data_general(data, directory: str = "data", file_name: Optional[str] = None):
    if not os.path.exists(directory):
        os.makedirs(directory)

    if file_name is None:
        file_name = get_temporary_name()

    file_path = os.path.join(directory, file_name)

    with open(file_path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def save_data_training(video, keys, directory: str = "data"):
    if not os.path.exists(directory):
        os.makedirs(directory)

    file_name = get_temporary_name()

    save_data_general(keys, directory, f"{file_name}.keys")
    save_video(video, file_name=os.path.join(directory, f"{file_name}.avi"))


def load_data_general(file_name: str):
    with open(file_name, "rb") as f:
        return pickle.load(f)


def load_data_training(file_name: str):
    keys = load_data_general(f"{file_name}.keys")
    video = load_video(file_name=f"{file_name}.avi")

    return video, keys
