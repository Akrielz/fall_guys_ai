import os.path
import pickle
import tempfile
from typing import Optional

from image_utils.video_handling import save_video, load_video


def get_temporary_name(directory: Optional[str] = None, return_full_path: bool = False):
    with tempfile.NamedTemporaryFile(dir=directory) as tf:
        file_name = tf.name

    if not return_full_path:
        file_name = file_name[file_name.rfind("\\") + 1:]

    return file_name


def save_data_general(data, file_name: Optional[str] = None, directory: Optional[str] = None):
    assert file_name is not None or directory is not None, "You must specify a file name or a directory"

    if directory is not None and not os.path.exists(directory):
        os.makedirs(directory)

    if file_name is None:
        file_name = get_temporary_name(directory, return_full_path=True)

    with open(file_name, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def save_data_training(video, keys, directory: str = "data"):
    if not os.path.exists(directory):
        os.makedirs(directory)

    file_name = get_temporary_name(directory)

    save_data_general(keys, file_name=os.path.join(directory, f"{file_name}.keys"))
    save_video(video, file_name=os.path.join(directory, f"{file_name}.avi"))


def load_data_general(file_name: str):
    with open(file_name, "rb") as f:
        return pickle.load(f)


def load_data_training(file_name: str):
    keys = load_data_general(f"{file_name}.keys")
    video = load_video(file_name=f"{file_name}.avi")

    return video, keys
