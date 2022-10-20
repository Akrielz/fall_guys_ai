import os
import pickle

import numpy as np

from data_utils.data_handler import load_data_general
from eda.keys_frequency import get_stats, list_to_int


def create_augmentation_files_for_map_dir(map_dir: str):
    real_keys_freq = get_stats(map_dir)

    # eliminate keys with 0 count
    real_keys_freq = {key: val for key, val in real_keys_freq.items() if val}

    # get max value
    max_val = max(real_keys_freq.values())

    # compute number of augmentations
    augmentation_freq = {key: max_val - val for key, val in real_keys_freq.items()}

    # filter out the ones with 0 augmentations
    augmentation_freq = {key: val for key, val in augmentation_freq.items() if val}

    # how many augmentations per state
    needed_augmentations_per_key = {key: val // real_keys_freq[key] for key, val in augmentation_freq.items() if val}

    # get all the files inside the dir map_dir
    files = [file_name for file_name in os.listdir(map_dir) if file_name.endswith(".keys")]

    for file_keys in files:
        file_path = os.path.join(map_dir, file_keys)
        keys = load_data_general(file_path)

        frames_to_augment = []
        for i, (key, _) in enumerate(keys):
            state = list_to_int(key[3:])

            if state not in needed_augmentations_per_key:
                continue

            frame_to_augment = (i, needed_augmentations_per_key[state])
            frames_to_augment.append(frame_to_augment)

        frames_to_augment = np.array(frames_to_augment)

        file_augmented = file_keys.replace(".keys", ".aug")
        file_augmented_path = os.path.join(map_dir, file_augmented)

        with open(file_augmented_path, "wb") as f:
            pickle.dump(frames_to_augment, f, protocol=pickle.HIGHEST_PROTOCOL)


def create_augmentation_files(data_dir: str = "data"):
    for map_dir in os.listdir(data_dir):
        map_dir_train = os.path.join(data_dir, map_dir, "train")
        if len(os.listdir(map_dir_train)) == 0:
            continue

        create_augmentation_files_for_map_dir(map_dir_train)

        map_dir_test = os.path.join(data_dir, map_dir, "test")
        if len(os.listdir(map_dir_test)) == 0:
            continue

        create_augmentation_files_for_map_dir(map_dir_test)


if __name__ == "__main__":
    create_augmentation_files(data_dir="data")