import os
import random

from data_utils.data_handler import load_data_general


def split_data(directory: str = "data"):
    unsorted_dir = os.path.join(directory)

    files = os.listdir(unsorted_dir)
    files = [file for file in files if file.endswith(".avi")]

    random.shuffle(files)

    if len(files) == 1:
        return [files[0]], []

    return files[:-1], files[-1:]


def move_data(files, directory: str = "data", split: str = "train"):
    unsorted_dir = os.path.join(directory)
    sorted_dir = os.path.join(directory, split)

    if not os.path.exists(sorted_dir):
        os.makedirs(sorted_dir)

    for file in files:
        try:
            os.rename(os.path.join(unsorted_dir, file), os.path.join(sorted_dir, file))
            os.rename(os.path.join(unsorted_dir, f"{file[:-4]}.keys"), os.path.join(sorted_dir, f"{file[:-4]}.keys"))
        except FileNotFoundError:
            print(f"File {file} not found")


def get_round_names(directory: str = "data"):
    """Get all the directories from the data directory"""
    dirs = os.listdir(directory)
    dirs = [dir for dir in dirs if os.path.isdir(os.path.join(directory, dir))]
    return dirs


def split_rounds():
    round_names = get_round_names("data")

    for round_name in round_names:
        files, test_files = split_data(os.path.join("data", round_name))
        move_data(files, os.path.join("data", round_name), "train")
        move_data(test_files, os.path.join("data", round_name), "test")


def find_broken():
    round_names = get_round_names("data")

    for round_name in round_names:
        files = os.listdir(os.path.join("data", round_name))
        if len(files) == 0:
            print(f"Round {round_name} is empty")
            continue

        if len(files) >= 3:
            print(f"Round {round_name} has too many files")
            continue

        for split in ["train", "test"]:
            files = os.listdir(os.path.join("data", round_name, split))
            files = [file for file in files if file.endswith(".avi")]

            for file in files:
                try:
                    load_data_general(os.path.join("data", round_name, split, f"{file[:-4]}.keys"))
                except FileNotFoundError:
                    print(f"File {file} not found in {round_name} {split}")


if __name__ == "__main__":
    find_broken()
