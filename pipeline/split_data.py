import os
import random


def split_data(directory: str = "data", split_train: float = 0.8, split_val: float = 0.1, split_test: float = 0.1):
    assert split_train + split_val + split_test == 1.0, "The sum of the splits must be 1.0"

    unsorted_dir = os.path.join(directory, "unsorted")

    files = os.listdir(unsorted_dir)
    files = [file for file in files if file.endswith(".avi")]

    random.shuffle(files)

    train = files[:int(len(files) * split_train)]
    val = files[int(len(files) * split_train):int(len(files) * (split_train + split_val))]
    test = files[int(len(files) * (split_train + split_val)):]

    return train, val, test


def move_data(files, directory: str = "data", split: str = "train"):
    unsorted_dir = os.path.join(directory, "unsorted")
    sorted_dir = os.path.join(directory, split)

    if not os.path.exists(sorted_dir):
        os.makedirs(sorted_dir)

    for file in files:
        os.rename(os.path.join(unsorted_dir, file), os.path.join(sorted_dir, file))
        os.rename(os.path.join(unsorted_dir, f"{file[:-4]}.keys"), os.path.join(sorted_dir, f"{file[:-4]}.keys"))


def main():
    train, val, test = split_data()

    move_data(train, split="train")
    move_data(val, split="val")
    move_data(test, split="test")


if __name__ == "__main__":
    main()