from typing import List

from pipeline.video_data_loader import VideoDataLoader


def list_to_int(a: List[bool]):
    return sum(1 << i for i, b in enumerate(a) if b)


def int_to_list(a: int):
    return [int(x) for x in list((bin(a)[2:])[::-1])]


def print_stats(dictionary):
    mapping = dict()
    for i in range(2 ** 7):
        word = []
        l = [int(x) for x in list((bin(i)[2:])[::-1])]
        for key, isPrees in zip(["S", "w", "a", "s", "d", "L", "R"], l):
            if isPrees:
                word.append(key)
        mapping[i] = "".join(word)

    for key, val in sorted(list(dictionary.items()), key=lambda x: x[1], reverse=True):
        if val:
            print(f"'{mapping[key]}' \t{key}\t {val}")


def get_stats(path: str):
    gatherer = VideoDataLoader(
        batch_size=1, time_size=8,
        data_dir=path, seed=0,
        progress_bar=True, balanced_data=False
    )

    my_dict = dict()
    for i in range(2 ** 7):
        my_dict[i] = 0

    for videos, keys, mouse, masks in gatherer.iter_epoch_data():
        for key in keys[0]:
            my_dict[list_to_int(key[3:])] += 1

    return my_dict


def get_key_mapping(round_dir: str):
    key_dict_train = get_stats(f"{round_dir}/train")
    key_dict_test = get_stats(f"{round_dir}/test")

    # combine train and test into a single dict
    key_dict = dict()
    for i in range(2 ** 7):
        key_dict[i] = key_dict_train[i] + key_dict_test[i]

    # filter out keys that are not used
    key_dict = {key: val for key, val in key_dict.items() if val}

    # create new class mapping for the keys
    key_mapping = dict()
    for i, key in enumerate(sorted(key_dict.keys())):
        key_mapping[key] = i

    return key_mapping


if __name__ == "__main__":
    key_dict = get_stats("data/big_fans/train")
    key_mapping = get_key_mapping("data/big_fans")
    print_stats(key_dict)
