from pipeline.data_loader import DataLoader


def list_to_int(a: list):
    return sum(1 << i for i, b in enumerate(a) if b)


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
    gatherer = DataLoader(
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


if __name__ == "__main__":
    print_stats(get_stats("data/big_fans/train"))
