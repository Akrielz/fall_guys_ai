import os

import numpy as np

from data_utils.data_handler import load_data_general
from image_utils.video_handler import load_video_batch_iterator


class Gatherer:
    def __init__(self, batch_size, time_size, data_dir):
        self.batch_size = batch_size
        self.time_size = time_size
        self.data_dir = data_dir

        self.file_names = self._read_file_names()

    def _read_file_names(self):
        return [file_name[:-4] for file_name in os.listdir(self.data_dir) if file_name.endswith(".avi")]

    def iter_epoch_file_names(self):
        permutation = np.random.permutation(len(self.file_names))

        for i in range(0, len(self.file_names), self.batch_size):
            batch = permutation[i:i + self.batch_size]
            batch = [self.file_names[index] for index in batch]

            yield batch

    def iter_epoch_data(self):
        for file_names in self.iter_epoch_file_names():
            generators = [self._iter_data(file_name) for file_name in file_names]
            break

    def _load_keys_batch_iterator(self, keys):
        for i in range(0, len(keys), self.time_size):
            yield keys[i:i + self.time_size]

    def _iter_data(self, file_name):
        file_path = os.path.join(self.data_dir, file_name)

        video = load_video_batch_iterator(f"{file_path}.avi", batch_size=self.time_size)

        keys = load_data_general(f"{file_path}.keys")
        keys = self._load_keys_batch_iterator(keys)

        for video_batch, key_batch in zip(video, keys):
            yield video_batch, key_batch


if __name__ == "__main__":
    gatherer = Gatherer(3, 1, "data/train")
    for videos, keys in gatherer.iter_epoch_data():
        pass
