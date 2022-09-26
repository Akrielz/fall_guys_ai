import os
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm

from data_utils.data_handler import load_data_general
from image_utils.video_handler import load_video_batch_iterator, load_video_len


class DataLoader:
    def __init__(
            self,
            batch_size: int,
            time_size: int,
            data_dir: str,
            seed: Optional[int] = None,
            progress_bar=False
    ):
        self.batch_size = batch_size
        self.time_size = time_size
        self.data_dir = data_dir
        self.seed = seed
        self.progress_bar = progress_bar

        self.file_names = self._read_file_names()

    def _read_file_names(self):
        return [file_name[:-4] for file_name in os.listdir(self.data_dir) if file_name.endswith(".avi")]

    def iter_epoch_file_names(self):
        if self.seed is not None:
            np.random.seed(self.seed)

        permutation = np.random.permutation(len(self.file_names))

        file_len_iter = range(0, len(self.file_names), self.batch_size)
        if self.progress_bar:
            file_len_iter = tqdm(file_len_iter)

        for i in file_len_iter:
            batch = permutation[i:i + self.batch_size]
            batch = [self.file_names[index] for index in batch]

            yield batch

    @staticmethod
    def _pad_data(data: np.ndarray, max_len: int):
        data_dtype = data.dtype
        if len(data) < max_len:
            pad_values = np.zeros([max_len - len(data), *data.shape[1:]], dtype=data_dtype)
            data = np.concatenate([data, pad_values], axis=0)

        return data

    def __len__(self):
        len_batch = len(self.file_names) // self.batch_size
        if len(self.file_names) % self.batch_size != 0:
            len_batch += 1

        return len_batch

    def iter_epoch_data(self, enumerate_files=False):
        file_names_iter = self.iter_epoch_file_names()
        if enumerate_files:
            file_names_iter = enumerate(file_names_iter)

        for file_batch_index, file_names in file_names_iter:
            lengths = self._get_video_lens(file_names)
            max_len = max(lengths)

            video_shape = None
            key_shape = None
            mouse_shape = None

            generators = [self._iter_data(file_name) for file_name in file_names]

            for time in range(0, max_len, self.time_size):
                video_batch = []
                key_batch = []
                mouse_batch = []
                mask_batch = []

                for length, generator in zip(lengths, generators):

                    if time < length:
                        video, keys, mouse = next(generator)
                        mask = np.ones(len(video), dtype=bool)

                        if video_shape is None:
                            video_shape = video.shape[1:]
                            key_shape = keys.shape[1:]
                            mouse_shape = mouse.shape[1:]
                    else:
                        video = np.zeros([self.time_size, *video_shape])
                        keys = np.zeros([self.time_size, *key_shape])
                        mouse = np.zeros([self.time_size, *mouse_shape])
                        mask = np.zeros(self.time_size, dtype=bool)

                    if len(video) < self.time_size:
                        video = self._pad_data(video, self.time_size)
                        keys = self._pad_data(keys, self.time_size)
                        mouse = self._pad_data(mouse, self.time_size)
                        mask = self._pad_data(mask, self.time_size)

                    video_batch.append(video)
                    key_batch.append(keys)
                    mouse_batch.append(mouse)
                    mask_batch.append(mask)

                video_batch = torch.from_numpy(np.array(video_batch))
                key_batch = torch.from_numpy(np.array(key_batch))
                mouse_batch = torch.from_numpy(np.array(mouse_batch))
                mask_batch = torch.from_numpy(np.array(mask_batch))

                output = (video_batch, key_batch, mouse_batch, mask_batch)
                if enumerate_files:
                    output = (file_batch_index, *output)

                yield output

    def _load_keys_time_batch_iterator(self, keys):
        for i in range(0, len(keys), self.time_size):
            sub_batch = keys[i:i + self.time_size]

            buttons_batch = []
            mouse_batch = []
            for buttons, mouse in sub_batch:
                buttons_batch.append(buttons)
                mouse_batch.append(mouse)

            yield np.array(buttons_batch), np.array(mouse_batch)

    def _get_video_lens(self, file_names):
        return [load_video_len(os.path.join(self.data_dir, f"{file_name}.avi")) for file_name in file_names]

    def _iter_data(self, file_name):
        file_path = os.path.join(self.data_dir, file_name)

        video = load_video_batch_iterator(f"{file_path}.avi", batch_size=self.time_size)

        keys = load_data_general(f"{file_path}.keys")
        keys = self._load_keys_time_batch_iterator(keys)

        for video_time_batch, (key_time_batch, mouse_time_batch) in zip(video, keys):
            yield video_time_batch, key_time_batch, mouse_time_batch


if __name__ == "__main__":
    gatherer = DataLoader(batch_size=1, time_size=1, data_dir="data/train", seed=0, progress_bar=True)
    for videos, keys, mouse, masks in (gatherer.iter_epoch_data()):
        pass
