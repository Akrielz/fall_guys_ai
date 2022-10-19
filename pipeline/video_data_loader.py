import os
from typing import Optional, List

import numpy as np
import torch
from tqdm import tqdm

from data_utils.data_handler import load_data_general
from image_utils.video_handler import load_video_batch_iterator, load_video_len
from pipeline.data_loader import DataLoader


class VideoDataLoader(DataLoader):
    def __init__(
            self,
            batch_size: int,
            time_size: int,
            data_dir: str,
            seed: Optional[int] = None,
            progress_bar: bool = False,
            balanced_data: bool = False
    ):
        # init upper
        super().__init__(batch_size, data_dir, seed, progress_bar)

        self.time_size = time_size
        self.balanced_data = balanced_data

    @staticmethod
    def _pad_data(data: np.ndarray, max_len: int):
        data_dtype = data.dtype
        if len(data) < max_len:
            pad_values = np.zeros([max_len - len(data), *data.shape[1:]], dtype=data_dtype)
            data = np.concatenate([data, pad_values], axis=0)

        return data

    def iter_epoch_file_names(self):
        permutation = np.random.permutation(len(self.file_names))

        file_len_iter = range(0, len(self.file_names), self.batch_size)
        if self.progress_bar:
            file_len_iter = tqdm(file_len_iter)

        for i in file_len_iter:
            batch = permutation[i:i + self.batch_size]
            batch = [self.file_names[index] for index in batch]

            yield batch

    def iter_epoch_data(
            self,
            enumerate_files: bool = False,
            balanced_data: Optional[bool] = None
    ):
        if balanced_data is None:
            balanced_data = self.balanced_data

        file_names_iter = self.iter_epoch_file_names()
        if enumerate_files:
            file_names_iter = enumerate(file_names_iter)

        for file_names_tuple in file_names_iter:

            if enumerate_files:
                file_batch_index, file_names = file_names_tuple
            else:
                file_names = file_names_tuple

            lengths = self._get_keys_lens(file_names, balanced_data=balanced_data)
            max_len = max(lengths)

            video_shape = None
            key_shape = None
            mouse_shape = None

            generators = [self._iter_data(file_name, balanced_data=balanced_data) for file_name in file_names]

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

    def _load_keys_time_batch_iterator(
            self,
            keys,
            time_size: int = None,
            mask: Optional[np.ndarray] = None
    ):
        if time_size is None:
            time_size = self.time_size

        buttons_batch, mouse_batch = [], []
        for i, (buttons, mouse) in enumerate(keys):
            if mask is not None and not mask[i]:
                continue

            buttons_batch.append(buttons)
            mouse_batch.append(mouse)

            if len(buttons_batch) == time_size:
                yield np.array(buttons_batch), np.array(mouse_batch)
                buttons_batch, mouse_batch = [], []

        if len(buttons_batch) > 0:
            yield np.array(buttons_batch), np.array(mouse_batch)

    def _get_video_lens(self, file_names: List[str]):
        return [load_video_len(os.path.join(self.data_dir, f"{file_name}.avi")) for file_name in file_names]

    def _get_keys_lens(self, file_names: List[str], balanced_data: bool = False):
        keys_batch = [load_data_general(os.path.join(self.data_dir, f"{file_name}.keys")) for file_name in file_names]
        if not balanced_data:
            return [len(keys) for keys in keys_batch]

        mask_batch = [self.balance_data_mask(keys) for keys in keys_batch]
        return [np.sum(mask) for mask in mask_batch]

    @staticmethod
    def balance_data_mask(keys):
        mask = np.zeros(len(keys), dtype=bool)
        mask[0] = True
        for i in range(1, len(keys)):
            if keys[i][0] != keys[i - 1][0]:
                mask[i] = True

        return mask

    def _iter_data(self, file_name: str, time_size: int = None, balanced_data: bool = False):
        # If time_size is None, use self.time_size
        if time_size is None:
            time_size = self.time_size

        # Compute file path
        file_path = os.path.join(self.data_dir, file_name)

        # Load keys
        keys = load_data_general(f"{file_path}.keys")

        # Apply balancing if needed
        mask = None
        if balanced_data:
            mask = self.balance_data_mask(keys)

        keys = self._load_keys_time_batch_iterator(keys, time_size=time_size, mask=mask)

        # Load video
        video = load_video_batch_iterator(f"{file_path}.avi", batch_size=time_size, mask=mask)

        # Create a generator that yields the data
        for video_time_batch, (key_time_batch, mouse_time_batch) in zip(video, keys):
            yield video_time_batch, key_time_batch, mouse_time_batch


if __name__ == "__main__":
    gatherer = VideoDataLoader(
        batch_size=1, time_size=8,
        data_dir="data/big_fans/train", seed=0,
        progress_bar=True, balanced_data=True
    )
    for videos, keys, mouse, masks in gatherer.iter_epoch_data():
        pass
