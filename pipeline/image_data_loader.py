import os
from typing import Optional

import numpy as np
import torch
from einops.layers.torch import Rearrange
from torch import nn
from tqdm import tqdm

from data_utils.data_handler import load_data_general
from image_utils.video_handler import load_images_augmented_iterator, load_video_len_augmented, load_images_iterator, \
    load_video_len
from image_utils.weak_image_augmentation import WeakAugmeneter
from pipeline.data_loader import DataLoader


class ImageDataLoader(DataLoader):
    def __init__(
            self,
            batch_size: int,
            data_dir: str,
            seed: Optional[int] = None,
            progress_bar: bool = False,
            return_device: torch.device = torch.device("cpu"),
            balance_data: bool = False,
    ):
        super().__init__(batch_size, data_dir, seed, progress_bar)

        self.own_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.return_device = return_device

        self.augmenter = nn.Sequential(
            Rearrange("b h w c-> b c h w"),
            WeakAugmeneter(),
            Rearrange("b c h w -> b h w c"),
        )
        self.augmenter.eval()
        self.augmenter.to(self.own_device)

        self.balance_data = balance_data

        self._total_len = None
        self._len = None

    def __len__(self):
        if self._total_len is None:
            file_names = self.file_names
            file_names_avi = [os.path.join(self.data_dir, f"{file_name}.avi") for file_name in file_names]

            if self.balance_data:
                all_len = [load_video_len_augmented(file_name) for file_name in file_names_avi]
            else:
                all_len = [load_video_len(file_name) for file_name in file_names_avi]

            self._total_len = sum(all_len)

            self._len = self._total_len // self.batch_size
            if self._total_len % self.batch_size != 0:
                self._len += 1

        return self._len

    def __iter__(self):
        frame_batch = []
        keys_batch = []
        mouse_batch = []
        to_augment_batch = []

        # init progress bar
        progress_bar = None
        if self.progress_bar:
            progress_bar = tqdm(range(len(self)))

        for frame, keys, mouse, to_augment in self._iter_data():
            # add to batch
            frame_batch.append(frame)
            keys_batch.append(keys)
            mouse_batch.append(mouse)
            to_augment_batch.append(to_augment)

            if len(frame_batch) < self.batch_size:
                continue

            # yield
            yield self._prepare_to_yield(frame_batch, keys_batch, mouse_batch, to_augment_batch)

            # reset
            frame_batch = []
            keys_batch = []
            mouse_batch = []
            to_augment_batch = []

            if progress_bar is not None:
                progress_bar.update(1)

        if len(frame_batch) == 0:
            return

        yield self._prepare_to_yield(frame_batch, keys_batch, mouse_batch, to_augment_batch)

        if progress_bar is not None:
            progress_bar.update(1)
            progress_bar.close()

    def _prepare_to_yield(self, frame_batch, keys_batch, mouse_batch, to_augment_batch):

        # Convert into tensors
        frame_batch = torch.from_numpy(np.array(frame_batch))
        keys_batch = torch.from_numpy(np.array(keys_batch))
        mouse_batch = torch.from_numpy(np.array(mouse_batch))
        to_augment_batch = torch.from_numpy(np.array(to_augment_batch))

        if not self.balance_data:
            # Put data on correct device
            frame_batch = frame_batch.to(self.return_device)
            keys_batch = keys_batch.to(self.return_device)
            mouse_batch = mouse_batch.to(self.return_device)
            return frame_batch, keys_batch, mouse_batch

        # Put data on own device
        frame_batch = frame_batch.to(self.own_device)
        keys_batch = keys_batch.to(self.own_device)
        mouse_batch = mouse_batch.to(self.own_device)
        to_augment_batch = to_augment_batch.to(self.own_device)

        # Augment data
        if to_augment_batch.sum() > 0:
            frame_batch = frame_batch.float()
            frame_batch[to_augment_batch] = frame_batch[to_augment_batch] / 255.0
            frame_batch[to_augment_batch] = self.augmenter(frame_batch[to_augment_batch])
            frame_batch[to_augment_batch] = frame_batch[to_augment_batch] * 255.0
            frame_batch = frame_batch.to(torch.uint8)

        # Put data on correct device
        if self.own_device != self.return_device:
            frame_batch = frame_batch.to(self.return_device)
            keys_batch = keys_batch.to(self.return_device)
            mouse_batch = mouse_batch.to(self.return_device)

        return frame_batch, keys_batch, mouse_batch

    def _iter_data(self):
        file_names = self.file_names
        file_names_avi = [os.path.join(self.data_dir, f"{file_name}.avi") for file_name in file_names]
        file_names_keys = [os.path.join(self.data_dir, f"{file_name}.keys") for file_name in file_names]

        iter_func = load_images_augmented_iterator if self.balance_data else load_images_iterator

        augmented_iterators = [
            iter_func(file_name=file_name, random_permutation=True)
            for file_name in file_names_avi
        ]

        all_keys = [load_data_general(file_name) for file_name in file_names_keys]

        while len(augmented_iterators) > 0:
            iter_index = np.random.randint(0, len(augmented_iterators))
            augmented_iterator = augmented_iterators[iter_index]

            try:
                if self.balance_data:
                    frame, to_augment, i = next(augmented_iterator)
                else:
                    frame, i = next(augmented_iterator)
                    to_augment = False

            except StopIteration:
                # remove iterator
                augmented_iterators.pop(iter_index)
                all_keys.pop(iter_index)
                continue

            yield frame, all_keys[iter_index][i][0], all_keys[iter_index][i][1], to_augment


if __name__ == "__main__":
    gatherer = ImageDataLoader(
        batch_size=1,
        data_dir="data/the_whirlygig/train",
        seed=0,
        progress_bar=True,
        balance_data=False,
    )

    print(len(gatherer))
