import os
from typing import Optional

import numpy as np


class DataLoader:
    def __init__(
            self,
            batch_size: int,
            data_dir: str,
            seed: Optional[int] = None,
            progress_bar: bool = False,
    ):
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.seed = seed
        self.progress_bar = progress_bar

        if self.seed is not None:
            np.random.seed(self.seed)

        self.file_names = self._read_file_names()

    def _read_file_names(self):
        return [file_name[:-4] for file_name in os.listdir(self.data_dir) if file_name.endswith(".avi")]

    def __len__(self):
        len_batch = len(self.file_names) // self.batch_size
        if len(self.file_names) % self.batch_size != 0:
            len_batch += 1

        return len_batch