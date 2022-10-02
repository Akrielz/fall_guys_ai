import os
from typing import Optional

import numpy as np
import torch

from data_utils.data_handler import load_data_general
from pipeline.data_loader import DataLoader


class PositiveWeightCalculator:
    def __init__(
            self,
            balanced_data: bool = False,
            data_dir: str = 'data/door_dash/train',
            num_classes: int = 7
    ):
        self.balanced_data = balanced_data
        self.data_dir = data_dir
        self.num_classes = num_classes
        self.eps = 1e-6

    def __call__(self):
        self.key_files = self._read_file_names()

        pos_count = np.zeros(self.num_classes, dtype=int)
        total_num = 0
        for key_file in self.key_files:
            key_path = os.path.join(self.data_dir, key_file)
            keys = load_data_general(key_path)

            mask = None
            if self.balanced_data:
                mask = DataLoader.balance_data_mask(keys)

            buttons = self._load_buttons(keys, mask=mask)
            pos_count = pos_count + np.array([sum(buttons[:, i] == 1) for i in range(buttons.shape[1])])
            total_num += buttons.shape[0]

        neg_count = total_num - pos_count
        pos_weights = neg_count / (pos_count + self.eps)
        return torch.from_numpy(pos_weights)

    def _load_buttons(
            self,
            keys,
            mask: Optional[np.ndarray] = None
    ):
        buttons_batch = []
        for i, (buttons, _) in enumerate(keys):
            if mask is not None and not mask[i]:
                continue

            buttons_batch.append(buttons[3:])

        return np.array(buttons_batch)

    def _read_file_names(self):
        return [file_name for file_name in os.listdir(self.data_dir) if file_name.endswith(".keys")]

    def calculate(self):
        return self()


if __name__ == "__main__":
    positive_weight_calculator = PositiveWeightCalculator(balanced_data=False)
    print(positive_weight_calculator())
