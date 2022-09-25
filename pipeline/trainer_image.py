import os
from typing import Optional, Literal

import cv2
import numpy as np
import torch.optim
from einops import rearrange
from torch import nn

from pipeline.data_loader import DataLoader


class TrainerImage:
    def __init__(
            self,
            model: nn.Module,
            optimizer: torch.optim.Optimizer,
            loss_fn: nn.Module,
            device: torch.device,
            seed: Optional[int] = None,
            data_dir: str = 'data',
            batch_size: int = 1,
            time_size: int = 16,
    ):
        # Save data
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.seed = seed
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.time_size = time_size

        # Move model to device
        self.model.to(self.device)

        # Create data gatherers
        train_data_dir = os.path.join(self.data_dir, 'train')
        train_gatherer = DataLoader(
            batch_size=self.batch_size,
            time_size=self.time_size,
            data_dir=train_data_dir,
            seed=self.seed,
            progress_bar=True
        )

        test_data_dir = os.path.join(self.data_dir, 'test')
        test_gatherer = DataLoader(
            batch_size=self.batch_size,
            time_size=self.time_size,
            data_dir=test_data_dir,
            seed=self.seed,
            progress_bar=True
        )

        val_data_dir = os.path.join(self.data_dir, 'val')
        val_gatherer = DataLoader(
            batch_size=self.batch_size,
            time_size=self.time_size,
            data_dir=val_data_dir,
            seed=self.seed,
            progress_bar=True
        )

        self.gatherers = {
            'train': train_gatherer,
            'test': test_gatherer,
            'val': val_gatherer,
        }

    def _collate_fn(
            self,
            frames: torch.Tensor,
            keys: torch.Tensor,
            masks: torch.Tensor
    ):
        """
        Collate function for DataLoader
        :param frames: (batch_size, time_size, height, width, channels)
        :param keys: (batch_size, time_size, num_keys)
        :param masks: (batch_size, time_size)

        Note: All the operations are kept on CPU
        """

        if masks.dtype != torch.bool:
            print('Converting masks to bool')

        # Combine batch and time dimensions
        frames, keys, masks = list(map(lambda x: rearrange(x, "b t ... -> (b t) ..."), [frames, keys, masks]))

        # Apply mask
        frames = frames[masks]
        keys = keys[masks]

        # Get black and white frames
        def threshold_frame(frame: np.ndarray):
            black_white_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            segmented_frame = cv2.Canny(black_white_frame, threshold1=119, threshold2=250)

            return segmented_frame

        frames_segmented = torch.from_numpy(np.array(list(map(threshold_frame, frames.numpy()))))
        frames_segmented = frames_segmented.unsqueeze(-1)
        frames = torch.cat([frames, frames_segmented], dim=-1)

        # Put channels in standard dl format
        frames = rearrange(frames, "b h w c -> b c h w")

        # Convert images to [0, 1]
        frames = frames.float()
        frames = frames / 255

        # Keys are [1 2 3 Space W A S D LeftClick RightClick]
        # Eliminate [1 2 3]
        keys = keys[:, 3:]

        # Cast to float
        keys = keys.float()

        return frames, keys, masks

    def _do_epoch(
            self,
            phase: Literal['train', 'test', 'val'] = 'train',
    ):
        for frames, keys, _, masks in self.gatherers[phase].iter_epoch_data():
            # Apply collate function
            frames, keys, masks = self._collate_fn(frames, keys, masks)

            # Move data to device
            frames = frames.to(self.device)
            keys = keys.to(self.device)
            masks = masks.to(self.device)

            # Forward pass
            if phase == 'train':
                self.optimizer.zero_grad()

            outputs = self.model(frames)
            loss = self.loss_fn(outputs, keys)

            if phase == 'train':
                loss.backward()
                self.optimizer.step()

    def train(self, num_epochs: int = 10):
        for epoch in range(num_epochs):
            self._do_epoch('train')
            self._do_epoch('val')
        pass

    def test(self):
        self._do_epoch('test')
