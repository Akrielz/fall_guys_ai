import os
from copy import deepcopy
from datetime import datetime
from typing import Optional, Literal, List, Tuple

import numpy as np
import torch.optim
from colorama import Fore
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Resize
from tqdm import tqdm
from vision_models_playground.models.augmenters import Augmeneter

from image_utils.image_handler import threshold_frame
from pipeline.video_data_loader import VideoDataLoader


class TrainerImage:
    def __init__(
            self,
            model: nn.Module,
            optimizer: torch.optim.Optimizer,
            loss_fn: nn.Module,
            device: torch.device,
            metrics: Optional[List[nn.Module]] = None,
            seed: Optional[int] = None,
            data_dir: str = 'data',
            batch_size: int = 1,
            time_size: int = 16,
            save_every_n_steps: int = 10,
            consider_last_n_losses: int = 10,
            consider_min_n_losses: int = 5,
            model_name: Optional[str] = None,
            apply_augmentations: bool = False,
            balanced_data: bool = False,
            scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
            original_image_size: Tuple[int, int] = (216, 384),
            resize_image_size: Optional[Tuple[int, int]] = None,
    ):
        data_dir = os.path.normpath(data_dir)

        # Save data
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.seed = seed
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.time_size = time_size
        self.save_every_n_steps = save_every_n_steps
        self.consider_last_n_losses = consider_last_n_losses
        self.consider_min_n_losses = consider_min_n_losses
        self.scheduler = scheduler
        self.original_image_size = original_image_size
        self.resize_image_size = resize_image_size

        # Move model to device
        self.model.to(self.device)

        # Create data gatherers
        train_data_dir = os.path.join(self.data_dir, 'train')
        train_gatherer = VideoDataLoader(
            batch_size=self.batch_size,
            time_size=self.time_size,
            data_dir=train_data_dir,
            seed=self.seed,
            progress_bar=False,
            balanced_data=balanced_data,
        )

        test_data_dir = os.path.join(self.data_dir, 'test')
        test_gatherer = VideoDataLoader(
            batch_size=self.batch_size,
            time_size=self.time_size,
            data_dir=test_data_dir,
            seed=self.seed,
            progress_bar=False
        )

        self.gatherers = {
            'train': train_gatherer,
            'test': test_gatherer,
        }

        if metrics is None:
            metrics = []

        for metric in metrics:
            metric.to(self.device)

        # Add metrics
        self.metrics = {
            'train': metrics,
            'test': deepcopy(metrics),
        }

        # Init best loss
        self.best_loss_mean = float('inf')
        self.last_n_losses = []

        # Create save dir for agents
        if model_name is None:
            model_name = self.model.__class__.__name__

        data_dir_format = data_dir
        if data_dir_format.startswith('data') and len(data_dir_format) > 4:
            data_dir_format = data_dir_format[5:]

        current_date = f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
        self.save_dir = os.path.join("trained_agents", data_dir_format, model_name, current_date)
        os.makedirs(self.save_dir, exist_ok=True)

        # Create agent name by date
        last_agent_name = "model_last"
        best_agent_name = "model_best"

        # Create tensorboard writer
        self.writer = SummaryWriter(log_dir=self.save_dir)

        # Create paths vars
        self.last_agent_path = os.path.join(self.save_dir, last_agent_name + '.pt')
        self.best_agent_path = os.path.join(self.save_dir, best_agent_name + '.pt')

        # Create image augmenter
        used_image_size = original_image_size if resize_image_size is None else resize_image_size

        self.rescale_layer = nn.Sequential(
            Rearrange('b h w c -> b c h w'),
            Resize(used_image_size),
            Rearrange('b c h w -> b h w c')
        ) if resize_image_size is not None else None

        self.image_augmenter = nn.Sequential(
            Rearrange('b h w c -> b c h w'),
            Augmeneter(
                image_size=used_image_size, background_color=1.0, rotation_angles=15
            ),
            Rearrange('b c h w -> b h w c')
        ) if apply_augmentations else None
        # Note: The image augmenter is working on CPU so it is compatible with the collate_fn

    def _collate_fn_cpu(
            self,
            frames: torch.Tensor,
            keys: torch.Tensor,
            masks: torch.Tensor,
            phase: Literal['train', 'test'],
    ):
        """
        Collate function for DataLoader
        :param frames: (batch_size, time_size, height, width, channels)
        :param keys: (batch_size, time_size, num_keys)
        :param masks: (batch_size, time_size)

        Note: All the operations are kept on CPU
        """

        # Combine batch and time dimensions
        frames, keys, masks = list(map(lambda x: rearrange(x, "b t ... -> (b t) ..."), [frames, keys, masks]))

        # Apply mask
        frames = frames[masks]
        keys = keys[masks]

        # Apply rescale layer
        if self.rescale_layer is not None:
            frames = self.rescale_layer(frames)

        # Apply augmentations
        if self.image_augmenter is not None and phase == 'train':
            frames_augmented = self.image_augmenter(frames)
            frames = torch.cat([frames, frames_augmented], dim=0)
            keys = torch.cat([keys, keys], dim=0)
            masks = torch.cat([masks, masks], dim=0)

        # Get black and white frames
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

    def _collate_fn_device(
            self,
            frames: torch.Tensor,
            keys: torch.Tensor,
            masks: torch.Tensor
    ):
        # Move data to device
        frames = frames.to(self.device)
        keys = keys.to(self.device)
        masks = masks.to(self.device)

        return frames, keys, masks

    def _save_best_model(self, loss: torch.Tensor):
        self.last_n_losses.append(loss.item())
        if len(self.last_n_losses) > self.consider_last_n_losses:
            self.last_n_losses.pop(0)

        if len(self.last_n_losses) >= self.consider_min_n_losses:
            loss_mean = sum(self.last_n_losses) / len(self.last_n_losses)

            if loss_mean < self.best_loss_mean:
                self.best_loss_mean = loss_mean
                torch.save(self.model.state_dict(), self.best_agent_path)

    def _do_epoch(
            self,
            phase: Literal['train', 'test'] = 'train',
            epoch: int = -1,
    ):
        # Choose the correct gatherer and metrics
        gatherer = self.gatherers[phase]
        metrics = self.metrics[phase]

        # Set model to train or eval mode
        if phase == 'train':
            self.model.train()
            color = Fore.CYAN
        else:
            self.model.eval()
            color = Fore.YELLOW

        gatherer_iter = gatherer.iter_epoch_data(enumerate_files=True)
        progress_bar = tqdm(gatherer_iter)
        for step_index, (file_index, frames, keys, _, masks) in enumerate(progress_bar):
            # Apply collate function
            frames, keys, masks = self._collate_fn_cpu(frames, keys, masks, phase)
            frames, keys, masks = self._collate_fn_device(frames, keys, masks)

            # Forward pass
            if phase == 'train':
                self.optimizer.zero_grad()

            predicted = self.model(frames)
            loss = self.loss_fn(predicted, keys)

            if phase == 'train':
                loss.backward()
                self.optimizer.step()
                self._save_best_model(loss)

            for metric in metrics:
                metric.update(predicted, keys.long())

            description = self._compute_progress_description(
                color, epoch, step_index, loss, metrics, phase, gatherer, file_index
            )
            progress_bar.set_description(description, refresh=False)

            if phase == 'train' and step_index % self.save_every_n_steps == 0:
                torch.save(self.model.state_dict(), self.last_agent_path)

    def _compute_progress_description(
            self,
            color,
            epoch,
            step_index,
            loss,
            metrics,
            phase,
            gatherer,
            file_index
    ):
        # Update progress bar
        metric_log = ''

        metric_values = []
        for metric in metrics:
            metric_values.append(metric.compute())
            metric_log += f'{metric.__repr__()[:-2]}: {metric_values[-1]:.4f} | '

        loss_name = "Loss" if len(self.loss_fn.__repr__()) > 30 else self.loss_fn.__repr__()[:-2]
        loss_log = f'{loss_name}: {loss.item():.4f}'
        if self.best_loss_mean != float('inf'):
            loss_log += f' | Best loss: {self.best_loss_mean:.4f}'

        description = color + f"{phase} Epoch: {epoch}, File: {file_index} / {len(gatherer)}, Step: {step_index} " \
                              f"| {loss_log} | {metric_log}"

        # Add the progress to the tensorboard writter
        self.writer.add_scalar(f'{phase}/loss', loss.item(), step_index)
        for metric_value, metric in zip(metric_values, metrics):
            self.writer.add_scalar(f'{phase}/{metric.__repr__()[:-2]}', metric_value, step_index)

        if self.best_loss_mean != float('inf'):
            self.writer.add_scalar(f'{phase}/best_loss', self.best_loss_mean, step_index)

        return description

    def train(self, num_epochs: int = 10, run_test_too: bool = False):
        for epoch in range(num_epochs):
            self._do_epoch('train', epoch)

            if self.scheduler is not None:
                self.scheduler.step()

            if run_test_too:
                with torch.no_grad():
                    self._do_epoch('test', epoch)

    @torch.no_grad()
    def test(self):
        self._do_epoch('test')
