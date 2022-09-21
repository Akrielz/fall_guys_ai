import torch.optim
from torch import nn


class TrainerImage:
    def __init__(
            self,
            model: nn.Module,
            optimizer: torch.optim.Optimizer,
            device: torch.device,
            seed: int = 0,
            data_dir: str = 'data',
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.seed = seed
        self.data_dir = data_dir

        self.model.to(self.device)

    def train(self):
        pass

    def test(self):
        pass