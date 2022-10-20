import torch
import torch.nn as nn
from torchmetrics import Accuracy

from model.image.fake import FakeModel
from pipeline.optimizer import get_optimizer
from pipeline.trainer_image import TrainerImage

if __name__ == "__main__":
    # Init vars
    num_classes = 2**7
    in_channels = 4
    balanced_data = True
    data_dir = 'data/block_party'
    device = torch.device('cuda')

    # Create FakeModel
    output = torch.zeros(num_classes).to(device)
    output[2] = 1
    model = FakeModel(output=output)

    # Create Optimizer
    optimizer = get_optimizer(params=model.parameters(), lr=5e-3)

    # Add lr scheduler
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    loss_fn = nn.CrossEntropyLoss()

    # Create metrics
    metrics = [
        Accuracy(num_classes=num_classes),  # For all inputs at once
    ]

    # Create Trainer
    trainer = TrainerImage(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        metrics=metrics,
        seed=0,
        data_dir=data_dir,
        batch_size=4,
        save_every_n_steps=100,
        model_name='fake',
        consider_last_n_losses=100,
        consider_min_n_losses=100,
        apply_augmentations=True,
        balanced_data=balanced_data,
        scheduler=scheduler,
        resize_image_size=(224, 224),
    )

    # Test
    trainer.test()

