import os

import torch
import torch.nn as nn
from vision_models_playground import models

from metrics.accuracy_complete import AccuracyComplete
from metrics.accuracy_partial import AccuracyPartial
from pipeline.optimizer import get_optimizer
from pipeline.positive_weights import PositiveWeightCalculator
from pipeline.trainer_image import TrainerImage

if __name__ == "__main__":
    # Init vars
    num_classes = 7
    in_channels = 4
    balanced_data = True
    data_dir = 'data/door_dash'
    device = torch.device('cuda')

    # Create model
    model = models.classifiers.build_cvt_13(num_classes=num_classes, in_channels=in_channels)

    # Create Optimizer
    optimizer = get_optimizer(params=model.parameters(), lr=5e-3)

    # Add lr scheduler
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    # Create Loss
    train_data_dir = os.path.join(data_dir, 'train')
    pos_weight = PositiveWeightCalculator(balanced_data, train_data_dir, num_classes).calculate()
    pos_weight = pos_weight.to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Create metrics
    metrics = [
        AccuracyComplete(num_classes=num_classes),  # For all inputs at once
        AccuracyPartial(num_classes=num_classes),  # For partial inputs
    ]

    # Create Trainer
    trainer = TrainerImage(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        metrics=metrics,
        seed=0,
        data_dir='data/door_dash',
        batch_size=1,
        time_size=4,
        save_every_n_steps=100,
        model_name='CvT13',
        consider_last_n_losses=100,
        consider_min_n_losses=100,
        apply_augmentations=True,
        balanced_data=balanced_data,
        scheduler=scheduler,
    )

    # Train
    trainer.train(num_epochs=20)
