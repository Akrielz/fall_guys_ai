import torch
import torch.nn as nn
from vision_models_playground import models

from metrics.accuracy_complete import AccuracyComplete
from metrics.accuracy_partial import AccuracyPartial
from pipeline.optimizer import get_optimizer
from pipeline.trainer_image import TrainerImage

if __name__ == "__main__":
    # Create model
    num_classes = 7
    in_channels = 4

    model = models.classifiers.build_cvt_13(num_classes=num_classes, in_channels=in_channels)

    # Create Optimizer
    optimizer = get_optimizer(params=model.parameters(), lr=5e-4)

    # Create Loss
    loss_fn = nn.BCEWithLogitsLoss()

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
        device=torch.device('cuda'),
        metrics=metrics,
        seed=0,
        data_dir='data',
        batch_size=1,
        time_size=4,
        save_every_n_steps=100,
        model_name='CvT13',
        consider_last_n_losses=100,
        consider_min_n_losses=100,
        apply_augmentations=True,
        balanced_data=True,
    )

    # Train
    trainer.train(num_epochs=10)
