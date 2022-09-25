import torch
import torch.nn as nn
from vision_models_playground import models

from pipeline.trainer_image import TrainerImage

if __name__ == "__main__":
    # Create model
    model = nn.Sequential(
        models.classifiers.build_resnet_18(in_channels=4, num_classes=7),
        nn.Sigmoid()
    )

    # Create Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Create Loss
    loss_fn = torch.nn.BCEWithLogitsLoss()

    # Create Trainer
    trainer = TrainerImage(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=torch.device('cuda'),
        seed=0,
        data_dir='data',
        batch_size=1,
        time_size=4,
    )

    # Train
    trainer.train(num_epochs=10)

