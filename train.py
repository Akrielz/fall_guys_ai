import os

import torch
import torch.nn as nn
from torchmetrics import Accuracy
from vision_models_playground import models

from eda.keys_frequency import get_key_mapping
from model.image.resnet_50 import build_resnet_50, load_resnet_50
from model.utility.load_agent import load_agent
from pipeline.optimizer import get_optimizer
from pipeline.trainer_image import TrainerImage

if __name__ == "__main__":

    # Init vars
    in_channels = 4
    balanced_data = True
    data_dir = 'data/roll_on'
    num_classes = 2**7
    device = torch.device('cuda')
    agent_path = "trained_agents/dizzy_heights/resnet50_pretrained/2022-10-21_20-13-54/model_last.pt"

    # Create model
    # model = models.classifiers.build_cvt_13(num_classes=num_classes, in_channels=in_channels)

    # Create ResNet50 pretrained model from torchvision
    # model = build_resnet_50(weights="IMAGENET1K_V2", in_channels=in_channels, num_classes=num_classes)
    # model = load_agent(agent_path, model, device)

    model = load_resnet_50(agent_path, device, in_channels=in_channels, num_classes=num_classes,
                           previous_num_classes=num_classes)

    # Create Optimizer
    optimizer = get_optimizer(params=model.parameters(), lr=5e-3)

    # Add lr scheduler
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    # Create Loss
    train_data_dir = os.path.join(data_dir, 'train')

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
        model_name='resnet50_pretrained',
        consider_last_n_losses=100,
        consider_min_n_losses=100,
        apply_augmentations=True,
        balanced_data=balanced_data,
        scheduler=scheduler,
        resize_image_size=(224, 224),
        prob_soft_aug_fail=0.02,
    )

    # Train
    trainer.train(num_epochs=20, run_test_too=True)
    # trainer.test()
