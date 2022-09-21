from vision_models_playground import models
from pipeline.trainer_image import TrainerImage

if __name__ == "__main__":
    model = models.classifiers.build_resnet_18(in_channels=3, num_classes=10)

    trainer = TrainerImage(
        model=model,
    )