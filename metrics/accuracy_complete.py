from torchmetrics import Accuracy


class AccuracyComplete(Accuracy):
    def __init__(self, **kwargs):
        super().__init__(subset_accuracy=True, **kwargs)
