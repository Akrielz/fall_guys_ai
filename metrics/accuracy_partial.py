from torchmetrics import Accuracy


class AccuracyPartial(Accuracy):
    def __init__(self, **kwargs):
        super().__init__(subset_accuracy=False, **kwargs)
