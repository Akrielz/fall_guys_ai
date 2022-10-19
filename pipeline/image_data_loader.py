from typing import Optional

from pipeline.data_loader import DataLoader


class ImageDataLoader(DataLoader):
    def __init__(
            self,
            batch_size: int,
            data_dir: str,
            seed: Optional[int] = None,
            progress_bar: bool = False,
    ):
        super().__init__(batch_size, data_dir, seed, progress_bar)

    def iter_epoch_data(
            self,
    ):
        file_names = self._read_file_names()



if __name__ == "__main__":
    gatherer = ImageDataLoader(
        batch_size=1,
        data_dir="data/block_party/train", seed=0,
        progress_bar=True
    )
    gatherer.iter_epoch_data()