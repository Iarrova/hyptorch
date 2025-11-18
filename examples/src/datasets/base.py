from abc import ABC, abstractmethod

import pytorch_lightning as pl
from src.config import Config
from torch.utils.data import DataLoader


class BaseDataModule(pl.LightningDataModule, ABC):
    _DATASET_NAME = ""
    _NUM_CLASSES = None

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.class_names = None

    @property
    def dataset_name(self):
        return self._DATASET_NAME

    @property
    def num_classes(self):
        return self._NUM_CLASSES

    @abstractmethod
    def prepare_data(self):
        pass

    @abstractmethod
    def setup(self, stage: str | None = None):
        pass

    @abstractmethod
    def train_dataloader(self) -> DataLoader:
        pass

    @abstractmethod
    def val_dataloader(self) -> DataLoader:
        pass

    @abstractmethod
    def test_dataloader(self) -> DataLoader:
        pass

    @abstractmethod
    def get_class_names(self):
        pass
