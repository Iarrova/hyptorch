from src.config import Config
from src.datasets.base import BaseDataModule
from src.datasets.cifar10 import CIFAR10DataModule
from src.datasets.cifar100 import CIFAR100DataModule
from src.datasets.utils import denormalize
from src.enums import Dataset

__all__ = ["CIFAR10DataModule", "CIFAR100DataModule", "create_dataset", "denormalize"]


DATASET_MAPPING = {
    Dataset.CIFAR10: CIFAR10DataModule,
    Dataset.CIFAR100: CIFAR100DataModule,
}


def create_dataset(name: Dataset, config: Config) -> BaseDataModule:
    if name not in DATASET_MAPPING:
        raise ValueError(f"Dataset {name} is not supported. Available: {list(DATASET_MAPPING.keys())}")

    dataset_class = DATASET_MAPPING[name]
    return dataset_class(config)
