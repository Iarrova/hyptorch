import torch
from src.config import Config
from src.datasets.base import BaseDataModule
from torchvision import datasets, transforms


class CIFAR10DataModule(BaseDataModule):
    _DATASET_NAME = "CIFAR10"
    _NUM_CLASSES = 10

    def __init__(self, config: Config):
        super().__init__(config)

    def prepare_data(self):
        datasets.CIFAR10(self.config.dataset.DATA_DIR, train=True, download=True)
        datasets.CIFAR10(self.config.dataset.DATA_DIR, train=False, download=True)

    def setup(self, stage: str | None = None):
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        if stage == "fit" or stage is None:
            cifar10 = datasets.CIFAR10(self.config.dataset.DATA_DIR, train=True, transform=self.transform)

            self.cifar10_train, self.cifar10_val = torch.utils.data.random_split(
                cifar10, lengths=[1 - self.config.dataset.VALIDATION_SIZE, self.config.dataset.VALIDATION_SIZE]
            )

        if stage == "test" or stage is None:
            self.cifar10_test = datasets.CIFAR10(self.config.dataset.DATA_DIR, train=False, transform=self.transform)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.cifar10_train, batch_size=self.config.dataset.BATCH_SIZE, shuffle=True, num_workers=5, pin_memory=True
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.cifar10_val, batch_size=self.config.dataset.BATCH_SIZE, shuffle=False, num_workers=5, pin_memory=True
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.cifar10_test, batch_size=self.config.dataset.BATCH_SIZE, shuffle=False, num_workers=5, pin_memory=True
        )

    def get_class_names(self):
        if self.class_names is None:
            dataset = datasets.CIFAR10(self.config.dataset.DATA_DIR, train=True, download=False)
            self.class_names = dataset.classes
        return self.class_names
