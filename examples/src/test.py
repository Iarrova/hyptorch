from dataclasses import dataclass

from src.datasets.base import BaseDataModule
from src.train import TrainResult


@dataclass
class TestResult:
    test_loss: float
    test_acc: float


def test_model(datamodule: BaseDataModule, train_result: TrainResult):
    test_result = train_result.trainer.test(ckpt_path="best", datamodule=datamodule)

    return TestResult(
        test_loss=test_result[0]["test_loss"],
        test_acc=test_result[0]["test_acc"],
    )
