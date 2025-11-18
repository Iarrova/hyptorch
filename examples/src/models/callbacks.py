from pytorch_lightning.callbacks import (
    Callback,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from src.config import Config


class MetricsHistoryCallback(Callback):
    def __init__(self):
        self.history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    def on_train_epoch_end(self, trainer, pl_module):
        if "train_loss" in trainer.logged_metrics:
            self.history["train_loss"].append(trainer.logged_metrics["train_loss"].item())
        if "train_acc" in trainer.logged_metrics:
            self.history["train_acc"].append(trainer.logged_metrics["train_acc"].item())

    def on_validation_epoch_end(self, trainer, pl_module):
        if "val_loss" in trainer.logged_metrics:
            self.history["val_loss"].append(trainer.logged_metrics["val_loss"].item())
        if "val_acc" in trainer.logged_metrics:
            self.history["val_acc"].append(trainer.logged_metrics["val_acc"].item())


def setup_callbacks(config: Config, experiment_name: str):
    checkpoint_callback = ModelCheckpoint(
        dirpath="./examples/outputs/checkpoints",
        filename=f"{experiment_name}-{{epoch:02d}}-{{val_acc:.4f}}",
        monitor="val_loss",
        mode="min",
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=config.training.EARLY_STOPPING_PATIENCE,
        verbose=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    callbacks = [checkpoint_callback, early_stop_callback, lr_monitor]

    return callbacks
