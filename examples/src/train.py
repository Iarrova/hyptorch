from dataclasses import dataclass

import pytorch_lightning as pl
from src.config import Config
from src.datasets import BaseDataModule
from src.models.base import BaseModel
from src.models.callbacks import MetricsHistoryCallback, setup_callbacks


@dataclass
class TrainResult:
    model: BaseModel
    trainer: pl.Trainer
    history: dict[str, list]


def train_model(config: Config, datamodule: BaseDataModule, model: BaseModel):
    callbacks = setup_callbacks(config, experiment_name=f"{model.name.lower()}-{datamodule.dataset_name.lower()}")
    metrics_callback = MetricsHistoryCallback()
    callbacks.append(metrics_callback)

    trainer = pl.Trainer(
        max_epochs=config.training.MAX_EPOCHS,
        callbacks=callbacks,
        enable_progress_bar=False,
        default_root_dir="./examples/outputs",
    )
    trainer.fit(model, datamodule)

    return TrainResult(model=model, trainer=trainer, history=metrics_callback.history)
