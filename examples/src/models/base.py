from abc import ABC, abstractmethod

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.optim as optim
from src.backbones import create_backbone
from src.config import Config
from src.enums import Backbone
from torchmetrics import Accuracy


class BaseModel(pl.LightningModule, ABC):
    _NAME = "BaseModel"

    def __init__(self, config: Config, backbone: Backbone, num_classes: int):
        super().__init__()
        self.config = config
        self.num_classes = num_classes

        self.backbone = self.create_backbone(backbone, config.model.PRETRAINED)
        self.backbone_feature_dim = self.backbone.feature_dim

        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)

    @property
    def name(self):
        return self._NAME

    def create_backbone(self, backbone: Backbone, pretrained: bool = True):
        return create_backbone(backbone, pretrained=pretrained)

    def _forward_backbone(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    @abstractmethod
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pass

    def _compute_step(self, batch):
        inputs, targets = batch
        logits, embeddings = self(inputs)
        return logits, embeddings, targets

    def _calculate_loss(self, logits, labels):
        return F.cross_entropy(logits, labels)

    def training_step(self, batch, batch_idx):
        logits, _, targets = self._compute_step(batch)
        loss = self._calculate_loss(logits, targets)

        acc = self.train_acc(logits, targets)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        logits, _, targets = self._compute_step(batch)
        loss = self._calculate_loss(logits, targets)

        acc = self.val_acc(logits, targets)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        logits, _, targets = self._compute_step(batch)
        loss = self._calculate_loss(logits, targets)

        acc = self.test_acc(logits, targets)
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.log("test_acc", acc, on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self):
        curvature_params = [p for n, p in self.named_parameters() if "_curvature" in n]
        other_params = [p for n, p in self.named_parameters() if "_curvature" not in n]

        optimizer = optim.AdamW(
            [
                {"params": other_params, "lr": self.config.training.LEARNING_RATE},
                {"params": curvature_params, "lr": self.config.hyperbolic.CURVATURE_LEARNING_RATE},
            ],
            weight_decay=self.config.training.WEIGHT_DECAY,
        )

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss", "frequency": 1},
        }
