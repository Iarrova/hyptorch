import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    from enum import StrEnum
    from typing import Optional

    import matplotlib.pyplot as plt
    import numpy as np
    import pytorch_lightning as pl
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint, ModelSummary
    from torchmetrics import Accuracy
    from torchvision import datasets, models, transforms
    return (
        Accuracy,
        EarlyStopping,
        F,
        LearningRateMonitor,
        ModelCheckpoint,
        ModelSummary,
        Optional,
        StrEnum,
        datasets,
        models,
        nn,
        optim,
        pl,
        torch,
        transforms,
    )


@app.cell
def _():
    from hyptorch import HyperbolicMLR, HypLinear, PoincareBall, ToPoincare, seed_everything
    return HypLinear, HyperbolicMLR, PoincareBall, ToPoincare, seed_everything


@app.cell
def _(pl, seed_everything):
    # All configuration variables for the notebook. Feel free to modify them as you experiment

    # Data parameters
    DATA_DIR: str = "./examples/outputs/data"
    BATCH_SIZE: int = 128
    VALIDATION_SIZE: float = 0.2

    # Model parameters
    CONV1_CHANNELS: int = 32
    CONV2_CHANNELS: int = 64
    HIDDEN_DIM: int = 128
    HYPERBOLIC_DIM: int = 2  # 2D for visualization
    NUM_CLASSES: int = 10

    # Hyperbolic parameters
    CURVATURE: float = 1.0

    # Training parameters
    MAX_EPOCHS: int = 10
    LEARNING_RATE: float = 1e-3
    WEIGHT_DECAY: float = 1e-4
    PATIENCE: int = 5

    # System
    SEED: int = 42

    pl.seed_everything(SEED)
    seed_everything(SEED)
    return (
        BATCH_SIZE,
        CURVATURE,
        DATA_DIR,
        HIDDEN_DIM,
        HYPERBOLIC_DIM,
        LEARNING_RATE,
        MAX_EPOCHS,
        NUM_CLASSES,
        PATIENCE,
        VALIDATION_SIZE,
        WEIGHT_DECAY,
    )


@app.cell
def _(StrEnum, mo):
    class Backbones(StrEnum):
        VGG16 = "VGG16"
        RESNET50 = "ResNet50"
        EFFICIENTNET_V2 = "EfficientNetV2"

    backbone_selector = mo.ui.dropdown(
        options=[Backbones.VGG16, Backbones.RESNET50, Backbones.EFFICIENTNET_V2],
        value=Backbones.EFFICIENTNET_V2,
        label="Select Backbone:",
    )
    backbone_selector
    return Backbones, backbone_selector


@app.cell
def _(mo):
    mo.md(
        r"""
    # Hyperbolic Image Classification with CIFAR10

    This notebook demonstrates how the `hyptorch` library integrates with PyTorch, by showing a simple example on building a hyperbolic neural networks for image classification using the CIFAR10 dataset.
    """
    )
    return


@app.cell
def _(
    BATCH_SIZE: int,
    DATA_DIR: str,
    Optional,
    VALIDATION_SIZE: float,
    datasets,
    pl,
    torch,
    transforms,
):
    class CIFAR10DataModule(pl.LightningDataModule):
        def __init__(self):
            super().__init__()

        def prepare_data(self):
            datasets.CIFAR10(DATA_DIR, train=True, download=True)
            datasets.CIFAR10(DATA_DIR, train=False, download=True)

        def setup(self, stage: Optional[str] = None):
            self.transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
            )

            if stage == "fit" or stage is None:
                cifar10 = datasets.CIFAR10(DATA_DIR, train=True, transform=self.transform)

                self.cifar10_train, self.cifar10_val = torch.utils.data.random_split(
                    cifar10, lengths=[1 - VALIDATION_SIZE, VALIDATION_SIZE]
                )

            if stage == "test" or stage is None:
                self.cifar10_test = datasets.CIFAR10(DATA_DIR, train=False, transform=self.transform)

        def train_dataloader(self):
            return torch.utils.data.DataLoader(
                self.cifar10_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=5, pin_memory=True
            )

        def val_dataloader(self):
            return torch.utils.data.DataLoader(
                self.cifar10_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=5, pin_memory=True
            )

        def test_dataloader(self):
            return torch.utils.data.DataLoader(
                self.cifar10_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=5, pin_memory=True
            )

    # Initialize data module
    datamodule = CIFAR10DataModule()
    datamodule.prepare_data()
    datamodule.setup()
    return (datamodule,)


@app.cell
def _(
    Accuracy,
    Backbones,
    F,
    LEARNING_RATE: float,
    NUM_CLASSES: int,
    WEIGHT_DECAY: float,
    models,
    nn,
    optim,
    pl,
):
    class BaseModel(pl.LightningModule):
        def __init__(self, backbone: str):
            super().__init__()

            self.backbone = backbone
            self._build_backbone()

            self.train_acc = Accuracy(task="multiclass", num_classes=NUM_CLASSES)
            self.val_acc = Accuracy(task="multiclass", num_classes=NUM_CLASSES)
            self.test_acc = Accuracy(task="multiclass", num_classes=NUM_CLASSES)

        def _build_backbone(self):
            if self.backbone == Backbones.VGG16:
                backbone_model = models.vgg16(weights=None)
                self.backbone_feature_dim = backbone_model.classifier[6].in_features
                backbone_model.classifier = nn.Sequential(*list(backbone_model.classifier.children())[:-1])

            elif self.backbone == Backbones.RESNET50:
                backbone_model = models.resnet50(weights=None)
                self.backbone_feature_dim = backbone_model.fc.in_features
                backbone_model.fc = nn.Identity()

            elif self.backbone == Backbones.EFFICIENTNET_V2:
                backbone_model = models.efficientnet_v2_s(weights=None)
                self.backbone_feature_dim = backbone_model.classifier[1].in_features
                backbone_model.classifier = nn.Identity()

            else:
                raise ValueError(f"Unknown backbone: {self.backbone}")

            self.backbone = nn.Sequential(backbone_model, nn.Flatten(start_dim=1))

            # Uncomment to freeze the feature extractor
            # for param in self.backbone.parameters():
            #     param.requires_grad = False

        def _forward_backbone(self, x):
            x = self.backbone(x)
            return x

        def forward(self, x):
            raise NotImplementedError("Subclasses must implement forward()")

        def training_step(self, batch, batch_idx):
            x, y = batch
            logits, embeddings = self(x)

            loss = F.cross_entropy(logits, y)
            acc = self.train_acc(logits, y)

            self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True)

            # Let subclasses add their specific metrics
            self._log_training_metrics(embeddings, logits, y)

            return loss

        def validation_step(self, batch, batch_idx):
            x, y = batch
            logits, embeddings = self(x)

            loss = F.cross_entropy(logits, y)
            acc = self.val_acc(logits, y)

            self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)

            # Let subclasses add their specific metrics
            self._log_validation_metrics(embeddings, logits, y)

            return loss

        def test_step(self, batch, batch_idx):
            x, y = batch
            logits, embeddings = self(x)

            loss = F.cross_entropy(logits, y)
            acc = self.test_acc(logits, y)

            self.log("test_loss", loss, on_step=False, on_epoch=True)
            self.log("test_acc", acc, on_step=False, on_epoch=True)

            return loss

        def configure_optimizers(self):
            optimizer = optim.Adam(self.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="max",
                factor=0.5,
                patience=3,
            )

            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "monitor": "val_acc", "frequency": 1},
            }

        def _log_training_metrics(self, embeddings, logits, labels):
            """Hook for subclass-specific training metrics."""
            pass

        def _log_validation_metrics(self, embeddings, logits, labels):
            """Hook for subclass-specific validation metrics."""
            pass
    return (BaseModel,)


@app.cell
def _(
    BaseModel,
    F,
    HIDDEN_DIM: int,
    HYPERBOLIC_DIM: int,
    NUM_CLASSES: int,
    nn,
    torch,
):
    class EuclideanModel(BaseModel):
        def __init__(self, backbone: str):
            super().__init__(backbone=backbone)
            self._build_euclidean_layers()

        def _build_euclidean_layers(self):
            self.fc1 = nn.Linear(self.backbone_feature_dim, HIDDEN_DIM)
            self.fc2 = nn.Linear(HIDDEN_DIM, HYPERBOLIC_DIM)

            self.classifier = nn.Linear(HYPERBOLIC_DIM, NUM_CLASSES)
            self.dropout = nn.Dropout(0.2)

        def forward(self, x):
            x = self._forward_backbone(x)
            x = F.relu(self.fc1(x))
            features = self.fc2(x)
            features_with_dropout = self.dropout(features)
            logits = self.classifier(features_with_dropout)

            return logits, features

        def _log_training_metrics(self, embeddings, logits, labels):
            probs = F.softmax(logits, dim=1)
            max_probs = torch.max(probs, dim=1)[0]
            self.log("train_prediction_confidence_mean", max_probs.mean(), on_step=False, on_epoch=True)

        def _log_validation_metrics(self, embeddings, logits, labels):
            probs = F.softmax(logits, dim=1)
            max_probs = torch.max(probs, dim=1)[0]
            self.log("val_prediction_confidence_mean", max_probs.mean(), on_step=False, on_epoch=True)
    return (EuclideanModel,)


@app.cell
def _(
    BaseModel,
    CURVATURE: float,
    F,
    HIDDEN_DIM: int,
    HYPERBOLIC_DIM: int,
    HypLinear,
    HyperbolicMLR,
    NUM_CLASSES: int,
    PoincareBall,
    ToPoincare,
    nn,
    torch,
):
    class HyperbolicModel(BaseModel):
        def __init__(self, backbone: str):
            super().__init__(backbone=backbone)

            self.manifold = PoincareBall(curvature=CURVATURE, trainable_curvature=True)
            self._build_hyperbolic_layers()

        def _build_hyperbolic_layers(self):
            self.to_poincare = ToPoincare(self.manifold)

            self.fc1 = HypLinear(self.backbone_feature_dim, HIDDEN_DIM, self.manifold)
            self.fc2 = HypLinear(HIDDEN_DIM, HYPERBOLIC_DIM, self.manifold)

            self.classifier = HyperbolicMLR(
                ball_dim=HYPERBOLIC_DIM, n_classes=NUM_CLASSES, manifold=self.manifold
            )
            self.dropout = nn.Dropout(0.2)

        def forward(self, x):
            x = self._forward_backbone(x)
            x = self.to_poincare(x)
            x = F.relu(self.fc1(x))
            features = self.fc2(x)
            features_with_dropout = self.dropout(features)
            logits = self.classifier(features_with_dropout)

            return logits, features

        def _log_training_metrics(self, embeddings, logits, labels):
            distances_to_boundary = 1.0 - torch.norm(embeddings, dim=1)
            self.log(
                "train_distance_to_boundary_mean", distances_to_boundary.mean(), on_step=False, on_epoch=True
            )

        def _log_validation_metrics(self, embeddings, logits, labels):
            distances_to_boundary = 1.0 - torch.norm(embeddings, dim=1)
            self.log(
                "val_distance_to_boundary_mean", distances_to_boundary.mean(), on_step=False, on_epoch=True
            )
    return (HyperbolicModel,)


@app.cell
def _(
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
    PATIENCE: int,
):
    def setup_callbacks():
        checkpoint_callback = ModelCheckpoint(
            dirpath="./examples/outputs/checkpoints",
            filename="hyperbolic-mnist-{epoch:02d}-{val_acc:.4f}",
            monitor="val_loss",
            mode="min",
        )

        # Early stopping - prevent overfitting
        early_stop_callback = EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=PATIENCE,
            verbose=True,
        )

        lr_monitor = LearningRateMonitor(logging_interval="epoch")

        # Model summary - show full model architecture
        model_summary = ModelSummary(max_depth=-1)

        callbacks = [checkpoint_callback, early_stop_callback, lr_monitor, model_summary]

        return callbacks
    return (setup_callbacks,)


@app.cell
def _(
    EuclideanModel,
    HyperbolicModel,
    MAX_EPOCHS: int,
    backbone_selector,
    datamodule,
    pl,
    setup_callbacks,
):
    def compare_models(datamodule, backbone: str):
        results = {}

        models = [EuclideanModel(backbone=backbone), HyperbolicModel(backbone=backbone)]
        for model in models:
            callbacks = setup_callbacks()

            trainer = pl.Trainer(max_epochs=MAX_EPOCHS, callbacks=callbacks, enable_progress_bar=False)
            trainer.fit(model, datamodule)
            test_results = trainer.test(ckpt_path="best", datamodule=datamodule)

            results[model.__class__.__name__] = {
                "test_acc": test_results[0]["test_acc"],
                "best_val_acc": trainer.callback_metrics.get("val_acc", 0.0),
                "model": model,
                "trainer": trainer,
            }

        return results

    results = compare_models(datamodule=datamodule, backbone=backbone_selector.value)
    results
    return (results,)


@app.cell
def _(results):
    results["HyperbolicModel"]["model"].manifold.curvature
    return


if __name__ == "__main__":
    app.run()
