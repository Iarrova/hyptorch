import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim

    from torchvision import datasets, transforms
    import matplotlib.pyplot as plt
    import numpy as np
    from typing import Tuple, Optional, Dict, Any

    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor, RichProgressBar, Callback
    from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
    from pytorch_lightning.utilities.model_summary import ModelSummary
    from torchmetrics import Accuracy
    return (
        Accuracy,
        EarlyStopping,
        F,
        LearningRateMonitor,
        ModelCheckpoint,
        Optional,
        datasets,
        nn,
        np,
        optim,
        pl,
        plt,
        torch,
        transforms,
    )


@app.cell
def _():
    from hyptorch import PoincareBall, HypLinear, ToPoincare, FromPoincare, HyperbolicMLR
    return HypLinear, HyperbolicMLR, PoincareBall, ToPoincare


@app.cell
def _(pl):
    # All configuration variables for the notebook. Feel free to modify them as you experiment

    # Data parameters
    DATA_DIR: str = './data'
    BATCH_SIZE: int = 128
    VALIDATION_SIZE: float = 0.2

    # Model parameters
    CONV1_CHANNELS: int = 32
    CONV2_CHANNELS: int = 64
    HIDDEN_DIM: int = 128
    HYPERBOLIC_DIM: int = 2  # 2D for visualization
    NUM_CLASSES: int = 10

    # Hyperbolic parameters
    CURVATURE: float = 0.05

    # Training parameters
    MAX_EPOCHS: int = 10
    LEARNING_RATE: float = 1e-3
    WEIGHT_DECAY: float = 1e-4
    PATIENCE: int = 5

    # System
    SEED: int = 42

    pl.seed_everything(SEED)
    return (
        BATCH_SIZE,
        CONV1_CHANNELS,
        CONV2_CHANNELS,
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
def _(mo):
    mo.md(
        r"""
    # Hyperbolic Image Classification with MNIST

    This notebook demonstrates how the `hyptorch` library integrates with PyTorch, by showing a simple example on building a hyperbolic neural networks for image classification using the MNIST dataset.
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
    class MNISTDataModule(pl.LightningDataModule):    
        def __init__(self):
            super().__init__()

        def prepare_data(self):
            datasets.MNIST(DATA_DIR, train=True, download=True)
            datasets.MNIST(DATA_DIR, train=False, download=True)

        def setup(self, stage: Optional[str] = None):
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])

            if stage == "fit" or stage is None:
                mnist = datasets.MNIST(DATA_DIR, train=True, transform=self.transform)

                self.mnist_train, self.mnist_val = torch.utils.data.random_split(
                    mnist, lengths=[1 - VALIDATION_SIZE, VALIDATION_SIZE]
                )

            if stage == "test" or stage is None:
                self.mnist_test = datasets.MNIST(DATA_DIR, train=False, transform=self.transform)

        def train_dataloader(self):
            return torch.utils.data.DataLoader(
                self.mnist_train,
                batch_size=BATCH_SIZE,
                shuffle=True,
                num_workers=5,
                pin_memory=True
            )

        def val_dataloader(self):
            return torch.utils.data.DataLoader(
                self.mnist_val,
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=5,
                pin_memory=True
            )

        def test_dataloader(self):
            return torch.utils.data.DataLoader(
                self.mnist_test,
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=5,
                pin_memory=True
            )

    # Initialize data module
    datamodule = MNISTDataModule()
    datamodule.prepare_data()
    datamodule.setup()
    return (datamodule,)


@app.cell
def _(nn):
    class ConvBlock(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size=3, pool_size=2):
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=pool_size, stride=pool_size)
            )

        def forward(self, x):
            return self.block(x)
    return (ConvBlock,)


@app.cell
def _(
    Accuracy,
    CONV1_CHANNELS: int,
    CONV2_CHANNELS: int,
    ConvBlock,
    F,
    LEARNING_RATE: float,
    NUM_CLASSES: int,
    WEIGHT_DECAY: float,
    nn,
    optim,
    pl,
    torch,
):
    class BaseMNISTModule(pl.LightningModule):    
        def __init__(self):
            super().__init__()

            # Build shared CNN backbone
            self._build_backbone()

            # Shared metrics
            self.train_acc = Accuracy(task='multiclass', num_classes=NUM_CLASSES)
            self.val_acc = Accuracy(task='multiclass', num_classes=NUM_CLASSES)
            self.test_acc = Accuracy(task='multiclass', num_classes=NUM_CLASSES)

            # Storage for embeddings
            self.validation_embeddings = []
            self.validation_labels = []

        def _build_backbone(self):
            self.backbone = nn.Sequential(
                ConvBlock(1, CONV1_CHANNELS, kernel_size=3),
                ConvBlock(CONV1_CHANNELS, CONV2_CHANNELS, kernel_size=3),
                nn.Flatten(start_dim=1)
            )

        def _forward_backbone(self, x):
            x = self.backbone(x)

            return x

        def forward(self, x):
            raise NotImplementedError("Subclasses must implement forward()")

        def _compute_shared_metrics(self, embeddings):
            """Compute metrics that work for both hyperbolic and euclidean embeddings."""
            embedding_norms = torch.norm(embeddings, dim=1)
            return {
                'embedding_norm_mean': embedding_norms.mean(),
                'embedding_norm_std': embedding_norms.std(),
                'embedding_norm_max': embedding_norms.max(),
                'embedding_norm_min': embedding_norms.min()
            }

        def training_step(self, batch, batch_idx):
            x, y = batch
            logits, embeddings = self(x)

            loss = F.cross_entropy(logits, y)
            acc = self.train_acc(logits, y)
            shared_metrics = self._compute_shared_metrics(embeddings)

            self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True)

            for name, value in shared_metrics.items():
                self.log(f'train_{name}', value, on_step=False, on_epoch=True)

            # Let subclasses add their specific metrics
            self._log_training_metrics(embeddings, logits, y)

            return loss

        def validation_step(self, batch, batch_idx):
            x, y = batch
            logits, embeddings = self(x)

            # Calculate loss and accuracy
            loss = F.cross_entropy(logits, y)
            acc = self.val_acc(logits, y)

            # Store embeddings for visualization (first batch only)
            if batch_idx == 0:
                self.validation_embeddings = embeddings.detach().cpu()
                self.validation_labels = y.detach().cpu()

            shared_metrics = self._compute_shared_metrics(embeddings)

            self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True)

            for name, value in shared_metrics.items():
                self.log(f'val_{name}', value, on_step=False, on_epoch=True)

            # Let subclasses add their specific metrics
            self._log_validation_metrics(embeddings, logits, y)

            return loss

        def test_step(self, batch, batch_idx):
            x, y = batch
            logits, embeddings = self(x)

            loss = F.cross_entropy(logits, y)
            acc = self.test_acc(logits, y)

            self.log('test_loss', loss, on_step=False, on_epoch=True)
            self.log('test_acc', acc, on_step=False, on_epoch=True)

            return loss

        def configure_optimizers(self):
            optimizer = optim.Adam(
                self.parameters(), 
                lr=LEARNING_RATE,
                weight_decay=WEIGHT_DECAY
            )

            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='max',
                factor=0.5,
                patience=3,
            )

            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val_acc',
                    'frequency': 1
                }
            }

        def _log_training_metrics(self, embeddings, logits, labels):
            """Hook for subclass-specific training metrics."""
            pass

        def _log_validation_metrics(self, embeddings, logits, labels):
            """Hook for subclass-specific validation metrics."""
            pass
    return (BaseMNISTModule,)


@app.cell
def _(
    BaseMNISTModule,
    CONV2_CHANNELS: int,
    F,
    HIDDEN_DIM: int,
    HYPERBOLIC_DIM: int,
    NUM_CLASSES: int,
    nn,
    torch,
):
    class EuclideanMNISTModule(BaseMNISTModule):    
        def __init__(self):
            super().__init__()
            self._build_euclidean_layers()

        def _build_euclidean_layers(self):
            self.fc1 = nn.Linear(5 * 5 * CONV2_CHANNELS, HIDDEN_DIM)
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
            classifier_weight_norm = torch.norm(self.classifier.weight)
            self.log('train_classifier_weight_norm', classifier_weight_norm,
                    on_step=False, on_epoch=True)

            probs = F.softmax(logits, dim=1)
            max_probs = torch.max(probs, dim=1)[0]
            self.log('train_prediction_confidence_mean', max_probs.mean(),
                    on_step=False, on_epoch=True)

        def _log_validation_metrics(self, embeddings, logits, labels):
            probs = F.softmax(logits, dim=1)
            max_probs = torch.max(probs, dim=1)[0]
            self.log('val_prediction_confidence_mean', max_probs.mean(),
                    on_step=False, on_epoch=True)
    return (EuclideanMNISTModule,)


@app.cell
def _(
    BaseMNISTModule,
    CONV2_CHANNELS: int,
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
    class HyperbolicMNISTModule(BaseMNISTModule):    
        def __init__(self):
            super().__init__()

            self.hyperbolic_model = PoincareBall(curvature=CURVATURE, trainable_curvature=True)
            self._build_hyperbolic_layers()

        def _build_hyperbolic_layers(self):
            self.to_poincare = ToPoincare(self.hyperbolic_model)
            self.fc1 = HypLinear(5 * 5 * CONV2_CHANNELS, HIDDEN_DIM, self.hyperbolic_model)
            self.fc2 = HypLinear(HIDDEN_DIM, HYPERBOLIC_DIM, self.hyperbolic_model)

            self.classifier = HyperbolicMLR(
                ball_dim=HYPERBOLIC_DIM,
                n_classes=NUM_CLASSES,
                model=self.hyperbolic_model
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
            self.log('train_distance_to_boundary_mean', distances_to_boundary.mean(),
                    on_step=False, on_epoch=True)

        def _log_validation_metrics(self, embeddings, logits, labels):
            distances_to_boundary = 1.0 - torch.norm(embeddings, dim=1)
            self.log('val_distance_to_boundary_mean', distances_to_boundary.mean(),
                    on_step=False, on_epoch=True)
    return (HyperbolicMNISTModule,)


@app.cell
def _(EarlyStopping, LearningRateMonitor, ModelCheckpoint, PATIENCE: int):
    def setup_callbacks():
        checkpoint_callback = ModelCheckpoint(
            dirpath=f"./examples/checkpoints",
            filename="hyperbolic-mnist-{epoch:02d}-{val_acc:.4f}",
            monitor='val_loss',
            mode='min',
        )

        # Early stopping - prevent overfitting
        early_stop_callback = EarlyStopping(
            monitor='val_loss',
            mode='min',
            patience=PATIENCE,
            verbose=True,
        )

        lr_monitor = LearningRateMonitor(logging_interval='epoch')

        callbacks = [checkpoint_callback, early_stop_callback, lr_monitor]

        return callbacks
    return (setup_callbacks,)


@app.cell
def _(
    EuclideanMNISTModule,
    HyperbolicMNISTModule,
    MAX_EPOCHS: int,
    datamodule,
    pl,
    setup_callbacks,
):
    def compare_models(datamodule):
        results = {}

        models = [EuclideanMNISTModule(), HyperbolicMNISTModule()]
        for model in models:
            callbacks = setup_callbacks()

            trainer = pl.Trainer(
                max_epochs=MAX_EPOCHS,
                callbacks=callbacks,
                enable_progress_bar=False
            )
            trainer.fit(model, datamodule)
            test_results = trainer.test(ckpt_path='best', datamodule=datamodule)

            results[model.__class__.__name__] = {
                'test_acc': test_results[0]['test_acc'],
                'best_val_acc': trainer.callback_metrics.get('val_acc', 0.0),
                'model': model,
                'trainer': trainer
            }

        return results

    results = compare_models(datamodule=datamodule)
    results
    return (results,)


@app.cell
def _(
    BATCH_SIZE: int,
    CURVATURE: float,
    F,
    datamodule,
    np,
    plt,
    results,
    torch,
):
    def analyze_hyperbolic_embeddings(model, datamodule):
        model.eval()
        test_loader = datamodule.test_dataloader()

        embeddings_list = []
        labels_list = []
        predictions_list = []
        confidences_list = []

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                if len(embeddings_list) * BATCH_SIZE >= 1000:
                    break

                data = data.to(model.device)
                logits, embeddings = model(data)

                # Get predictions and confidences
                probs = F.softmax(logits, dim=1)
                predictions = torch.argmax(logits, dim=1)
                confidences = torch.max(probs, dim=1)[0]

                embeddings_list.append(embeddings.cpu())
                labels_list.append(target)
                predictions_list.append(predictions.cpu())
                confidences_list.append(confidences.cpu())

        # Combine all data
        embeddings = torch.cat(embeddings_list, dim=0)[:1000]
        labels = torch.cat(labels_list, dim=0)[:1000]
        predictions = torch.cat(predictions_list, dim=0)[:1000]
        confidences = torch.cat(confidences_list, dim=0)[:1000]

        # Convert to numpy
        emb_np = embeddings.numpy()
        lab_np = labels.numpy()
        pred_np = predictions.numpy()
        conf_np = confidences.numpy()

        # Calculate distances from origin (uncertainty measure)
        distances = np.linalg.norm(emb_np, axis=1)

        # Create visualization
        fig = plt.figure(figsize=(12, 10))

        # Plot: Embeddings colored by true labels
        ax = plt.subplot(1, 1, 1)
        colors = plt.cm.tab10(np.arange(10))
    
        # Draw Poincaré disk boundary
        circle = plt.Circle((0, 0), 1/torch.sqrt(torch.tensor(CURVATURE)).item(), fill=False, color='black', linewidth=2, linestyle='--', label='Poincaré boundary')
        ax.add_patch(circle)

        for digit in range(10):
            mask = lab_np == digit
            if mask.sum() > 0:
                ax.scatter(emb_np[mask, 0], emb_np[mask, 1], 
                           c=[colors[digit]], label=f'Digit {digit}', alpha=0.7, s=30)

        ax.set_xlim(-1.1/torch.sqrt(torch.tensor(CURVATURE)).item(), 1.1/torch.sqrt(torch.tensor(CURVATURE)).item())
        ax.set_ylim(-1.1/torch.sqrt(torch.tensor(CURVATURE)).item(), 1.1/torch.sqrt(torch.tensor(CURVATURE)).item())
        ax.set_aspect('equal')
        ax.set_xlabel('Dimension 1', fontsize=12)
        ax.set_ylabel('Dimension 2', fontsize=12)
        ax.set_title('Hyperbolic Embeddings in Poincaré Disk', fontweight='bold', fontsize=14)
        ax.legend(ncol=2, fontsize=9, loc='upper right')
        ax.grid(True, alpha=0.3)
    
        # Add statistics
        avg_distance = distances.mean()
        stats_text = f'Avg distance from origin: {avg_distance:.3f}\nMax distance: {distances.max():.3f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.show()

        return fig

    # Run advanced analysis
    analyze_hyperbolic_embeddings(
        results["HyperbolicMNISTModule"]["model"], datamodule
    )
    return


@app.cell
def _(BATCH_SIZE: int, F, datamodule, np, plt, results, torch):
    def analyze_euclidean_embeddings(model, datamodule):
        model.eval()
        test_loader = datamodule.test_dataloader()

        embeddings_list = []
        labels_list = []
        predictions_list = []
        confidences_list = []

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                if len(embeddings_list) * BATCH_SIZE >= 1000:
                    break

                data = data.to(model.device)
                logits, embeddings = model(data)

                # Get predictions and confidences
                probs = F.softmax(logits, dim=1)
                predictions = torch.argmax(logits, dim=1)
                confidences = torch.max(probs, dim=1)[0]

                embeddings_list.append(embeddings.cpu())
                labels_list.append(target)
                predictions_list.append(predictions.cpu())
                confidences_list.append(confidences.cpu())

        # Combine all data
        embeddings = torch.cat(embeddings_list, dim=0)[:1000]
        labels = torch.cat(labels_list, dim=0)[:1000]
        predictions = torch.cat(predictions_list, dim=0)[:1000]
        confidences = torch.cat(confidences_list, dim=0)[:1000]

        # Convert to numpy
        emb_np = embeddings.numpy()
        lab_np = labels.numpy()
        pred_np = predictions.numpy()
        conf_np = confidences.numpy()

        # Calculate distances from origin
        distances = np.linalg.norm(emb_np, axis=1)

        # Create visualization
        fig = plt.figure(figsize=(12, 10))

        # Plot: Embeddings colored by true labels
        ax = plt.subplot(1, 1, 1)
        colors = plt.cm.tab10(np.arange(10))

        for digit in range(10):
            mask = lab_np == digit
            if mask.sum() > 0:
                ax.scatter(emb_np[mask, 0], emb_np[mask, 1], 
                           c=[colors[digit]], label=f'Digit {digit}', alpha=0.7, s=30)

        # Set adaptive axis limits
        x_min, x_max = emb_np[:, 0].min(), emb_np[:, 0].max()
        y_min, y_max = emb_np[:, 1].min(), emb_np[:, 1].max()
        x_padding = (x_max - x_min) * 0.1
        y_padding = (y_max - y_min) * 0.1
        ax.set_xlim(x_min - x_padding, x_max + x_padding)
        ax.set_ylim(y_min - y_padding, y_max + y_padding)
    
        ax.set_aspect('equal')
        ax.set_xlabel('Dimension 1', fontsize=12)
        ax.set_ylabel('Dimension 2', fontsize=12)
        ax.set_title('Euclidean Embeddings', fontweight='bold', fontsize=14)
        ax.legend(ncol=2, fontsize=9, loc='upper right')
        ax.grid(True, alpha=0.3)
    
        # Add statistics
        avg_distance = distances.mean()
        std_distance = distances.std()
        stats_text = f'Avg distance from origin: {avg_distance:.3f}\nStd distance: {std_distance:.3f}\nMax distance: {distances.max():.3f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

        plt.tight_layout()
        plt.show()

        return fig

    analyze_euclidean_embeddings(
        results["EuclideanMNISTModule"]["model"], datamodule
    )
    return


if __name__ == "__main__":
    app.run()
