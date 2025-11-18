import marimo

__generated_with = "0.17.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import matplotlib.pyplot as plt
    import pytorch_lightning as pl

    from src.config import Config, DatasetConfig, HyperbolicConfig, ModelConfig, TrainingConfig
    from src.datasets import create_dataset, denormalize
    from src.enums import Dataset, Backbone
    from src.models import EuclideanModel, HybridModel, HyperbolicModel
    from src.test import test_model
    from src.train import train_model
    from src.visualization.training import plot_training_metrics

    from hyptorch import seed_everything
    return (
        Backbone,
        Config,
        Dataset,
        DatasetConfig,
        EuclideanModel,
        HybridModel,
        HyperbolicConfig,
        HyperbolicModel,
        ModelConfig,
        TrainingConfig,
        create_dataset,
        denormalize,
        pl,
        plot_training_metrics,
        plt,
        seed_everything,
        test_model,
        train_model,
    )


@app.cell
def _(pl, seed_everything):
    SEED = 42

    pl.seed_everything(SEED)
    seed_everything(SEED)
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Hyperbolic Image Classification with CIFAR

    This notebook demonstrates how the `hyptorch` library integrates with PyTorch, by showing a simple example on building a hyperbolic neural networks for image classification using the CIFAR dataset.
    """)
    return


@app.cell
def _(Dataset, mo):
    dataset_selector = mo.ui.dropdown(options=[Dataset.CIFAR10, Dataset.CIFAR100], value=Dataset.CIFAR10, label="Dataset:")

    mo.md(f"""
    ## Dataset Settings
    {dataset_selector}
    """)
    return (dataset_selector,)


@app.cell
def _(Backbone, mo):
    backbone_selector = mo.ui.dropdown(
        options=[Backbone.VGG16, Backbone.ResNet50, Backbone.EfficientNetV2], 
        value=Backbone.EfficientNetV2, 
        label="Backbone:"
    )
    pretrained_checkbox = mo.ui.checkbox(value=True, label="Use Pretrained Weights")

    mo.md(f"""
    ## Model Architecture
    {backbone_selector} {pretrained_checkbox}
    """)
    return backbone_selector, pretrained_checkbox


@app.cell
def _(
    Config,
    DatasetConfig,
    HyperbolicConfig,
    ModelConfig,
    TrainingConfig,
    pretrained_checkbox,
):
    # Here you can change more advanced configuration if desired
    config = Config(
        dataset=DatasetConfig(
            DATA_DIR="./examples/outputs/data", 
            BATCH_SIZE=128, 
            VALIDATION_SIZE=0.2
        ),
        model=ModelConfig(
            EMBEDDING_DIMENSION=8, 
            PRETRAINED=pretrained_checkbox.value
        ),
        hyperbolic=HyperbolicConfig(
            CURVATURE=0.05,
            TRAINABLE_CURVATURE=True,
            CURVATURE_LEARNING_RATE=0.1,
        ),
        training=TrainingConfig(
            MAX_EPOCHS=1,
            LEARNING_RATE=1e-3,
            WEIGHT_DECAY=1e-4,
            EARLY_STOPPING_PATIENCE=10,
        ),
    )
    return (config,)


@app.cell
def _(mo):
    mo.md(r"""
    # Dataset Preview

    Before training the models let's visualize some sample images from the selected dataset to understand what we're working with.
    """)
    return


@app.cell
def _(config, create_dataset, dataset_selector):
    datamodule = create_dataset(dataset_selector.value, config)
    datamodule.prepare_data()
    datamodule.setup()
    return (datamodule,)


@app.cell
def _(datamodule, dataset_selector, denormalize, plt):
    def preview_dataset(datamodule):
        class_names = datamodule.get_class_names()
        train_loader = datamodule.train_dataloader()

        images, labels = next(iter(train_loader))

        fig, axes = plt.subplots(2, 5, figsize=(12, 5))
        fig.suptitle(f"{dataset_selector.value} Sample Images")

        for idx, ax in enumerate(axes.flat):
            if idx < len(images):
                img = denormalize(images[idx])
                ax.imshow(img)
                ax.set_title(f"{class_names[labels[idx]]}")
                ax.axis("off")

        plt.tight_layout()
        return fig

    preview_dataset(datamodule)
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Model Training

    Now let's train and compare three different model architectures:
    - **Euclidean Model**: Traditional neural network with Euclidean geometry.
    - **Hyperbolic Model**: Neural network using hyperbolic geometry for embeddings.
    - **Hybrid Model**: Combines both Euclidean and hyperbolic components.

    Click the button below to start training:
    """)
    return


@app.cell
def _(mo):
    train_button = mo.ui.run_button(label="Train Models")
    train_button
    return (train_button,)


@app.cell
def _(
    EuclideanModel,
    HybridModel,
    HyperbolicModel,
    backbone_selector,
    config,
    datamodule,
    mo,
    train_button,
    train_model,
):
    def train_models(config, datamodule):
        results = {}

        models = [
            EuclideanModel(config, backbone_selector.value, datamodule.num_classes),
            HyperbolicModel(config, backbone_selector.value, datamodule.num_classes),
            HybridModel(config, backbone_selector.value, datamodule.num_classes),
        ]

        with mo.status.progress_bar(total=3) as bar:
            for model in models:
                bar.update(increment=1, title=f"Training {model.name}", subtitle=f"Dataset: {datamodule.dataset_name}")

                train_result = train_model(config, datamodule, model)
                results[model.name] = train_result

        return results

    results = train_models(config, datamodule) if train_button.value else None
    return (results,)


@app.cell
def _(mo, plot_training_metrics, results):
    mo.vstack([
        mo.md("## Training Results"),
        plot_training_metrics(results) if results else mo.md("Press the `Train Models` button to visualize results"),
    ])
    return


@app.cell
def _(mo, results):
    mo.vstack([
        mo.md("# Model Testing"),
        mo.md("Finally, once we have our models trained, we compare their results over the testing set.") if results else mo.md("Train the models before visualizing testing results"),
    ])
    return


@app.cell
def _(datamodule, results, test_model):
    def test_models(datamodule, results):
        test_results = {} 

        for model in results:
            test_result = test_model(datamodule, results[model])
            test_results[model] = test_result

        return test_results


    test_results = test_models(datamodule, results) if results is not None else None
    return (test_results,)


@app.cell
def _(mo, test_results):
    def generate_comparison_table(test_results):
        table_data = []
        for model in test_results:
            result = test_results[model]
            table_data.append({
                "Model": model,
                "Test Accuracy": f"{result.test_acc * 100:.2f}%",
                "Test Loss": f"{result.test_loss:.4f}",
            })

        return mo.ui.table(table_data)

    generate_comparison_table(test_results) if test_results else None
    return


if __name__ == "__main__":
    app.run()
