import marimo

__generated_with = "0.17.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import pytorch_lightning as pl
    from collections import defaultdict

    from src.config import Config, DatasetConfig, HyperbolicConfig, ModelConfig, TrainingConfig
    from src.datasets import create_dataset
    from src.enums import Dataset, Backbone
    from src.models import HybridModel, HyperbolicModel
    from src.test import test_model
    from src.train import train_model
    from src.visualization.curvature import visualize_curvatures

    from hyptorch import seed_everything
    return (
        Backbone,
        Config,
        Dataset,
        DatasetConfig,
        HybridModel,
        HyperbolicConfig,
        HyperbolicModel,
        ModelConfig,
        TrainingConfig,
        create_dataset,
        defaultdict,
        pl,
        seed_everything,
        test_model,
        train_model,
        visualize_curvatures,
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
    # Curvature Hyperparameter for Hyperbolic Models

    In this notebook, we explore the effect of choosing an appropiate curvature hyperparameter when training hyperbolic models. This is a **VERY IMPORTANT** step, and will make or break your model.

    The library provides a way to make this hyperparameter trainable, in order to alleviate some of this.
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
            EMBEDDING_DIMENSION=32, 
            PRETRAINED=pretrained_checkbox.value
        ),
        hyperbolic=HyperbolicConfig(
            CURVATURE=0.05,
            TRAINABLE_CURVATURE=False,
            CURVATURE_LEARNING_RATE=0.1,
        ),
        training=TrainingConfig(
            MAX_EPOCHS=1,
            LEARNING_RATE=1e-3,
            WEIGHT_DECAY=1e-4,
            EARLY_STOPPING_PATIENCE=3,
        ),
    )
    return (config,)


@app.cell
def _(config, create_dataset, dataset_selector):
    datamodule = create_dataset(dataset_selector.value, config)
    datamodule.prepare_data()
    datamodule.setup()
    return (datamodule,)


@app.cell
def _(mo):
    CURVATURES = [0.0005, 0.005, 0.05, 0.5, 1, 2, 5, 10]
    mo.md(f"""
    ## Curvatures
    The following curvatures will be used to train the models:

    {CURVATURES}
    """)
    return (CURVATURES,)


@app.cell
def _(mo):
    train_button = mo.ui.run_button(label="Train Models")
    train_button
    return (train_button,)


@app.cell
def _(
    CURVATURES,
    HybridModel,
    HyperbolicModel,
    backbone_selector,
    config,
    datamodule,
    defaultdict,
    mo,
    train_button,
    train_model,
):
    def train_models(config, datamodule, curvatures):
        results = defaultdict(dict)

        with mo.status.progress_bar(total=2*len(curvatures)) as bar:
            for curvature in curvatures:
                config.hyperbolic.CURVATURE = curvature 

                models = [
                    HyperbolicModel(config, backbone_selector.value, datamodule.num_classes),
                    HybridModel(config, backbone_selector.value, datamodule.num_classes),
                ]

                for model in models:
                    bar.update(increment=1, title=f"Training {model.name}", subtitle=f"Curvature: {curvature}")

                    train_result = train_model(config, datamodule, model)
                    results[model.name][f"{model.name}-{curvature}"] = train_result

        return dict(results)

    results = train_models(config, datamodule, CURVATURES) if train_button.value else None
    return (results,)


@app.cell
def _(datamodule, defaultdict, results, test_model):
    def test_models(datamodule, results):
        test_results = defaultdict(dict)

        for model in results:
            for model_curvature in results[model]:
                test_result = test_model(datamodule, results[model][model_curvature])
                test_results[model][model_curvature] = test_result

        return dict(test_results)

    test_results = test_models(datamodule, results) if results else None
    return (test_results,)


@app.cell
def _(CURVATURES, test_results, visualize_curvatures):
    visualize_curvatures(CURVATURES, test_results) if test_results else None
    return


if __name__ == "__main__":
    app.run()
