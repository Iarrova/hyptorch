import matplotlib.pyplot as plt
from src.train import TrainResult


def plot_training_metrics(results: dict[str, TrainResult]):
    fig, (ax_train, ax_val) = plt.subplots(2, 1, figsize=(16, 16))

    _plot_metrics_subplot(results, ax_train, "train", "Training Metrics")
    _plot_metrics_subplot(results, ax_val, "val", "Validation Metrics")

    plt.tight_layout()
    return fig


def _plot_metrics_subplot(results: dict[str, TrainResult], ax_primary, metric_prefix, title):
    # Create secondary y-axis for accuracy
    ax_secondary = ax_primary.twinx()

    for model in results:
        history = results[model].history
        loss_key = f"{metric_prefix}_loss"
        acc_key = f"{metric_prefix}_acc"

        epochs = range(1, len(history[loss_key]) + 1)

        # Plot loss on primary y-axis
        ax_primary.plot(epochs, history[loss_key], linestyle="-", marker="o", label=f"{model} (Loss)")

        # Plot accuracy on secondary y-axis (converted to percentage)
        acc_percent = [acc * 100 for acc in history[acc_key]]
        ax_secondary.plot(epochs, acc_percent, linestyle="--", marker="s", label=f"{model} (Acc)")

    ax_primary.set_xlabel("Epoch")
    ax_primary.set_ylabel("Loss")
    ax_secondary.set_ylabel("Accuracy (%)")
    ax_primary.set_title(title)
    ax_primary.grid(True, alpha=0.3)

    # Combine legends from both y-axes
    lines_primary, labels_primary = ax_primary.get_legend_handles_labels()
    lines_secondary, labels_secondary = ax_secondary.get_legend_handles_labels()
    ax_primary.legend(
        lines_primary + lines_secondary, labels_primary + labels_secondary, loc="upper right", bbox_to_anchor=(1.25, 1)
    )
