import matplotlib.pyplot as plt


def visualize_dimensions(dimensions, test_results):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    markers = ["o", "s", "^"]

    for idx, model in enumerate(test_results):
        accuracy = [test_results[model][f"{model}-{d}"].test_acc for d in dimensions]
        loss = [test_results[model][f"{model}-{d}"].test_loss for d in dimensions]

        marker = markers[idx % len(markers)]

        ax1.plot(dimensions, loss, label=model, marker=marker)
        ax2.plot(dimensions, accuracy, label=model, marker=marker)

    ax1.set_xscale("log")
    ax1.set_xlabel("Dimension")
    ax1.set_ylabel("Test Loss")
    ax1.set_title("Loss vs Dimension")
    ax1.legend()
    ax1.grid(True, which="both", linestyle="--", alpha=0.6)

    ax2.set_xscale("log")
    ax2.set_xlabel("Dimension")
    ax2.set_ylabel("Test Accuracy")
    ax2.set_title("Accuracy vs Dimension")
    ax2.legend()
    ax2.grid(True, which="both", linestyle="--", alpha=0.6)

    plt.tight_layout()
    return fig
