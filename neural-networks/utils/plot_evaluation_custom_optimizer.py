"""
Utility functions for plotting learning and accuracy curves.
"""

import matplotlib.pyplot as plt


def plot_learning_curve(
    train_losses_custom_optimizer,
    train_losses_optimizer,
    valid_losses,
    hyperparams,
    base_path=None,
):
    """
    Plot the learning curve showing training and validation losses across epochs.
    """
    plt.figure(figsize=(10, 6), dpi=110)
    plt.plot(train_losses_custom_optimizer, label="Training Loss (Custom optimizer)")
    plt.plot(train_losses_optimizer, label="Training Loss (Optimizer)")
    plt.plot(valid_losses, label="Validation Loss")
    plt.title("Learning Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    hyperparams_str = f"Epochs: {hyperparams[0]}, LR: {hyperparams[1]}"
    plt.text(
        0.5,
        0.95,
        hyperparams_str,
        horizontalalignment="center",
        verticalalignment="center",
        transform=plt.gca().transAxes,
    )
    plt.savefig(f"{base_path}/lc_{hyperparams[0]}_{hyperparams[1]}.png")
    plt.close()


def plot_accuracy_curve(
    train_accuracies_custom_optimizer,
    train_accuracies_optimizer,
    valid_accuracies,
    hyperparams,
    base_path=None,
):
    """
    Plot the accuracy curve showing training and validation accuracies across epochs.
    """
    plt.figure(figsize=(10, 6), dpi=110)
    plt.plot(
        train_accuracies_custom_optimizer, label="Training Accuracy (Custom Optimizer)"
    )
    plt.plot(train_accuracies_optimizer, label="Training Accuracy (Optimizer)")
    plt.plot(valid_accuracies, label="Validation Accuracy")
    plt.title("Accuracy Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid()
    hyperparams_str = f"Epochs: {hyperparams[0]}, LR: {hyperparams[1]}"
    plt.text(
        0.5,
        0.95,
        hyperparams_str,
        horizontalalignment="center",
        verticalalignment="center",
        transform=plt.gca().transAxes,
    )
    plt.savefig(f"{base_path}/acc_{hyperparams[0]}_{hyperparams[1]}.png")
    plt.close()
