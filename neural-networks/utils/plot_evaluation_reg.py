"""
Utility functions for plotting learning and accuracy curves.
"""

import matplotlib.pyplot as plt


def plot_learning_curve(train_losses, valid_losses, hyperparams, base_path=None):
    """
    Plot the learning curve showing training and validation losses across epochs.

    Parameters:
    - train_losses (list): Training loss values for each epoch.
    - valid_losses (list): Validation loss values for each epoch.
    - hyperparams (list): List containing hyperparameters used for the plot title.
    - base_path (str): Base path to save the plot as an image.

    Returns:
    - None
    """
    plt.figure(figsize=(10, 6), dpi=110)
    plt.plot(train_losses, label="Training Loss")
    plt.plot(valid_losses, label="Validation Loss")
    plt.title("Learning Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    hyperparams_str = (
        f"Epochs: {hyperparams[0]}, LR: {hyperparams[1]}, LR Decay Factor:"
        + f"{hyperparams[2]}, LR Decay Step: {hyperparams[3]}, Regularization: {hyperparams[4]}"
    )
    plt.text(
        0.5,
        0.95,
        hyperparams_str,
        horizontalalignment="center",
        verticalalignment="center",
        transform=plt.gca().transAxes,
    )
    plt.savefig(
        f"{base_path}/lc_{hyperparams[0]}_{hyperparams[1]}_{hyperparams[2]}_{hyperparams[3]}_{hyperparams[4]}.png"
    )
    plt.close()


def plot_accuracy_curve(train_accuracies, valid_accuracies, hyperparams, base_path=None):
    """
    Plot the accuracy curve showing training and validation accuracies across epochs.

    Parameters:
    - train_accuracies (list): Training accuracy values for each epoch.
    - valid_accuracies (list): Validation accuracy values for each epoch.
    - hyperparams (list): List containing hyperparameters used for the plot title.
    - base_path (str): Base path to save the plot as an image.

    Returns:
    - None
    """
    plt.figure(figsize=(10, 6), dpi=110)
    plt.plot(train_accuracies, label="Training Accuracy")
    plt.plot(valid_accuracies, label="Validation Accuracy")
    plt.title("Accuracy Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid()
    hyperparams_str = (
        f"Epochs: {hyperparams[0]}, LR: {hyperparams[1]}, LR Decay Factor:"
        + f"{hyperparams[2]}, LR Decay Step: {hyperparams[3]}, Regularization: {hyperparams[4]}"
    )
    plt.text(
        0.5,
        0.95,
        hyperparams_str,
        horizontalalignment="center",
        verticalalignment="center",
        transform=plt.gca().transAxes,
    )
    plt.savefig(
        f"{base_path}/acc_{hyperparams[0]}_{hyperparams[1]}_{hyperparams[2]}_{hyperparams[3]}_{hyperparams[4]}.png"
    )
    plt.close()
