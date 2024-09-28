"""
Module for hyperparameter search and evaluation.
"""

import csv

import torch
from data_loaders.mnist_data import CustomMnistDataset
from data_loaders.read_data import read_mnist
from data_loaders.standard_scaler import StandardScaler
from networks.mnist_q2 import FeedForward
from networks.test_network import test_network
from networks.train_network import train_network
from torch import nn
from torch.utils.data import DataLoader
from utils.generate_hyperparams import get_hyperparams_q2
from utils.plot_evaluation import plot_accuracy_curve, plot_learning_curve

MINIMUM_LEARNING_RATE = 1e-6
PRINT_INTERVAL = 150
BASE_PATH = "hyperparams/question_2/"


def save_to_csv(data):
    """
    Save the results data to a CSV file.
    """
    headers = [
        "num_epochs",
        "learning_rate",
        "lr_decay_factor",
        "lr_decay_step",
        "final_train_loss",
        "final_valid_loss",
        "final_train_acc",
        "final_valid_acc",
    ]

    with open(BASE_PATH + "results.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(data)


def hyperparams_search_q2():
    """
    Perform hyperparameter search for a neural network model on mnist data.
    """
    # Load training data
    train = read_mnist("data/mnist_train.csv")
    train_features = train[:, 1:]

    # Scaling data
    ss = StandardScaler()
    ss.fit(train_features)

    # Dataset loaders
    train_data = CustomMnistDataset("data/mnist_train.csv", scaler=ss)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    valid_data = CustomMnistDataset("data/mnist_valid.csv", scaler=ss)
    valid_loader = DataLoader(valid_data, batch_size=64, shuffle=True)

    hyperparams_list = get_hyperparams_q2()
    device = "cpu"
    results = []

    for hyperparams in hyperparams_list:
        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []
        num_epochs, learning_rate, lr_decay_factor, lr_decay_step = hyperparams
        original_learning_rate = learning_rate

        ff = FeedForward().to(device)
        loss_func = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(ff.parameters(), lr=learning_rate)

        for epoch in range(num_epochs):
            # Fetch train and valid losses
            train_loss, train_accuracy = train_network(
                train_loader, ff, loss_func, optimizer, device
            )
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)

            val_loss, val_accuracy = test_network(valid_loader, ff, loss_func, device)
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)

            # Print loss
            if (epoch + 1) % PRINT_INTERVAL == 0:
                print(f"---Epoch [{epoch + 1}/{num_epochs}]---")
                print(f"Train Loss: {train_loss:.6f}")
                print(f"Valid Loss: {val_loss:.6f}")
                print(f"Train Accuracy: {train_accuracy:.6f}")
                print(f"Valid Accuracy: {val_accuracy:.6f}\n")

            # Learning rate decay schedule
            if (epoch + 1) % lr_decay_step == 0:
                learning_rate = max(
                    optimizer.param_groups[0]["lr"] * lr_decay_factor,
                    MINIMUM_LEARNING_RATE,
                )
                optimizer.param_groups[0]["lr"] = learning_rate
                print(f"New LR is: {optimizer.param_groups[0]['lr']:.8f}")
        results.append(
            list(hyperparams) + [train_loss, val_loss, train_accuracy, val_accuracy]
        )
        plot_learning_curve(
            train_losses,
            val_losses,
            [num_epochs, original_learning_rate, lr_decay_factor, lr_decay_step],
            BASE_PATH,
        )
        plot_accuracy_curve(
            train_accuracies,
            val_accuracies,
            [num_epochs, original_learning_rate, lr_decay_factor, lr_decay_step],
            BASE_PATH,
        )
    save_to_csv(results)


if __name__ == "__main__":
    hyperparams_search_q2()
