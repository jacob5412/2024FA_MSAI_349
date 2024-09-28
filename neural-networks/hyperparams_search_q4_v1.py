"""
Module for hyperparameter search and evaluation.
"""

import csv

import torch
from data_loaders.insurability_data import CustomInsurabilityDataset
from data_loaders.read_data import read_insurability
from data_loaders.standard_scaler import StandardScaler
from networks.insurability_q4_v1 import CustomSGD, FeedForward
from networks.test_network import test_network
from networks.train_network import train_network
from torch import nn
from torch.utils.data import DataLoader
from utils.generate_hyperparams import get_hyperparams_q4
from utils.plot_evaluation_custom_optimizer import (
    plot_accuracy_curve,
    plot_learning_curve,
)

PRINT_INTERVAL = 250
BASE_PATH = "hyperparams/question_4_v1/"


def save_to_csv(data):
    """
    Save the results data to a CSV file.
    """
    headers = [
        "num_epochs",
        "learning_rate",
        "final_train_loss",
        "final_valid_loss",
        "final_train_acc",
        "final_valid_acc",
    ]

    with open(BASE_PATH + "results.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(data)


def hyperparams_search_q4():
    """
    Perform hyperparameter search for a neural network model on insurability data.
    """
    # Load training data
    train = read_insurability("data/three_train.csv")
    train_features = train[:, 1:]

    # Scaling data
    ss = StandardScaler()
    ss.fit(train_features)

    # Dataset loaders
    train_data = CustomInsurabilityDataset("data/three_train.csv", scaler=ss)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    valid_data = CustomInsurabilityDataset("data/three_valid.csv", scaler=ss)
    valid_loader = DataLoader(valid_data, batch_size=64, shuffle=True)

    hyperparams_list = get_hyperparams_q4()
    device = "cpu"
    results = []

    for hyperparams in hyperparams_list:
        train_losses_custom_optimizer = []
        train_accuracies_custom_optimizer = []
        train_losses_optimizer = []
        train_accuracies_optimizer = []
        val_losses = []
        val_accuracies = []
        num_epochs, learning_rate = hyperparams

        ff_custom_optimizer = FeedForward().to(device)
        ff_optimizer = FeedForward().to(device)

        optimizer = torch.optim.SGD(
            ff_optimizer.parameters(),
            lr=learning_rate,
        )
        custom_optimizer = CustomSGD(ff_custom_optimizer.parameters(), lr=learning_rate)
        loss_func = nn.CrossEntropyLoss()

        for epoch in range(num_epochs):
            # Print gradients before
            if (epoch + 1) % PRINT_INTERVAL == 0:
                print("---Before---")
                print("Custom Optimizer Gradients:")
                for name, param in ff_custom_optimizer.named_parameters():
                    if param.grad is not None and name != "bias":
                        print(f"Parameter: {name}, Gradient: {param.grad}")

                print("Optimizer Gradients:")
                for name, param in ff_optimizer.named_parameters():
                    if param.grad is not None and name != "bias":
                        print(f"Parameter: {name}, Gradient: {param.grad}\n")

            # Fetch train and valid losses
            (
                train_loss_custom_optimizer,
                train_accuracy_custom_optimizer,
            ) = train_network(
                train_loader, ff_custom_optimizer, loss_func, custom_optimizer, device
            )
            train_losses_custom_optimizer.append(train_loss_custom_optimizer)
            train_accuracies_custom_optimizer.append(train_accuracy_custom_optimizer)

            train_loss_optimizer, train_accuracy_optimizer = train_network(
                train_loader, ff_optimizer, loss_func, optimizer, device
            )
            train_losses_optimizer.append(train_loss_optimizer)
            train_accuracies_optimizer.append(train_accuracy_optimizer)

            val_loss, val_accuracy = test_network(
                valid_loader, ff_custom_optimizer, loss_func, device
            )
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)

            # Print loss and gradients after
            if (epoch + 1) % PRINT_INTERVAL == 0:
                print("---After---")
                print("Custom Optimizer Gradients:")
                for name, param in ff_custom_optimizer.named_parameters():
                    if param.grad is not None and name != "bias":
                        print(f"Parameter: {name}, Gradient: {param.grad}")

                print("Optimizer Gradients:")
                for name, param in ff_optimizer.named_parameters():
                    if param.grad is not None and name != "bias":
                        print(f"Parameter: {name}, Gradient: {param.grad}\n")

                print(f"---Epoch [{epoch + 1}/{num_epochs}]---")
                print(f"Train Loss (custom Optimizer): {train_loss_custom_optimizer:.6f}")
                print(f"Train Loss (Optimizer): {train_loss_optimizer:.6f}")
                print(f"Valid Loss: {val_loss:.6f}")
                print(
                    f"Train Accuracy (custom Optimizer): {train_accuracy_custom_optimizer:.6f}"
                )
                print(f"Train Accuracy (Optimizer): {train_accuracy_optimizer:.6f}")
                print(f"Valid Accuracy: {val_accuracy:.6f}\n")

        results.append(
            list(hyperparams)
            + [
                train_loss_custom_optimizer,
                val_loss,
                train_accuracy_custom_optimizer,
                val_accuracy,
            ]
        )
        plot_learning_curve(
            train_losses_custom_optimizer,
            train_losses_optimizer,
            val_losses,
            [num_epochs, learning_rate],
            BASE_PATH,
        )
        plot_accuracy_curve(
            train_accuracies_custom_optimizer,
            train_accuracies_optimizer,
            val_accuracies,
            [num_epochs, learning_rate],
            BASE_PATH,
        )
    save_to_csv(results)


if __name__ == "__main__":
    hyperparams_search_q4()
