"""
Module for training and testing a neural network model on insurability data.
"""

import torch
from data_loaders.insurability_data import CustomInsurabilityDataset
from data_loaders.read_data import read_insurability
from data_loaders.standard_scaler import StandardScaler
from networks.evaluate_network import evaluate_network
from networks.insurability_q4_v2 import CustomSGD, FeedForward
from networks.test_network import test_network
from networks.train_network import train_network
from torch import nn
from torch.utils.data import DataLoader
from utils.plot_evaluation_custom_optimizer import (
    plot_accuracy_curve,
    plot_learning_curve,
)

PRINT_INTERVAL = 250
BASE_PATH = "results/question_4_v2/"


def train_and_test_q4():
    """
    Train and test a FeedForward network on insurability data.
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
    test_data = CustomInsurabilityDataset("data/three_test.csv", scaler=ss)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=True)

    # Initialize FeedForward model, loss function, optimizer, and lists to track metrics
    device = "cpu"
    num_epochs = 3000
    learning_rate = 0.01
    ff_custom_optimizer = FeedForward().to(device)
    ff_optimizer = FeedForward().to(device)
    optimizer = torch.optim.SGD(
        ff_optimizer.parameters(),
        lr=learning_rate,
    )
    custom_optimizer = CustomSGD(ff_custom_optimizer.parameters(), lr=learning_rate)
    loss_func = nn.CrossEntropyLoss()
    train_losses_custom_optimizer = []
    train_accuracies_custom_optimizer = []
    train_losses_optimizer = []
    train_accuracies_optimizer = []
    val_losses = []
    val_accuracies = []

    for epoch in range(num_epochs):
        # Train the network and validate
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

        # Print Loss
        if (epoch + 1) % PRINT_INTERVAL == 0:
            print(f"---Epoch [{epoch + 1}/{num_epochs}]---")
            print(f"Train Loss (custom Optimizer): {train_loss_custom_optimizer:.6f}")
            print(f"Train Loss (Optimizer): {train_loss_optimizer:.6f}")
            print(f"Valid Loss: {val_loss:.6f}")
            print(
                f"Train Accuracy (custom Optimizer): {train_accuracy_custom_optimizer:.6f}"
            )
            print(f"Train Accuracy (Optimizer): {train_accuracy_optimizer:.6f}")
            print(f"Valid Accuracy: {val_accuracy:.6f}\n")

    # Plot learning and accuracy curves
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

    # Evaluate Network
    (
        test_loss,
        test_accuracy,
        precision,
        recall,
        f1,
        macro_precision,
        macro_recall,
        conf_matrix,
    ) = evaluate_network(test_loader, ff_custom_optimizer, loss_func, device)
    print("---Test Results---")
    print(f"Loss: {test_loss:.6f}")
    print(f"Accuracy: {test_accuracy:.6f}")
    print(f"Precision: {precision:.6f}")
    print(f"Recall: {recall:.6f}")
    print(f"F1-Score: {f1:.6f}")
    print(f"Macro-Precision: {macro_precision:.6f}")
    print(f"Macro-Recall: {macro_recall:.6f}")
    print("Confusion Matrix:")
    print(conf_matrix)


if __name__ == "__main__":
    train_and_test_q4()
