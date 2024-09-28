import torch


def train_network(dataloader, model, loss_func, optimizer, device="cpu"):
    model.train()

    num_batches = 0
    train_loss = 0
    correct_predictions = 0
    total_samples = 0

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)

        # Make some predictions and get the error
        pred = model(X)
        loss = loss_func(pred, y)

        # Calculate accuracy
        _, predicted_labels = torch.max(pred, 1)
        correct_predictions += (predicted_labels == y).sum().item()
        total_samples += y.size(0)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        num_batches = num_batches + 1

    train_loss /= num_batches
    train_accuracy = correct_predictions / total_samples
    return train_loss, train_accuracy
