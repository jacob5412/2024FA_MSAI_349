import torch


def test_network(dataloader, model, loss_func, device="cpu"):
    num_batches = 0
    model.eval()
    test_loss = 0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            # Calculate loss
            pred = model(X)
            test_loss += loss_func(pred, y).item()
            num_batches = num_batches + 1

            # Calculate accuracy
            _, predicted_labels = torch.max(pred, 1)
            correct_predictions += (predicted_labels == y).sum().item()
            total_samples += y.size(0)

    test_loss /= num_batches
    test_accuracy = correct_predictions / total_samples
    return test_loss, test_accuracy
