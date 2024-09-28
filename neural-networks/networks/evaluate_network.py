import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)


def evaluate_network(dataloader, model, loss_func, device="cpu"):
    num_batches = 0
    model.eval()
    test_loss = 0
    predicted_labels_list = []
    true_labels_list = []

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            # Calculate loss
            pred = model(X)
            test_loss += loss_func(pred, y).item()
            num_batches = num_batches + 1

            # Collect predicted and true labels
            _, predicted_labels = torch.max(pred, 1)
            predicted_labels_list.extend(predicted_labels.cpu().numpy())
            true_labels_list.extend(y.cpu().numpy())

    test_loss /= num_batches
    test_accuracy = accuracy_score(true_labels_list, predicted_labels_list)

    # Calculate precision, recall, F1 score
    precision = precision_score(
        true_labels_list, predicted_labels_list, average="weighted"
    )
    recall = recall_score(true_labels_list, predicted_labels_list, average="weighted")
    f1 = f1_score(true_labels_list, predicted_labels_list, average="weighted")

    # Calculate macro-average precision and recall
    macro_precision, macro_recall, _, _ = precision_recall_fscore_support(
        true_labels_list, predicted_labels_list, average="macro"
    )

    # confusion matrix
    conf_matrix = confusion_matrix(true_labels_list, predicted_labels_list)

    return (
        test_loss,
        test_accuracy,
        precision,
        recall,
        f1,
        macro_precision,
        macro_recall,
        conf_matrix,
    )
