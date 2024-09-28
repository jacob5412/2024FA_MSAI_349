"""
Evaluation Metrics Utilities
"""

import numpy as np


def create_confusion_matrix(num_classes, actual_labels, predicted_labels):
    """
    Create a confusion matrix for a n-class classification problem.

    Args:
        num_classes (int): The total number of classes.
        actual_labels (np.ndarray): An array of actual class labels.
        predicted_labels (np.ndarray): An array of predicted class labels.

    Returns:
        confusion_matrix (np.ndarray): each element (i, j) represents the
        number of data points that belong to class i (actual) and were
        predicted to be in class j (predicted).
    """
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
    for actual_label, predicted_label in zip(actual_labels, predicted_labels):
        confusion_matrix[actual_label - 1, predicted_label - 1] += 1

    return confusion_matrix


def display_confusion_matrix(confusion_matrix):
    """
    Display a confusion matrix with row and column indices.

    Args:
        confusion_matrix (np.ndarray): each element (i, j) represents the
        number of data points that belong to class i (actual) and were
        predicted to be in class j (predicted).
    """
    num_classes = confusion_matrix.shape[0]
    max_count_width = len(str(confusion_matrix.max()))

    print("i-th --> Actual; j-th --> Predictions\n")
    tab_header = "     " + " ".join(
        f"{i:>{max_count_width}}  " for i in range(0, num_classes)
    )  # prediction labels
    print(tab_header)
    print("   " + ("-" * (max_count_width + 3) * num_classes))
    for i in range(num_classes):
        row = f"{i:>{max_count_width}} |"  # actual labels
        for j in range(num_classes):  # prediction vs actual
            row += f" {confusion_matrix[i, j]:>{max_count_width}}  "
        print(row)
    print("\n")


def eval_metrics_from_confusion_matrix(confusion_matrix):
    """
    Calculate class-specific precision and recall, their macro
    averages and accuracy using a confusion matrix through one-vs-all.

    Args:
        confusion_matrix (np.ndarray): each element (i, j) represents the
        number of data points that belong to class i (actual) and were
        predicted to be in class j (predicted).

    Returns:
        eval_metrics (dict): A dictionary where keys are class labels and
        values are dictionaries containing 'accuracy', 'precision', and
        'recall'.
    """
    num_classes = confusion_matrix.shape[0]
    eval_metrics = {}
    class_precisions = []
    class_recalls = []
    diagonal_confusion_mat = np.diagonal(confusion_matrix)

    for i in range(num_classes):
        eval_metrics[i] = {}
        true_positive = confusion_matrix[i, i]
        # false positives = column sum - true positives
        false_positive = np.sum(confusion_matrix[:, i]) - true_positive
        # false negatives =  row sum - true positives
        false_negative = np.sum(confusion_matrix[i, :]) - true_positive
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)
        class_precisions.append(precision)
        class_recalls.append(recall)
        eval_metrics[i]["precision"] = precision
        eval_metrics[i]["recall"] = recall

    accuracy = np.sum(diagonal_confusion_mat) / np.sum(confusion_matrix)
    macro_avg_precision = sum(class_precisions) / num_classes
    macro_avg_recall = sum(class_recalls) / num_classes

    eval_metrics["overall"] = {}
    eval_metrics["overall"]["accuracy"] = accuracy
    eval_metrics["overall"]["macro_avg_precision"] = macro_avg_precision
    eval_metrics["overall"]["macro_avg_recall"] = macro_avg_recall
    return eval_metrics


def display_eval_metrics(eval_metrics, indent=0):
    """
    Display evaluation metrics by recursively printing a dictionary.

    Args:
        eval_metrics (dict): A dictionary containing class-specific and
                             overall evaluation metrics.
        indent (int): Indentation level for displaying nested dictionaries.
    """
    for key, val in eval_metrics.items():
        print("     " * indent + str(key) + ":")
        if isinstance(val, dict):
            display_eval_metrics(val, indent + 1)
        else:
            print("     " * (indent + 1) + f"{val:.4f}")
