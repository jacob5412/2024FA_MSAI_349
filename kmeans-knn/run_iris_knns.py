"""
Testing KNN on the Iris Dataset
"""

import numpy as np
from k_nearest_neighbor import KNearestNeighbor
from utilities.evaluation_utils import (
    create_confusion_matrix,
    display_confusion_matrix,
    display_eval_metrics,
    eval_metrics_from_confusion_matrix,
)
from utilities.read_data import get_numerical_features, get_numerical_labels, read_data

if __name__ == "__main__":
    NUM_NEIGHBORS = 3
    NUM_CLASSES = 3

    train_set = read_data("iris_dataset/train.csv")
    train_set_labels = np.array(get_numerical_labels(train_set, data_type=int))
    train_set_features = np.array(get_numerical_features(train_set, data_type=float))

    validation_set = read_data("iris_dataset/valid.csv")
    validation_set_labels = np.array(get_numerical_labels(validation_set, data_type=int))
    validation_set_features = np.array(
        get_numerical_features(validation_set, data_type=float)
    )

    print("--- Testing KNNs on Iris ---\n")
    knearest = KNearestNeighbor(NUM_NEIGHBORS, "mode")
    knearest.fit(train_set_features, train_set_labels)
    predicted_labels = knearest.predict(validation_set_features)

    confusion_matrix = create_confusion_matrix(
        NUM_CLASSES, validation_set_labels, predicted_labels
    )
    display_confusion_matrix(confusion_matrix)
    eval_metrics = eval_metrics_from_confusion_matrix(confusion_matrix)
    display_eval_metrics(eval_metrics)
