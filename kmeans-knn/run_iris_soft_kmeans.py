"""
Testing Soft K-means on the Iris Dataset
"""

import logging

import numpy as np
from soft_kmeans import SoftKMeans
from utilities.evaluation_utils import (
    create_confusion_matrix,
    display_confusion_matrix,
    display_eval_metrics,
    eval_metrics_from_confusion_matrix,
)
from utilities.read_data import get_numerical_features, get_numerical_labels, read_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("soft-soft_kmeans-training")


if __name__ == "__main__":
    training_set = read_data("iris_dataset/train.csv")
    validation_set = read_data("iris_dataset/valid.csv")
    validation_set_labels = np.array(get_numerical_labels(validation_set, int))
    validation_set_features = np.array(get_numerical_features(validation_set, float))
    training_set_labels = np.array(get_numerical_labels(training_set, int))
    training_set_features = np.array(get_numerical_features(training_set, float))

    n_clusters = 3
    soft_kmeans = SoftKMeans(n_clusters)
    print("--- Training Soft K-means on Iris ---\n")
    soft_kmeans.fit(training_set_features)
    predicted_labels = soft_kmeans.predict(training_set_features, training_set_labels)
    confusion_mat = create_confusion_matrix(
        n_clusters, training_set_labels, predicted_labels
    )
    display_confusion_matrix(confusion_mat)
    eval_metrics = eval_metrics_from_confusion_matrix(confusion_mat)
    display_eval_metrics(eval_metrics)

    print("--- Testing Soft K-means on Iris ---\n")
    predicted_labels = soft_kmeans.predict(validation_set_features, validation_set_labels)
    confusion_mat = create_confusion_matrix(
        n_clusters, validation_set_labels, predicted_labels
    )
    display_confusion_matrix(confusion_mat)
    eval_metrics = eval_metrics_from_confusion_matrix(confusion_mat)
    display_eval_metrics(eval_metrics)
