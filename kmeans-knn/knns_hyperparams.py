"""
Selecting the best hyperparameters for KNNs based on
the average accuracy over multiple iterations.
"""

import csv
import logging

import numpy as np
from k_nearest_neighbor import KNearestNeighbor
from utilities.evaluation_utils import (
    create_confusion_matrix,
    eval_metrics_from_confusion_matrix,
)
from utilities.pca_utils import PCA
from utilities.preprocessing_utils import GrayscaleScaler, StandardScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("knns-hyperparameter-selection")


def read_hyperparams(filepath):
    with open(filepath, "r", newline="") as csvfile:
        reader = csv.DictReader(csvfile)

        # Iterate through the rows and yield hyperparameter values as tuples
        for row in reader:
            k_neighbors = int(row["k_neighbors"])
            pca_components = (
                int(row["pca_components"]) if row["pca_components"] != "None" else None
            )
            scaler = None if row["scaler"] == "None" else row["scaler"]
            distance_metric = row["distance_metrics"]

            yield (k_neighbors, pca_components, scaler, distance_metric)


def get_best_hyperparams(
    training_set_features: np.ndarray,
    training_set_labels: np.ndarray,
    validation_set_features: np.ndarray,
    validation_set_labels: np.ndarray,
    num_classes: int,
    ignore_first: bool,
    hyperparam_filepath: str,
    hyperparams_results_filepath: str,
) -> tuple:
    avg_accuracies = {}
    csv_data = []
    num_iterations = 25
    training_indices = np.arange(training_set_features.shape[0])
    validation_indices = np.arange(validation_set_features.shape[0])

    for hyperparams in read_hyperparams(hyperparam_filepath):
        k_neighbors = hyperparams[0]
        pca_component = hyperparams[1]
        scaler = hyperparams[2]
        distance_metric = hyperparams[3]
        total_accuracy = 0
        for _ in range(num_iterations):
            # Shuffle Data
            np.random.shuffle(training_indices)
            np.random.shuffle(validation_indices)
            shuffled_training_set_features = training_set_features[training_indices]
            shuffled_training_set_labels = training_set_labels[training_indices]
            shuffled_validation_set_features = validation_set_features[validation_indices]
            shuffled_validation_set_labels = validation_set_labels[validation_indices]

            # Scale Data
            if scaler == "GrayscaleScaler":
                scaler_obj = GrayscaleScaler()
                scaled_training_set_features = scaler_obj.fit_transform(
                    shuffled_training_set_features
                )
                scaled_validation_set_features = scaler_obj.fit_transform(
                    shuffled_validation_set_features
                )
            elif scaler == "StandardScaler":
                scaler_obj = StandardScaler()
                scaler_obj.fit(shuffled_training_set_features)
                scaled_training_set_features = scaler_obj.transform(
                    shuffled_training_set_features
                )
                scaled_validation_set_features = scaler_obj.transform(
                    shuffled_validation_set_features
                )
            else:
                scaled_training_set_features = shuffled_training_set_features
                scaled_validation_set_features = shuffled_validation_set_features

            # Fit PCA
            if pca_component is None:
                transformed_train_features = shuffled_training_set_features
                transformed_valid_features = shuffled_validation_set_features
            else:
                pca = PCA(pca_component)
                pca.fit(scaled_training_set_features)
                transformed_train_features = pca.transform(scaled_training_set_features)
                transformed_valid_features = pca.transform(scaled_validation_set_features)

            # Train KNNs
            knns = KNearestNeighbor(k_neighbors)
            knns.fit(transformed_train_features, shuffled_training_set_labels)
            predicted_labels = knns.predict(
                transformed_valid_features,
                ignore_first,
                distance_metric,
            )

            # Evaluate
            confusion_mat = create_confusion_matrix(
                num_classes, shuffled_validation_set_labels, predicted_labels
            )
            eval_metrics = eval_metrics_from_confusion_matrix(confusion_mat)
            total_accuracy += eval_metrics["overall"]["accuracy"]
        avg_accuracy = total_accuracy / num_iterations
        avg_accuracies[hyperparams] = total_accuracy / num_iterations

        logger.info(
            "Accuracy for hyperparams %s is %.3f",
            hyperparams,
            avg_accuracy,
        )
        csv_data.append(
            [k_neighbors, pca_component, scaler, distance_metric, avg_accuracy]
        )

    # Output the data to a CSV file
    with open(hyperparams_results_filepath, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "k_neighbors",
                "pca_component",
                "scaler",
                "distance_metric",
                "avg_accuracy",
            ]
        )
        writer.writerows(csv_data)

    return max(avg_accuracies, key=lambda key: avg_accuracies[key])
