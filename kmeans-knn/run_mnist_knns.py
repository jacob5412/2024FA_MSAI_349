"""
Run KNNs algorithm
"""

import argparse
import logging

import numpy as np
from k_nearest_neighbor import KNearestNeighbor
from knns_hyperparams import get_best_hyperparams
from utilities.evaluation_utils import (
    create_confusion_matrix,
    display_confusion_matrix,
    display_eval_metrics,
    eval_metrics_from_confusion_matrix,
)
from utilities.pca_utils import PCA
from utilities.preprocessing_utils import GrayscaleScaler, StandardScaler
from utilities.read_data import get_numerical_features, get_numerical_labels, read_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("knns-hyperparameter-training")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MNIST on KNNs.")
    parser.add_argument(
        "--no-tuning",
        dest="tuning",
        action="store_false",
        help="Set to False to disable hyperparameter tuning",
    )
    args = parser.parse_args()

    hyperparams_filepath = "hyperparams_files/knns_hyperparams.csv"
    hyperparams_results_filepath = "hyperparams_files/knns_hyperparams_results.csv"
    training_set = read_data("mnist_dataset/train.csv")
    training_set_labels = np.array(get_numerical_labels(training_set))
    training_set_features = np.array(get_numerical_features(training_set))
    validation_set = read_data("mnist_dataset/valid.csv")
    validation_set_labels = np.array(get_numerical_labels(validation_set))
    validation_set_features = np.array(get_numerical_features(validation_set))
    testing_set = read_data("mnist_dataset/test.csv")
    testing_set_labels = np.array(get_numerical_labels(testing_set))
    testing_set_features = np.array(get_numerical_features(testing_set))

    NUM_CLASSES = 10
    IGNORE_FIRST = False
    if not args.tuning:
        # Updating with empirical data
        best_k = 4
        best_pca_num_components = 550
        best_scaler = "GrayScaler"
        best_distance_metric = "cosine"
    else:
        # Hyperparameter-Tuning
        best_hyperparams = get_best_hyperparams(
            training_set_features,
            training_set_labels,
            validation_set_features,
            validation_set_labels,
            NUM_CLASSES,
            IGNORE_FIRST,
            hyperparams_filepath,
            hyperparams_results_filepath,
        )

        best_k = best_hyperparams[0]
        best_pca_num_components = best_hyperparams[1]
        best_scaler = best_hyperparams[2]
        best_distance_metric = best_hyperparams[3]

        logger.info(
            "Best hyperparams are %s",
            best_hyperparams,
        )

    # Training & testing final K-means
    if best_scaler == "GrayscaleScaler":
        grayscale_scaler = GrayscaleScaler()
        scaled_training_set_features = grayscale_scaler.fit_transform(
            training_set_features
        )
        scaled_testing_set_features = grayscale_scaler.fit_transform(testing_set_features)
    elif best_scaler == "StandardScaler":
        standard_scaler = StandardScaler()
        standard_scaler.fit(training_set_features, training_set_labels)
        scaled_training_set_features = standard_scaler.transform(training_set_features)
        scaled_testing_set_features = standard_scaler.transform(testing_set_features)
    else:
        scaled_training_set_features = training_set_features
        scaled_testing_set_features = testing_set_features

    if best_pca_num_components is None:
        transformed_train_features = scaled_training_set_features
        transformed_test_features = scaled_testing_set_features
    else:
        pca = PCA(best_pca_num_components)
        pca.fit(scaled_training_set_features)
        transformed_train_features = pca.transform(scaled_training_set_features)
        transformed_test_features = pca.transform(scaled_testing_set_features)

    knns = KNearestNeighbor(best_k)
    knns.fit(transformed_train_features, training_set_labels)
    predicted_labels = knns.predict(
        transformed_test_features, IGNORE_FIRST, best_distance_metric
    )
    confusion_mat = create_confusion_matrix(
        NUM_CLASSES, testing_set_labels, predicted_labels
    )
    display_confusion_matrix(confusion_mat)
    eval_metrics = eval_metrics_from_confusion_matrix(confusion_mat)
    display_eval_metrics(eval_metrics)
