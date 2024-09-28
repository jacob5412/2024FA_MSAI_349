"""
K-Nearest Neighbor Classifer
"""

from collections import Counter

import numpy as np
from utilities import distance_utils


class KNearestNeighbor:
    """
    A K-Nearest Neighbor Classifier.

    Attributes:
        n_neighbors (int): The number of nearest neighbors to consider during
        prediction.
        aggregator (str): The method for aggregating neighbor labels.
    """

    def __init__(self, n_neighbors: int, aggregator="mode") -> None:
        self.n_neighbors = n_neighbors
        self.aggregator = aggregator

    def _get_distances(
        self, train_feature: np.ndarray, test_feature: np.ndarray, metric: str
    ) -> np.ndarray:
        """
        Calculate the distances between training and test features.

        Args:
            train_feature (np.ndarray): The feature from the training set.
            test_feature (np.ndarray): The feature from the test set.
            metric (str): The distance metric to use for the calculation.

        Returns:
            np.ndarray: An array of distances between the two features.
        """
        distance_metric = getattr(distance_utils, metric + "_distance")
        distances = distance_metric(train_feature, test_feature)
        return distances

    def _label_voting(self, neighbors: np.ndarray, neighbor_distances: np.ndarray) -> int:
        """
        Perform label voting among the nearest neighbors.

        Args:
            neighbors (np.ndarray): An array of labels from the
            nearest neighbors.
            neighbor_distances (np.ndarray): An array of the distances
            between features and query point of size
            (n_neighbors, n_feature_samples)

        Returns:
            The label with the highest vote.
        """
        if self.aggregator == "mode":
            label_counts = Counter(neighbors)
            most_common_labels = label_counts.most_common()
            return most_common_labels[0][0]
        elif self.aggregator == "weighted_mode":
            weights = 10.0 / neighbor_distances
            weighted_votes = np.bincount(neighbors, weights=weights)
            majority_label = np.argmax(weighted_votes)
            return majority_label

    def fit(
        self,
        features: np.ndarray,
        labels: np.ndarray[int],
    ) -> None:
        """
        Fit the K-Nearest Neighbor classifier with training data.

        Args:
            features (np.ndarray): The features from the training set.
            labels (np.ndarray): The corresponding labels for the
            training data as integers.
        """
        self.features = features
        self.labels = labels

    def predict(
        self, query: np.ndarray, ignore_first=False, metric="euclidean"
    ) -> np.ndarray[int]:
        """
        Predict labels for a query or a batch of queries.

        Args:
            query (np.ndarray): The query for which labels are to be
            predicted of size (n_samples, n_features).
            ignore_first (bool): Whether to ignore the first neighbor label
            (False by default). Used when testing on training set itself.
            metric (str): The distance metric to use for the prediction.

        Returns:
            The predicted labels for the query
        """
        distances = self._get_distances(self.features, query, metric)
        predicted_labels = np.zeros(query.shape[0], dtype=int)
        # find indices of shortest distances of size
        # (n_feature_samples, n_query_samples)
        sorted_indices = np.argsort(distances, axis=0)
        # the indices of the nearest neighbors for each query point
        # of size (n_feature_samples, n_query_samples)
        sorted_labels = np.take(np.array(self.labels), sorted_indices, axis=0)
        # take the n neighbors, neighbors will be of size
        # (n_neighbors + 1, n_feature_samples)
        neighbors = sorted_labels[: self.n_neighbors + 1,]
        neighbor_distances = np.sort(distances, axis=0)[: self.n_neighbors + 1,]
        for i in range(neighbors.shape[1]):
            predicted_labels[i] = self._label_voting(
                neighbors[1:, i] if ignore_first else neighbors[: self.n_neighbors, i],
                (
                    neighbor_distances[1:, i]
                    if ignore_first
                    else neighbor_distances[: self.n_neighbors, i]
                ),
            )
        return predicted_labels
