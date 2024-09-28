"""
K-means
"""

import numpy as np
from utilities import distance_utils


class KMeans:
    """
    This class implements the traditional KMeans algorithm with hard
    assignments.
    """

    def __init__(self, n_clusters: int) -> None:
        """
        The KMeans algorithm has two steps:

        1. Update assignments
        2. Update the means

        Args:
            n_clusters (int): Number of clusters to cluster the given data
                              into.
        """
        self.n_clusters = n_clusters
        self.centroids = None

    def fit(
        self,
        features: np.ndarray,
        metric="euclidean",
        rtol_threshold=1e-5,
        atol_threshold=1e-7,
        n_iterations=10000,
    ) -> None:
        """
        Fit KMeans to the given data using `self.n_clusters` number of
        clusters.

        Args:
            features (np.ndarray): array containing inputs of size
            (n_samples, n_features).
            metric (str): distance metric to use (e.g., "euclidean"
            or "cosine").
            rtol_threshold (float): Relative tolerance for convergence.
            atol_threshold (float): Absolute tolerance for convergence.
            n_iterations (int): The maximum number of iterations.
        """

        self.centroids = self._initialize_centroids(features)
        previous_centroids = np.zeros_like(self.centroids)

        while not self._has_converged(
            previous_centroids, rtol_threshold, atol_threshold, n_iterations
        ):
            n_iterations -= 1
            previous_centroids = self.centroids
            cluster_assignments = self._update_cluster_assignments(features, metric)
            self._update_centroids(features, cluster_assignments)

    def predict(
        self, features: np.ndarray, labels: np.ndarray[int], metric="euclidean"
    ) -> np.ndarray[int]:
        """
        Given features, an np.ndarray of size (n_samples, n_features), predict
        labels based on weighted voting.

        Args:
            features (np.ndarray): array containing inputs of size
            (n_samples, n_features).
            labels (np.ndarray): array containing integer input labels.
            metric (str): distance metric to use (e.g., "euclidean"
            or "cosine").

        Returns:
            predictions (np.ndarray): predicted cluster membership for each
            feature, of size (n_samples,). Each element of the array is the
            index of the cluster the sample belongs to.
        """
        distance_metric = getattr(distance_utils, metric + "_distance")
        distances = distance_metric(features, self.centroids)
        cluster_assignments = np.argmin(distances, axis=1)
        predicted_labels = np.zeros_like(cluster_assignments, dtype=int)
        cluster_labels = np.unique(cluster_assignments)

        # for each cluster, find all the data points belonging to it
        for cluster_id in cluster_labels:
            cluster_mask = cluster_assignments == cluster_id
            if np.sum(cluster_mask) > 0:
                true_labels = labels[cluster_mask]
                # weight them based on their distance to the centroid
                weights = 10.0 / (1.0 + distances[cluster_mask, cluster_id])
                # weight votes based on these distances
                weighted_votes = np.bincount(true_labels, weights=weights)
                # find the most common label
                majority_label = np.argmax(weighted_votes)
                predicted_labels[cluster_mask] = majority_label
        return predicted_labels

    def _initialize_centroids(self, features: np.ndarray) -> np.ndarray:
        """
        Randomly select n_clusters data points from features as initial means.

        Args:
            features (np.ndarray): array containing inputs of size
            (n_samples, n_features).

        Returns:
            array containing data points of size (n_clusters, n_features).
        """
        random_indices = np.random.choice(
            features.shape[0], self.n_clusters, replace=False
        )
        return features[random_indices]

    def _has_converged(
        self,
        previous_centroids: np.ndarray,
        rtol_threshold: float,
        atol_threshold: float,
        n_iterations: int,
    ) -> bool:
        """
        K-means converges after n_iterations or when current centroid and
        previous centroid are within a certain threshold.

        Args:
            previous_centroids (np.ndarray): array containing centroids
            from previous time-step, of size (n_clusters, n_features).
            rtol_threshold (float): Relative tolerance for convergence.
            atol_threshold (float): Absolute tolerance for convergence.
            n_iterations (int): The maximum number of iterations.

        Returns:
            True if k-means has converged, False otherwise.
        """
        no_centroids_change = np.allclose(
            self.centroids, previous_centroids, rtol=rtol_threshold, atol=atol_threshold
        )
        max_iterations = n_iterations <= 0
        return no_centroids_change or max_iterations

    def _update_cluster_assignments(
        self, features: np.ndarray, metric: str
    ) -> np.ndarray:
        """
        Update cluster assignments based on a distance metric.

        Args:
            features (np.ndarray): array containing inputs of size
            (n_samples, n_features).
            metric (str): The distance metric to use for the calculation.

        Returns:
            cluster_assignments (np.ndarray): array containing cluster
            assignments of size (n_features, ).
        """
        distance_metric = getattr(distance_utils, metric + "_distance")
        # distances (n_samples, n_clusters)
        distances = distance_metric(features, self.centroids)
        # cluster assignments is a list of indices of least distance
        cluster_assignments = np.argmin(distances, axis=1)
        return cluster_assignments

    def _update_centroids(
        self, features: np.ndarray, cluster_assignments: np.ndarray
    ) -> None:
        """
        Update centroids based on the new cluster assignments.

        Args:
            features (np.ndarray): array containing inputs of size
            (n_samples, n_features).
            cluster_assignments (np.ndarray): array containing cluster
            assignments of size (n_features, ).
        """
        # use cluster assignments as a mask to obtain features belonging
        # to the same cluster
        self.centroids = np.array(
            [
                np.mean(features[cluster_assignments == i], axis=0)
                for i in range(self.n_clusters)
            ]
        )
