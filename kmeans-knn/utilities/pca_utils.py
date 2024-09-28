"""
Principle Component Analysis
"""

import numpy as np


class PCA:
    """
    Performs PCA to reduce the dimensionality of the input features.

    https://www.askpython.com/python/examples/principal-component-analysis

    Attributes:
        num_components (int): The number of principal components to retain.
        components (np.ndarray): The retained principal components.
        mean (np.ndarray): The mean of the input data used for centering.
    """

    def __init__(self, num_components) -> None:
        self.num_components = num_components
        self.components = None
        self.mean = None

    def fit(self, features):
        """
        Fit the PCA model to the input data and reduce its dimensionality.

        Args:
            features (np.ndarray): Input data of shape (n_samples, n_features).
        """
        # Center the data by subtracting the mean from each feature.
        self.mean = np.mean(features, axis=0)
        features_centered = features - self.mean

        # Calculate the covariance matrix of the mean-centered data.
        cov_mat = np.cov(features_centered, rowvar=False)

        # Calculate Eigenvalues and Eigenvectors of the covariance matrix
        eigen_values, eigen_vectors = np.linalg.eigh(cov_mat)

        # Sort the Eigenvectors in descending order
        sorted_indices = np.argsort(eigen_values)[::-1]
        eigen_vectors_sorted = eigen_vectors[:, sorted_indices]

        # Select the first n Eigenvectors
        self.components = eigen_vectors_sorted[:, : self.num_components]

    def transform(self, features):
        """
        Transform new data using the previously fitted PCA model.

        Args:
            features (np.ndarray): New data of shape (n_samples, n_features).

        Returns:
            transformed_data (np.ndarray): Transformed data of shape
                                           (n_samples, num_components).
        """
        # Center the features again
        features_centered = features - self.mean

        # Transform it using the n Eigenvectors
        transformed_data = np.dot(features_centered, self.components)
        return transformed_data
