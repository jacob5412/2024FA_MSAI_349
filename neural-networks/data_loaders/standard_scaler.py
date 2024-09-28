import numpy as np


class StandardScaler:
    """
    StandardScaler scales input data to have a mean of 0 and a
    standard deviation of 1.

    Attributes:
        mean (numpy.ndarray): The mean values for each feature.
        std (numpy.ndarray): The standard deviation values for each feature.
        epsilon (float): To handle divide by zero errors.
    """

    def __init__(self, epsilon=1e-10) -> None:
        self.mean = None
        self.std = None
        self.epsilon = epsilon

    def fit(self, features):
        """
        Compute the mean and standard deviation of input features for scaling.

        Args:
            features (np.ndarray): Input data of shape (n_samples, n_features).
        """
        self.mean = np.mean(features, axis=0)
        self.std = np.std(features, axis=0)

    def transform(self, features):
        """
        Scale the input features using the computed mean and
        standard deviation.

        Args:
            features (np.ndarray): Input data of shape (n_samples, n_features).

        Returns:
            np.ndarray: Scaled data with a mean of 0 and standard deviation
                        of 1.
        """
        return (features - self.mean) / (self.std + self.epsilon)

    def inverse_transform(self, features_scaled):
        """
        Inverse transform scaled data to the original scale using the
        computed mean and standard deviation.

        Args:
            features_scaled (np.ndarray): Scaled data of shape
                                          (n_samples, n_features).

        Returns:
            np.ndarray: Data in the original scale.
        """
        return features_scaled * (self.std + self.epsilon) + self.mean
