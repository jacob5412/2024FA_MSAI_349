"""
Unit tests
"""

import unittest
from random import randint

import numpy as np
from sklearn.decomposition import PCA as sklearn_PCA
from sklearn.metrics import pairwise
from utilities.distance_utils import cosine_distance, euclidean_distance
from utilities.pca_utils import PCA


class TestDistanceMetrics(unittest.TestCase):
    """
    Unit tests for custom distance metrics (cosine and Euclidean distances)
    and comparisons with scikit-learn's pairwise distances.
    """

    def setUp(self) -> None:
        self.mat1 = np.array([[1, 2, 32], [4, 5, 6]])
        self.mat2 = np.array([[4, 543, 6], [7.2, 8, 9]])
        self.mat3 = np.array([[0.00013, 0.4311, 4134], [-341, 5.0, 6.0]])
        random_mat_nrows = randint(1, 10)
        random_mat_ncols = randint(1, 10)
        self.mat4 = np.random.rand(random_mat_nrows, random_mat_ncols)
        self.mat5 = np.random.rand(random_mat_nrows, random_mat_ncols)
        self.mat6 = np.array([[-1.0, 2.4, 4.0, 514.31]])
        self.mat7 = np.array([[7, 8, 10]])
        self.vec1 = np.array([431, 2, 3])
        self.vec2 = np.array([3, 431, -1])

    def test_euclidean_distance_1(self):
        """
        Test Euclidean distance between mat1 and mat2.
        """
        custom_result = euclidean_distance(self.mat1, self.mat2)
        sklearn_result = pairwise.euclidean_distances(self.mat1, self.mat2)
        np.testing.assert_array_almost_equal(custom_result, sklearn_result, decimal=1)

    def test_euclidean_distance_2(self):
        """
        Test Euclidean distance between mat2 and mat3.
        """
        custom_result = euclidean_distance(self.mat2, self.mat3)
        sklearn_result = pairwise.euclidean_distances(self.mat2, self.mat3)
        np.testing.assert_array_almost_equal(custom_result, sklearn_result, decimal=1)

    def test_euclidean_distance_3(self):
        """
        Test Euclidean distance with matrices of different length.
        """
        with self.assertRaises(ValueError):
            euclidean_distance(self.mat3, self.mat6)

    def test_euclidean_distance_4(self):
        """
        Test Euclidean distance with random length matrices.
        """
        custom_result = euclidean_distance(self.mat4, self.mat5)
        sklearn_result = pairwise.euclidean_distances(self.mat4, self.mat5)
        np.testing.assert_array_almost_equal(custom_result, sklearn_result, decimal=1)

    def test_euclidean_distance_5(self):
        """
        Test Euclidean distance with different length matrices.
        """
        custom_result = euclidean_distance(self.mat3, self.mat7)
        sklearn_result = pairwise.euclidean_distances(self.mat3, self.mat7)
        np.testing.assert_array_almost_equal(custom_result, sklearn_result, decimal=1)

    def test_euclidean_distance_6(self):
        """
        Test Euclidean distance with vectors.
        """
        custom_result = euclidean_distance(self.vec1, self.vec2)
        sklearn_result = pairwise.euclidean_distances(
            self.vec1.reshape(1, -1), self.vec2.reshape(1, -1)
        )
        np.testing.assert_array_almost_equal(custom_result, sklearn_result, decimal=1)

    def test_cosine_distance_1(self):
        """
        Test Cosine distance between mat1 and mat2.
        """
        custom_result = cosine_distance(self.mat1, self.mat2)
        sklearn_result = pairwise.cosine_distances(self.mat1, self.mat2)
        np.testing.assert_array_almost_equal(custom_result, sklearn_result, decimal=1)

    def test_cosine_distance_2(self):
        """
        Test Cosine distance between mat2 and mat3.
        """
        custom_result = cosine_distance(self.mat2, self.mat3)
        sklearn_result = pairwise.cosine_distances(self.mat2, self.mat3)
        np.testing.assert_array_almost_equal(custom_result, sklearn_result, decimal=1)

    def test_cosine_distance_3(self):
        """
        Test Cosine distance with matrices of different length.
        """
        with self.assertRaises(ValueError):
            cosine_distance(self.mat3, self.mat6)

    def test_cosine_distance_4(self):
        """
        Test Cosine distance with random length matrices.
        """
        custom_result = cosine_distance(self.mat4, self.mat5)
        sklearn_result = pairwise.cosine_distances(self.mat4, self.mat5)
        np.testing.assert_array_almost_equal(custom_result, sklearn_result, decimal=1)

    def test_cosine_distance_5(self):
        """
        Test Cosine distance with different length matrices.
        """
        custom_result = cosine_distance(self.mat3, self.mat7)
        sklearn_result = pairwise.cosine_distances(self.mat3, self.mat7)
        np.testing.assert_array_almost_equal(custom_result, sklearn_result, decimal=1)

    def test_cosine_distance_6(self):
        """
        Test Cosine distance with vectors.
        """
        custom_result = cosine_distance(self.vec1, self.vec2)
        sklearn_result = pairwise.cosine_distances(
            self.vec1.reshape(1, -1), self.vec2.reshape(1, -1)
        )
        np.testing.assert_array_almost_equal(custom_result, sklearn_result, decimal=1)


class TestPCA(unittest.TestCase):
    """
    Test cases for the PCA (Principal Component Analysis).
    """

    def setUp(self) -> None:
        self.mat1 = np.array([[41, 2, 3], [41, 341, 431], [431, 698, 9], [10, 431, -314]])
        self.mat2 = np.array([[1, 431.2], [4, -35], [431.1, 0.843], [984, 11]])
        self.num_components = 2

    @staticmethod
    def _flip_signs(mat_a, mat_b):
        """
        Utility function for resolving the sign ambiguity in PCA.
        https://stackoverflow.com/questions/58666635/implementing-pca-with-numpy

        Args:
            mat_a (np.ndarray): First matrix.
            mat_b (np.ndarray): Second matrix.

        Returns:
            np.ndarray, np.ndarray: Two matrices with aligned signs.
        """
        signs = np.sign(mat_a) * np.sign(mat_b)
        return mat_a, mat_b * signs

    @staticmethod
    def _standard_scaler(mat):
        """
        Standardize a matrix by subtracting the mean and dividing by the
        standard deviation.

        Args:
            mat (np.ndarray): Input matrix to be standardized.

        Returns:
            np.ndarray: Standardized matrix.
        """
        mat_mean = np.mean(mat, axis=0)
        mat_std = np.std(mat, axis=0)
        return (mat - mat_mean) / mat_std

    def test_pca_sklearn_1(self):
        """
        Test PCA with a standardized matrix.
        """
        scaled_mat1 = self._standard_scaler(self.mat1)
        custom_pca = PCA(self.num_components)
        custom_pca.fit(scaled_mat1)
        custom_reduced_data = custom_pca.transform(scaled_mat1)
        sklearn_pca = sklearn_PCA(self.num_components)
        sklearn_reduced_data = sklearn_pca.fit_transform(scaled_mat1)
        np.testing.assert_array_almost_equal(
            *self._flip_signs(custom_reduced_data, sklearn_reduced_data), decimal=1
        )

    def test_pca_sklearn_2(self):
        """
        Test PCA with a standardized matrix.
        """
        scaled_mat2 = self._standard_scaler(self.mat2)
        custom_pca = PCA(self.num_components)
        custom_pca.fit(scaled_mat2)
        custom_reduced_data = custom_pca.transform(scaled_mat2)
        sklearn_pca = sklearn_PCA(self.num_components)
        sklearn_reduced_data = sklearn_pca.fit_transform(scaled_mat2)
        np.testing.assert_array_almost_equal(
            *self._flip_signs(custom_reduced_data, sklearn_reduced_data), decimal=1
        )


if __name__ == "__main__":
    unittest.main()
