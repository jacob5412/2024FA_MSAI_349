"""
Distance Utils
"""

import numpy as np


def euclidean_distance(mat_a, mat_b):
    """
    Calculate the pairwise Euclidean distances between row vectors
    of two matrices. Uses efficient euclidean distance, similar to
    sklearn's approach.

    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.euclidean_distances.html

    Args:
        mat_a (numpy.ndarray): The first matrix.
        mat_b (numpy.ndarray): The second matrix.

    Returns:
        Resulting matrix will have dimensions corresponding to the number
        of rows in mat_a and mat_b, where each element (i, j) in the
        matrix represents the Euclidean distance between the i-th row of
        mat_a and the j-th row of mat_b.

    Raises:
        ValueError: If the input matrices mat_a and mat_b do not have the
                    compatible shapes for dot products.
    """
    # if vectors, then reshape
    if len(mat_a.shape) == 1:
        mat_a = mat_a.reshape(1, -1)
    if len(mat_b.shape) == 1:
        mat_b = mat_b.reshape(1, -1)
    if mat_a.shape[-1] != mat_b.shape[-1]:
        raise ValueError("Input matrices must have compatible shapes for dot products")

    # calculating distance
    l1_norm_a = np.sum(mat_a**2, axis=1, keepdims=True)
    l1_norm_b = np.sum(mat_b**2, axis=1, keepdims=True)
    dist = np.sqrt(np.maximum(l1_norm_a - 2 * np.dot(mat_a, mat_b.T) + l1_norm_b.T, 0))
    return dist


def cosine_distance(mat_a, mat_b):
    """
    Calculate the pairwise Cosine distance between row vectors
    of two matrices.

    Args:
        mat_a (numpy.ndarray): The first matrix.
        mat_b (numpy.ndarray): The second matrix.

    Returns:
        Resulting matrix will have dimensions corresponding to the number
        of rows in mat_a and mat_b, where each element (i, j) in the
        matrix represents the Cosine distance between the i-th row of
        mat_a and the j-th row of mat_b.

    Raises:
        ValueError: If the input matrices mat_a and mat_b do not have the
                    compatible shapes for dot products.
    """
    # if vectors, then reshape
    if len(mat_a.shape) == 1:
        mat_a = mat_a.reshape(1, -1)
    if len(mat_b.shape) == 1:
        mat_b = mat_b.reshape(1, -1)
    if mat_a.shape[-1] != mat_b.shape[-1]:
        raise ValueError("Input matrices must have compatible shapes for dot products")

    # calculating distance
    l2_norm_a = np.sqrt(np.sum(mat_a**2, axis=1, keepdims=True))
    l2_norm_b = np.sqrt(np.sum(mat_b**2, axis=1, keepdims=True))
    cosine_sim = np.dot(mat_a, mat_b.T) / (l2_norm_a * l2_norm_b.T)
    return 1 - cosine_sim
