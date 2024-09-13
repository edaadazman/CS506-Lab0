## Please fill in all the parts labeled as ### YOUR CODE HERE

import numpy as np
import pytest
from utils import *


def test_dot_product():
    vector1 = np.array([1, 2, 3])
    vector2 = np.array([4, 5, 6])

    result = dot_product(vector1, vector2)

    assert result == 32, f"Expected 32, but got {result}"


def test_cosine_similarity():
    target_vector = np.array([1, 0])

    # Define the array of vectors to compare against
    vectors = np.array(
        [
            [1, 0],
            [0, 1],
            [-1, 0],
        ]
    )

    # Call the nearest_neighbor function
    result = nearest_neighbor(target_vector, vectors)

    # Define the expected index
    expected_index = 0

    # Assert that the result matches the expected index
    assert (
        result == expected_index
    ), f"Expected index {expected_index}, but got {result}"


def test_nearest_neighbor():
    ### YOUR CODE HERE
    # Define two vectors
    v1 = np.array([1, 0])
    v2 = np.array([1, 1])

    # Compute the cosine similarity using the function
    result = cosine_similarity(v1, v2)

    # Calculate the expected result manually
    # Cosine similarity = (1*1 + 0*1) / (sqrt(1^2 + 0^2) * sqrt(1^2 + 1^2)) = 1 / (1 * sqrt(2))
    expected_result = 1 / np.sqrt(2)

    # Assert that the result is close to the expected result
    assert np.isclose(
        result, expected_result
    ), f"Expected {expected_result}, but got {result}"
