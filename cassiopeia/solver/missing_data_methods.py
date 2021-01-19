"""This file contains included missing data imputation methods."""

import numpy as np
import pandas as pd
from typing import List, Tuple, Union


def assign_missing_average(
    character_matrix: pd.DataFrame,
    missing_char: int,
    left_set: List[int],
    right_set: List[int],
    missing: List[int],
) -> Tuple[List[int], List[int]]:
    """Implements the "Average" missing data imputation method.

    An on-the-fly missing data imputation method for the Vanilla Greedy
    Solver. It takes in a set of samples that have a missing value at the
    character chosen to split on in a partition. For each of these samples,
    it calculates the average number of mutations that samples on each side
    of the partition share with it and places the sample on the side with
    the higher value.

    Args:
        character_matrix: The character matrix containing the observed character states for
            the samples
        missing_char: The character representing missing values
        left_set: A list of the samples on the left of the partition,
            represented as integer indices
        right_set: A list of the samples on the right of the partition,
            represented as integer indices
        missing: A list of samples with missing data to be imputed,
            represented as integer indices

    Returns:
        A tuple of lists, representing the left and right partitions with
        missing samples imputed
    """
    subset_character_matrix_left = character_matrix[left_set, :]
    subset_character_matrix_right = character_matrix[right_set, :]

    for i in missing:
        left_score = 0
        right_score = 0

        for char in range(character_matrix.shape[1]):
            state = character_matrix[i][char]
            if state != missing_char and state != 0:
                state_counts = np.unique(
                    subset_character_matrix_left[:, char], return_counts=True
                )
                ind = np.where(state_counts[0] == state)
                if len(ind[0]) > 0:
                    left_score += state_counts[1][ind[0][0]]
                else:
                    left_score += 0

        for char in range(character_matrix.shape[1]):
            state = character_matrix[i][char]
            if state != missing_char and state != 0:
                state_counts = np.unique(
                    subset_character_matrix_right[:, char], return_counts=True
                )
                ind = np.where(state_counts[0] == state)
                if len(ind[0]) > 0:
                    right_score += state_counts[1][ind[0][0]]
                else:
                    right_score += 0

        if left_score / len(left_set) > right_score / len(right_set):
            left_set.append(i)
        else:
            right_set.append(i)

    return left_set, right_set
