"""This file contains included missing data imputation methods."""


import numpy as np
import pandas as pd

from cassiopeia.mixins import unravel_ambiguous_states
from cassiopeia.solver import solver_utilities


def assign_missing_average(
    character_matrix: pd.DataFrame,
    missing_state_indicator: int,
    left_set: list[str],
    right_set: list[str],
    missing: list[str],
    weights: dict[int, dict[int, float]] | None = None,
) -> tuple[list[str], list[str]]:
    """Implements the "Average" missing data imputation method.

    An on-the-fly missing data imputation method for the VanillaGreedy
    Solver and variants. It takes in a set of samples that have a missing
    value at the character chosen to split on in a partition. For each of
    these samples, it calculates the average number of mutations that
    samples on each side of the partition share with it and places the
    sample on the side with the higher value.

    Args:
        character_matrix: The character matrix containing the observed
            character states for the samples
        missing_state_indicator: The character representing missing values
        left_set: A list of the samples on the left of the partition,
            represented by their names in the original character matrix
        right_set: A list of the samples on the right of the partition,
            represented by their names in the original character matrix
        missing: A list of samples with missing data to be imputed,
            represented by their names in the original character matrix
        weights: A set of optional weights for character/state mutation pairs

    Returns
    -------
        A tuple of lists, representing the left and right partitions with
        missing samples imputed
    """
    # A helper function to calculate the number of shared character/state pairs
    # shared between a missing sample and a side of the partition
    sample_names = list(character_matrix.index)
    character_array = character_matrix.to_numpy()
    left_indices = solver_utilities.convert_sample_names_to_indices(
        sample_names, left_set
    )
    right_indices = solver_utilities.convert_sample_names_to_indices(
        sample_names, right_set
    )
    missing_indices = solver_utilities.convert_sample_names_to_indices(
        sample_names, missing
    )

    def score_side(subset_character_states, query_states, weights):
        score = 0
        for char in range(len(subset_character_states)):

            query_state = [
                q
                for q in query_states[char]
                if q != 0 and q != missing_state_indicator
            ]
            all_states = np.array(subset_character_states[char])
            for q in query_state:
                if weights:
                    score += weights[char][q] * np.count_nonzero(
                        all_states == q
                    )
                else:
                    score += np.count_nonzero(all_states == q)

        return score

    subset_character_array_left = character_array[left_indices, :]
    subset_character_array_right = character_array[right_indices, :]

    all_left_states = [
        unravel_ambiguous_states(subset_character_array_left[:, char])
        for char in range(subset_character_array_left.shape[1])
    ]
    all_right_states = [
        unravel_ambiguous_states(subset_character_array_right[:, char])
        for char in range(subset_character_array_right.shape[1])
    ]

    for sample_index in missing_indices:

        all_states_for_sample = [
            unravel_ambiguous_states([character_array[sample_index, char]])
            for char in range(character_array.shape[1])
        ]

        left_score = score_side(
            np.array(all_left_states, dtype=object),
            np.array(all_states_for_sample, dtype=object),
            weights,
        )
        right_score = score_side(
            np.array(all_right_states, dtype=object),
            np.array(all_states_for_sample, dtype=object),
            weights,
        )

        if (left_score / len(left_set)) > (right_score / len(right_set)):
            left_set.append(sample_names[sample_index])
        else:
            right_set.append(sample_names[sample_index])

    return left_set, right_set
