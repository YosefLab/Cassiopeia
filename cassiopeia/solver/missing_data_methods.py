"""This file contains included missing data imputation methods."""
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from cassiopeia.mixins import is_ambiguous_state, unravel_ambiguous_states
from cassiopeia.solver import solver_utilities


def assign_missing_average(
    character_matrix: pd.DataFrame,
    missing_state_indicator: int,
    left_set: List[str],
    right_set: List[str],
    missing: List[str],
    weights: Optional[Dict[int, Dict[int, float]]] = None,
) -> Tuple[List[str], List[str]]:
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

    Returns:
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

    def score_side(subset_character_matrix, missing_sample):
        score = 0
        for char in range(character_matrix.shape[1]):
            state = character_array[missing_sample, char]
            if state != missing_state_indicator and state != 0:
                all_states = (
                    unravel_ambiguous_states(subset_character_matrix[:, char])
                )
                state_counts = np.unique(all_states, return_counts=True)

                if is_ambiguous_state(state):
                    for ambig_state in state: 
                        ind = np.where(state_counts[0] == ambig_state)
                        if len(ind[0]) > 0:
                            if weights:
                                score += (
                                    weights[char][ambig_state] * state_counts[1][ind[0][0]]
                                )
                            else:
                                score += state_counts[1][ind[0][0]]

                else:
                    ind = np.where(state_counts[0] == state)
                    if len(ind[0]) > 0:
                        if weights:
                            score += (
                                weights[char][state] * state_counts[1][ind[0][0]]
                            )
                        else:
                            score += state_counts[1][ind[0][0]]
                            
        return score

    subset_character_array_left = character_array[left_indices, :]
    subset_character_array_right = character_array[right_indices, :]

    for sample_index in missing_indices:
        left_score = score_side(subset_character_array_left, sample_index)
        right_score = score_side(subset_character_array_right, sample_index)

        if left_score / len(left_set) > right_score / len(right_set):
            left_set.append(sample_names[sample_index])
        else:
            right_set.append(sample_names[sample_index])

    return left_set, right_set
