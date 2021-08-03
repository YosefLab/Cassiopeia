"""
This file stores a subclass of GreedySolver, the VanillaGreedySolver. The
inference procedure here is the "vanilla" Cassiopeia-Greedy, originally proposed
in Jones et al, Genome Biology (2020). In essence, the algorithm proceeds by
recursively splitting samples into mutually exclusive groups based on the
presence, or absence, of the most frequently occurring mutation.
"""
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from cassiopeia.solver import (
    GreedySolver,
    missing_data_methods,
    solver_utilities,
)


class VanillaGreedySolver(GreedySolver.GreedySolver):
    """
    A class for the basic Cassiopeia-Greedy solver.

    The VanillaGreedySolver implements a top-down algorithm that optimizes
    for parsimony by recursively splitting the sample set based on the most
    presence, or absence, of the most frequent mutation. Multiple missing data
    imputation methods are included for handling the case when a sample has a
    missing value on the character being split, where presence or absence of the
    character is ambiguous. The user can also specify a missing data method.

    TODO(richardyz98): Implement fuzzysolver

    Args:
        missing_data_classifier: Takes either a string specifying one of the
            included missing data imputation methods, or a function
            implementing the user-specified missing data method. The default is
            the "average" method
        prior_transformation: A function defining a transformation on the priors
            in forming weights to scale frequencies. One of the following:
                "negative_log": Transforms each probability by the negative
                    log (default)
                "inverse": Transforms each probability p by taking 1/p
                "square_root_inverse": Transforms each probability by the
                    the square root of 1/p

    Attributes:
        prior_transformation: Function to transform priors, if these are
            available.
        missing_data_classifier: Function to classify missing data during
            character splits.
    """

    def __init__(
        self,
        missing_data_classifier: Callable = missing_data_methods.assign_missing_average,
        prior_transformation: str = "negative_log",
    ):

        super().__init__(prior_transformation)

        self.missing_data_classifier = missing_data_classifier

    def perform_split(
        self,
        character_matrix: pd.DataFrame,
        samples: List[int],
        weights: Optional[Dict[int, Dict[int, float]]] = None,
        missing_state_indicator: int = -1,
    ) -> Tuple[List[str], List[str]]:
        """Partitions based on the most frequent (character, state) pair.

        Uses the (character, state) pair to split the list of samples into
        two partitions. In doing so, the procedure makes use of the missing
        data classifier to classify samples that have missing data at that
        character where presence or absence of the character is ambiguous.

        Args:
            character_matrix: Character matrix
            samples: A list of samples to partition
            weights: Weighting of each (character, state) pair. Typically a
                transformation of the priors.
            missing_state_indicator: Character representing missing data.

        Returns:
            A tuple of lists, representing the left and right partition groups
        """

        sample_indices = solver_utilities.convert_sample_names_to_indices(
            character_matrix.index, samples
        )
        mutation_frequencies = self.compute_mutation_frequencies(
            samples, character_matrix, missing_state_indicator
        )

        best_frequency = 0
        chosen_character = 0
        chosen_state = 0
        for character in mutation_frequencies:
            for state in mutation_frequencies[character]:
                if state != missing_state_indicator and state != 0:
                    # Avoid splitting on mutations shared by all samples
                    if (
                        mutation_frequencies[character][state]
                        < len(samples)
                        - mutation_frequencies[character][
                            missing_state_indicator
                        ]
                    ):
                        if weights:
                            if (
                                mutation_frequencies[character][state]
                                * weights[character][state]
                                > best_frequency
                            ):
                                chosen_character, chosen_state = (
                                    character,
                                    state,
                                )
                                best_frequency = (
                                    mutation_frequencies[character][state]
                                    * weights[character][state]
                                )
                        else:
                            if (
                                mutation_frequencies[character][state]
                                > best_frequency
                            ):
                                chosen_character, chosen_state = (
                                    character,
                                    state,
                                )
                                best_frequency = mutation_frequencies[
                                    character
                                ][state]

        if chosen_state == 0:
            return samples, []

        left_set = []
        right_set = []
        missing = []

        unique_character_array = character_matrix.to_numpy()
        sample_names = list(character_matrix.index)

        for i in sample_indices:
            if unique_character_array[i, chosen_character] == chosen_state:
                left_set.append(sample_names[i])
            elif (
                unique_character_array[i, chosen_character]
                == missing_state_indicator
            ):
                missing.append(sample_names[i])
            else:
                right_set.append(sample_names[i])

        left_set, right_set = self.missing_data_classifier(
            character_matrix,
            missing_state_indicator,
            left_set,
            right_set,
            missing,
            weights=weights,
        )

        return left_set, right_set
