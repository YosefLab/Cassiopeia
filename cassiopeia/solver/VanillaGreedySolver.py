"""
This file stores a subclass of GreedySolver, the VanillaGreedySolver. The
inference procedure here is the "vanilla" Cassiopeia-Greedy, originally proposed
in Jones et al, Genome Biology (2020). In essence, the algorithm proceeds by
recursively splitting samples into mutually exclusive groups based on the
presence, or absence, of the most frequently occurring mutation.
"""
import numpy as np
import pandas as pd
from typing import Callable, Dict, List, Optional, Tuple, Union

from cassiopeia.solver import GreedySolver
from cassiopeia.solver import missing_data_methods
from cassiopeia.solver import solver_utilities


class VanillaGreedySolver(GreedySolver.GreedySolver):
    """
    TODO: Implement fuzzysolver
    The VanillaGreedySolver implements a top-down algorithm that optimizes
    for parsimony by recursively splitting the sample set based on the most
    presence, or absence, of the most frequent mutation. Multiple missing data
    imputation methods are included for handling the case when a sample has a
    missing value on the character being split, where presence or absence of the
    character is ambiguous. The user can also specify a missing data method.

    Args:
        character_matrix: A character matrix of observed character states for
            all samples
        missing_char: The character representing missing values
        missing_data_classifier: Takes either a string specifying one of the
            included missing data imputation methods, or a function
            implementing the user-specified missing data method. The default is
            the "average" method
        meta_data: Any meta data associated with the samples
        priors: Prior probabilities of observing a transition from 0 to any
            state for each character
        prior_transformation: A function defining a transformation on the priors
            in forming weights to scale frequencies

    Attributes:
        character_matrix: The character matrix describing the samples
        missing_char: The character representing missing values
        meta_data: Data table storing meta data for each sample
        priors: Prior probabilities of character state transitions
        weights: Weights on character/mutation pairs, derived from priors
        tree: The tree built by `self.solve()`. None if `solve` has not been
            called yet
        unique_character_matrix: A character matrix with duplicate rows filtered
        duplicate_groups: A mapping of samples to the set of duplicates that
            share the same character vector. Uses the original sample names
    """

    def __init__(
        self,
        character_matrix: pd.DataFrame,
        missing_char: int,
        missing_data_classifier: Callable = missing_data_methods.assign_missing_average,
        meta_data: Optional[pd.DataFrame] = None,
        priors: Optional[Dict[int, Dict[int, float]]] = None,
        prior_transformation: Optional[
            Callable[[float], float]
        ] = "negative_log",
    ):

        super().__init__(
            character_matrix,
            missing_char,
            meta_data,
            priors,
            prior_transformation,
        )
        self.missing_data_classifier = missing_data_classifier

    def perform_split(
        self,
        samples: List[str],
    ) -> Tuple[List[str], List[str]]:
        """Performs a partition based on the most frequent (character, state) pair.

        Uses the (character, state) pair to split the list of samples into
        two partitions. In doing so, the procedure makes use of the missing
        data classifier to classify samples that have missing data at that
        character where presence or absence of the character is ambiguous.

        Args:
            samples: A list of samples, represented by their names in the
                original character matrix

        Returns:
            A tuple of lists, representing the left and right partition groups
        """
        sample_indices = solver_utilities.convert_sample_names_to_indices(
            self.unique_character_matrix.index, samples
        )
        mutation_frequencies = self.compute_mutation_frequencies(samples)

        best_frequency = 0
        chosen_character = 0
        chosen_state = 0
        for character in mutation_frequencies:
            for state in mutation_frequencies[character]:
                if state != self.missing_char and state != 0:
                    # Avoid splitting on mutations shared by all samples
                    if (
                        mutation_frequencies[character][state]
                        < len(samples)
                        - mutation_frequencies[character][self.missing_char]
                    ):
                        if self.weights:
                            if (
                                mutation_frequencies[character][state]
                                * self.weights[character][state]
                                > best_frequency
                            ):
                                chosen_character, chosen_state = (
                                    character,
                                    state,
                                )
                                best_frequency = (
                                    mutation_frequencies[character][state]
                                    * self.weights[character][state]
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

        unique_character_array = self.unique_character_matrix.to_numpy()
        indices = list(self.unique_character_matrix.index)

        for i in sample_indices:
            if unique_character_array[i, chosen_character] == chosen_state:
                left_set.append(indices[i])
            elif (
                unique_character_array[i, chosen_character] == self.missing_char
            ):
                missing.append(indices[i])
            else:
                right_set.append(indices[i])

        left_set, right_set = self.missing_data_classifier(
            self.unique_character_matrix,
            self.missing_char,
            left_set,
            right_set,
            missing,
            weights=self.weights,
        )

        return left_set, right_set
