"""
This file stores a subclass of GreedySolver, the VanillaGreedySolver. The
inference procedure here is the "vanilla" Cassiopeia-Greedy, originally proposed
in Jones et al, Genome Biology (2020). In essence, the algorithm proceeds by
recursively splitting samples into mutually exclusive groups based on the
presence, or absence, of the most frequently occurring mutation.
"""
import pandas as pd
from typing import Callable, Dict, List, Optional, Tuple, Union

from cassiopeia.solver import GreedySolver
from cassiopeia.solver.missing_data_methods import assign_missing_average


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
            the "average" method.
        meta_data: Any meta data associated with the samples
        priors: Prior probabilities of observing a transition from 0 to any
            state for each character. Sets weights to be negative log
            transformations of these probabilities.
        weights: A set of optional weights on character/mutation pairs to scale
            frequency. Overrides weights from priors

    Attributes:
        character_matrix: The character matrix describing the samples
        missing_char: The character representing missing values
        meta_data: Data table storing meta data for each sample
        priors: Prior probabilities of character state transitions
        weights: Weights on character/mutation pairs, derived from priors or
            explicitly provided
        tree: The tree built by `self.solve()`. None if `solve` has not been
            called yet
        prune_cm: A character matrix with duplicate rows filtered out
    """

    def __init__(
        self,
        character_matrix: pd.DataFrame,
        missing_char: str,
        missing_data_classifier: Union[Callable, str] = "average",
        meta_data: Optional[pd.DataFrame] = None,
        priors: Optional[Dict[int, Dict[int, float]]] = None,
        weights: Optional[Dict[int, Dict[int, float]]] = None,
    ):

        super().__init__(
            character_matrix, missing_char, meta_data, priors, weights
        )

        self.missing_data_classifier = missing_data_classifier

    def perform_split(
        self,
        mutation_frequencies: Dict[int, Dict[int, int]],
        samples: List[int],
    ) -> Tuple[List[int], List[int]]:
        """Performs a partition based on the most frequent (character, state) pair.

        Uses the (character, state) pair to split the list of samples into
        two partitions. In doing so, the procedure makes use of the missing
        data classifier to classify samples that have missing data at that
        character where presence or absence of the character is ambiguous.

        Args:
            mutation_frequencies: A dictionary containing the frequencies of
                each character/state pair that appear in the character matrix
                restricted to the sample set
            samples: A list of samples to partition

        Returns:
            A tuple of lists, representing the left and right partitions
        """
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

        for i in samples:
            if self.prune_cm.iloc[i, chosen_character] == chosen_state:
                left_set.append(i)
            elif self.prune_cm.iloc[i, chosen_character] == self.missing_char:
                missing.append(i)
            else:
                right_set.append(i)

        if self.missing_data_classifier == "average":
            left_set, right_set = assign_missing_average(
                self.prune_cm, self.missing_char, left_set, right_set, missing
            )
        return left_set, right_set
