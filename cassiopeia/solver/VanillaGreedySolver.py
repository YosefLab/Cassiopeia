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
            the "average" method
        meta_data: Any meta data associated with the samples
        priors: Prior probabilities of observing a transition from 0 to any
            state for each character
        prior_function: A function defining a transformation on the priors
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
            out
    """

    def __init__(
        self,
        missing_data_classifier: Union[Callable, str] = "average",
        prior_function: Optional[Callable[[float], float]] = None,
    ):

        super().__init__(prior_function)

        self.missing_data_classifier = missing_data_classifier

    def perform_split(
        self,
        character_matrix: pd.DataFrame,
        mutation_frequencies: Dict[int, Dict[int, int]],
        samples: List[Union[int, str]],
        weights: Optional[Dict[int, Dict[int, float]]] = None,
        missing_state_indicator: int = -1,
    ) -> Tuple[List[Union[int, str]], List[Union[int, str]]]:
        """Performs a partition based on the most frequent (character, state) pair.

        Uses the (character, state) pair to split the list of samples into
        two partitions. In doing so, the procedure makes use of the missing
        data classifier to classify samples that have missing data at that
        character where presence or absence of the character is ambiguous.

        Args:
            character_matrix: Character matrix
            mutation_frequencies: A dictionary containing the frequencies of
                each character/state pair that appear in the character matrix
                restricted to the sample set
            samples: A list of samples to partition
            weights: Weighting of each (character, state) pair. Typically a
                transformation of the priors.
            missing_state_indicator: Character representing missing data.

        Returns:
            A tuple of lists, representing the left and right partitions
        """
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

        for i in samples:
            if character_matrix.loc[i, :][chosen_character] == chosen_state:
                left_set.append(i)
            elif (
                character_matrix.loc[i, :][chosen_character]
                == missing_state_indicator
            ):
                missing.append(i)
            else:
                right_set.append(i)

        if self.missing_data_classifier == "average":
            left_set, right_set = assign_missing_average(
                character_matrix,
                missing_state_indicator,
                left_set,
                right_set,
                missing,
            )
        return left_set, right_set
