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


class VanillaGreedySolver(GreedySolver.GreedySolver):
    def __init__(
        self,
        character_matrix: pd.DataFrame,
        missing_char: str,
        missing_data_classifier: Union[Callable, str],
        meta_data: Optional[pd.DataFrame] = None,
        priors: Optional[Dict] = None,
        fuzzy_solver: bool = False,
    ):

        super().__init__(character_matrix, missing_char, meta_data, priors)

        self.missing_data_classifier = missing_data_classifier
        self.fuzzy_solver = fuzzy_solver

    def perform_split(
        self,
        mutation_frequencies: Dict[int, Dict[str, int]],
        samples: List[int] = None,
    ) -> Tuple[List[int], List[int]]:
        """Performs a partition based on the most frequent (character, state) pair.

        Uses the (character, state) pair to split the list of samples into
        two partitions. In doing so, the procedure makes use of the missing
        data classifier passed into the class.

        Args:
            mutation_frequencies: A dictionary containing the frequencies of
                each character/state pair that appear in the character matrix
                restricted to the sample set
            samples: A list of samples to partition

        Returns:
            A tuple of lists, representing the left and right partitions
        """
        if not samples:
            samples = list(range(self.prune_cm.shape[0]))
        freq = 0
        char = 0
        state = 0
        for i in mutation_frequencies:
            for j in mutation_frequencies[i]:
                if j != self.missing_char and j != "0":
                    if (
                        mutation_frequencies[i][j] > freq
                        and mutation_frequencies[i][j]
                        < len(samples)
                        - mutation_frequencies[i][self.missing_char]
                    ):
                        char, state = i, j
                        freq = mutation_frequencies[i][j]
        left_set = []
        right_set = []
        missing = []

        for i in samples:
            if self.prune_cm.iloc[i, char] == state:
                left_set.append(i)
            elif self.prune_cm.iloc[i, char] == self.missing_char:
                missing.append(i)
            else:
                right_set.append(i)

        left_set, right_set = self.assign_missing_average(
            left_set, right_set, missing
        )

        if len(left_set) == 0 or len(right_set) == 0:
            return self.random_nontrivial_cut(samples)
        return left_set, right_set

    def random_nontrivial_cut(
        self, samples: List[int]
    ) -> Tuple[List[int], List[int]]:
        """Performs a random partition of the samples, but garuntees that both
        sides of the partition contain at least one sample.

        Args:
            samples: A list of samples to paritition

        Returns:
            A tuple of lists, representing the left and right partitions
        """
        assert len(samples) > 1
        left_set = []
        right_set = []
        left_set.append(samples[0])
        right_set.append(samples[1])
        for i in range(2, len(samples)):
            if np.random.random() > 0.5:
                left_set.append(samples[i])
            else:
                right_set.append(samples[i])
        return left_set, right_set

    def assign_missing_average(
        self, left_set: List[int], right_set: List[int], missing: List[int]
    ) -> Tuple[List[int], List[int]]:
        """Implements the "Average" missing data imputation method.

        An on-the-fly missing data imputation method for the Vanilla Greedy
        Solver. It takes in a set of samples that have a missing value at the
        character chosen to split on in a partition. For each of these samples,
        it calculates the average number of mutations that samples on each side
        of the partition share with it and places the sample on the side with
        the higher value.

        Args:
            left_set: A list of the samples on the left of the partition
            right_set: A list of the samples on the right of the partition
            missing: A list of samples with missing data to be imputed

        Returns:
            A tuple of lists, representing the left and right partitions with
            missing samples imputed
        """
        for i in missing:
            left_score = 0
            right_score = 0

            subset_cm = self.prune_cm.iloc[left_set, :]
            for char in range(self.prune_cm.shape[1]):
                state = self.prune_cm.iloc[i, char]
                if state != self.missing_char and state != "0":
                    state_counts = np.unique(
                        subset_cm.iloc[:, char], return_counts=True
                    )
                    ind = np.where(state_counts[0] == state)
                    if len(ind[0]) > 0:
                        left_score += state_counts[1][ind[0][0]]
                    else:
                        left_score += 0

            subset_cm = self.prune_cm.iloc[right_set, :]
            for char in range(self.prune_cm.shape[1]):
                state = self.prune_cm.iloc[i, char]
                if state != self.missing_char and state != "0":
                    state_counts = np.unique(
                        subset_cm.iloc[:, char], return_counts=True
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
