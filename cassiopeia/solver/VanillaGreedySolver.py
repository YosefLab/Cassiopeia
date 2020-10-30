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
from cassiopeia.solver import utils


class VanillaGreedySolver(GreedySolver.GreedySolver):
    def __init__(
        self,
        character_matrix: pd.DataFrame,
        unedit_char: str,
        missing_char: str,
        meta_data: Optional[pd.DataFrame] = None,
        priors: Optional[Dict] = None,
        missing_data_classifier = Union[Callable, str],
        fuzzy_solver: bool = False,
    ):

        super().__init__(character_matrix, unedit_char, missing_char, meta_data, priors)

        self.missing_data_classifier = missing_data_classifier
        self.fuzzy_solver = fuzzy_solver

    def perform_split(
        self, samples: List[int] = None
    ) -> Tuple[List[int], List[int]]:
        """Performs a partition based on the most frequent (character, state) pair.
        
        Uses the (character, state) pair to split the list of samples into
        two partitions. In doing so, the procedure makes use of the missing
        data classifier passed into the class.

        Args:
            samples: A list of samples to partition
        
        Returns:
            A tuple of lists, representing the left and right partitions
        """
        if not samples:
            samples = list(range(self.prune_cm.shape[0]))
        F = utils.compute_mutation_frequencies(samples)
        freq = 0
        char = 0
        state = 0
        for i in F:
            for j in F[i]:
                if j != self.missing_char and j != '0':
                    if F[i][j] > freq and F[i][j] < len(samples) - F[i][self.missing_char]:
                        char, state = i,j
                        freq = F[i][j]
        if freq == 0:
            return self.random_nontrivial_cut(samples)
        S = []
        Sc = []
        missing = []

        for i in samples:
            if self.prune_cm.iloc[i, char] == state:
                S.append(i)
            elif self.prune_cm.iloc[i, char] == -1:
                missing.append(i)
            else:
                Sc.append(i)
        
        S, Sc = self.assign_missing_average(S, Sc, missing)
            
        return S, Sc

    def random_nontrivial_cut(self, samples):
        assert len(samples) > 1
        S = []
        Sc = []
        S.append(samples[0])
        Sc.append(samples[1])
        for i in range(2,len(samples)):
            if np.random.random() > 0.5:
                S.append(samples[i])
            else:
                Sc.append(samples[i])
        return S, Sc

    def assign_missing_average(self, S, Sc, missing):
        for i in missing:
            s_score = 0
            sc_score = 0

            subset_cm = self.prune_cm.iloc[S, :]
            for char in range(self.prune_cm.shape[1]):
                state = self.prune_cm.iloc[i, char]
                if state != self.missing_char and state != '0':
                    state_counts = np.unique(subset_cm.iloc[:,char], return_counts = True)
                    ind = np.where(state_counts[0] == state)
                    if len(ind[0]) > 0:
                        s_score += state_counts[1][ind[0][0]]
                    else:
                        s_score += 0

            subset_cm = self.prune_cm.iloc[Sc, :]
            for char in range(self.prune_cm.shape[1]):
                state = self.prune_cm.iloc[i, char]
                if state != self.missing_char and state != '0':
                    state_counts = np.unique(subset_cm.iloc[:,char], return_counts = True)
                    ind = np.where(state_counts[0] == state)
                    if len(ind[0]) > 0:
                        sc_score += state_counts[1][ind[0][0]]
                    else:
                        sc_score += 0

            if s_score/len(S) > sc_score/len(Sc):
                S.append(i)
            else:
                Sc.append(i)

        return S, Sc