"""
This file stores a subclass of CassiopeiaSolver, the GreedySolver. The inference
procedure here is the "vanilla" Cassiopeia-Greedy, originally proposed in 
Jones et al, Genome Biology (2020). In essence, the algorithm proceeds by
recursively splitting samples into mutually exclusive groups based on the
presence, or absence, of the most frequently occurring mutation.
"""
import pandas as pd
from typing import Callable, Dict, List, Optional, Tuple, Union

from cassiopeia.solver import CassiopeiaSolver


class GreedySolver(CassiopeiaSolver.CassiopeiaSolver):
    def __init__(
        self,
        character_matrix: pd.DataFrame,
        meta_data: Optional[pd.DataFrame] = None,
        priors: Optional[Dict] = None,
        missing_data_classifier=Union[Callable, str],
        fuzzy_solver: bool = False,
    ):

        super().__init__(character_matrix, meta_data, priors)

        self.missing_data_classifier = missing_data_classifier
        self.fuzzy_solver = fuzzy_solver

    def solve(self):
        """Implements the solver routine for Cassiopeia-Greedy

        Cassiopeia Greedy proceeds by identifying first a mutation that is most
        likley to have occurred earliest in the phylogeny, partitioning samples
        into two sets based on the presence or absence of this mutation, and then
        recursing on these two sets until sets of one sample are obtained.
        To note, at each partitioning step, samples with missing data at the
        character of interest are assigned to one of the two partitions based on
        the function stored in self.missing_data_classifier.
        """
        pass

    def find_split(self) -> Tuple[int, str]:
        """Identifies a (character, mutation) pair on which to split.

        Identifies the most (character, mutation) pair that was most likely to
        have occurred earliest in the phylogeny. Without priors, this is simply
        the most frequently occurring mutation. With priors, this is is the most
        frequently occurring rare mutation (i.e. frequencies are weighted by a
        prior probability of the mutation occurring).
        """
        pass

    def perform_split(
        self, samples: List[int], split: Tuple[int, str]
    ) -> Tuple[nx.DiGraph, nx.DiGraph]:
        """Performs a partition based on the (character, mutation) pair.
        
        Uses the (character, mutation) pair to split the list of samples into
        two partitions. In doing so, the procedure makes use of the missing
        data classifier passed into the class.
        """
        pass
