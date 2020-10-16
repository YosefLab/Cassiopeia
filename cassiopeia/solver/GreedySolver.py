"""
This file stores a subclass of CassiopeiaSolver, the GreedySolver. The inference
procedure here is the "vanilla" Cassiopeia-Greedy, originally proposed in 
Jones et al, Genome Biology (2020). In essence, the algorithm proceeds by
recursively splitting samples into mutually exclusive groups based on the
presence, or absence, of the most frequently occurring mutation.
"""
import abc
import pandas as pd
from typing import List, Optional, Tuple

from cassiopeia.solver import CassiopeiaSolver


class GreedySolver(CassiopeiaSolver.CassiopeiaSolver):

    def __init__(
        self,
        character_matrix: pd.DataFrame,
        meta_data: Optional[pd.DataFrame] = None,
        priors: Optional[Dict] = None,
    ):

        super().__init__(character_matrix, meta_data, priors)

    @abc.abstractmethod
    def perform_split(
        self, samples: List[int]
    ) -> Tuple[List[int], List[int]]:
        """Performs a partition of the samples.

        Args:
            samples: A list of samples to partition
        
        Returns:
            A tuple of lists, representing the left and right partitions
        """
        pass
    
    def compute_mutation_frequencies(self, samples: List[int]) -> pd.DataFrame:
        """Computes the frequency of mutations in the list of samples.

        Args:
            samples: A list of samples

        Returns:
            A dataframe mapping mutations to frequencies.
        """
        pass