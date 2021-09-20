import numpy as np

from cassiopeia.data.CassiopeiaTree import CassiopeiaTree
from cassiopeia.solver import dissimilarity_functions

from typing import Callable, Dict, List, Optional, Tuple, Union

from .FitnessEstimator import FitnessEstimator


class SimilarityBasedFitness(FitnessEstimator):
    """
    TODO
    """
    def __init__(
        self,
        dissimilarity_function: Optional[
            Callable[
                [np.array, np.array, int, Dict[int, Dict[int, float]]], float
            ]
        ] = dissimilarity_functions.weighted_hamming_distance,
        power: int = 2,
    ):
        self._dissimilarity_function = dissimilarity_function
        self._power = power

    def estimate_fitness(self, tree: CassiopeiaTree):
        """
        TODO
        """
        tree.compute_dissimilarity_map(
            self._dissimilarity_function,
        )
        dissimilarity_map = tree.get_dissimilarity_map()
        dissimilarity_map = 1.0 / (dissimilarity_map + 1) ** self._power
        fitness = dissimilarity_map.sum(axis=1)
        for leaf in tree.leaves:
            tree.set_attribute(leaf, "fitness", fitness[leaf])
