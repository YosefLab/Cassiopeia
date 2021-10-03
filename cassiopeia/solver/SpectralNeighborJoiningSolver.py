"""
This file stores a subclass of DistanceSolver, SpectralNeighborJoining. This algorithm is based on the one developed by Jaffe, et al. (2020) in their paper titled "Spectral neighbor joining for reconstruction of latent tree models. 
"""


from typing import Callable, Dict, List, Optional, Tuple, Union

import abc
import networkx as nx
import numpy as np
import pandas as pd

from cassiopeia.data import CassiopeiaTree
from cassiopeia.solver import (
    DistanceSolver,
    dissimilarity_functions,
    solver_utilities,
)


class SpectralNeighborJoiningSolver(DistanceSolver.DistanceSolver):
    """
    Spectral Neighbor-joining class for Cassiopeia.

    Implements a variation on the Spectral Neighbor-Joining algorithm described by Jaffe et al. in their 2020 paper. This class implements the abstract class DistanceSolver. This class inherits the 'solve' method from DistanceSolver with the find_cherry method implemented using a spectral method as in Jaffe et al. (2020).

    Args:
        dissimilarity_function: A function by which to compute the dissimilarity map, corresponding to the affinity R(i, j) in Jaffe et al. (2020). By default we will use exp(-d(i,j)) where d is the metric given by weighted_hamming_distance from dissimilarity functions without weights given.



    """

    def __init__(
        self,
        dissimilarity_function: Optional[
            Callable[
                [np.array, np.array, int, Dict[int, Dict[int, float]]], float
            ]
        ] = dissimilarity_functions.weighted_hamming_distance,
    ):
        super().__init__(dissimilarity_function=dissimilarity_function)
