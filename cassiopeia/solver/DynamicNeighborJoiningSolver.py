"""
This file stores a subclass of NeighborJoiningSolver, DynamicNeighborJoining.
The inference procedure is the Dymanic Neighbor-Joining algorithm proposed by
Clausen (2023). https://pubmed.ncbi.nlm.nih.gov/36453849/
"""
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from cassiopeia.mixins import DistanceSolverError
from cassiopeia.solver import (
    NeighborJoiningSolver,
    dissimilarity_functions,
)

class DynamicNeighborJoiningSolver(NeighborJoiningSolver):
    """
    Dynamic Neighbor-Joining class for Cassiopeia.

    Implements the Dynamic Neighbor-Joining algorithm described by Clausen (2023)
    as a derived class of CCPhyloSolver. This class inherits the generic
    `solve` method. This algorithm is guaranteed to return an exact solution.

    Args:
        dissimilarity_function: A function by which to compute the dissimilarity
            map. Optional if a dissimilarity map is already provided.
        add_root: Whether or not to add an implicit root the tree, i.e. a root
            with unmutated characters. If set to False, and no explicit root is
            provided in the CassiopeiaTree, then will return an unrooted,
            undirected tree
        prior_transformation: Function to use when transforming priors into
            weights. Supports the following transformations:
                "negative_log": Transforms each probability by the negative
                    log (default)
                "inverse": Transforms each probability p by taking 1/p
                "square_root_inverse": Transforms each probability by the
                    the square root of 1/p

    Attributes:
        dissimilarity_function: Function used to compute dissimilarity between
            samples.
        add_root: Whether or not to add an implicit root the tree.
        prior_transformation: Function to use when transforming priors into
            weights.

    """

    def __init__(
        self,
        dissimilarity_function: Optional[
            Callable[
                [np.array, np.array, int, Dict[int, Dict[int, float]]], float
            ]
        ] = dissimilarity_functions.weighted_hamming_distance,
        add_root: bool = False,
        prior_transformation: str = "negative_log",
    ):
        # setup fast solver
        self.fast_solver = "ccphylo"
        self.fast_method = "dnj"

        super().__init__(
            dissimilarity_function=dissimilarity_function,
            add_root=add_root,
            prior_transformation=prior_transformation,
            fast = True,
        )