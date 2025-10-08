"""
This file stores a subclass of DistanceSolver, UPGMA. The inference procedure is
a hierarchical clustering algorithm proposed by Sokal and Michener (1958) that
iteratively joins together samples with the minimum dissimilarity.
"""
from collections import defaultdict
from collections.abc import Callable

import networkx as nx
import numba
import numpy as np
import pandas as pd

from cassiopeia.data import CassiopeiaTree
from cassiopeia.mixins import DistanceSolverError
from cassiopeia.solver import DistanceSolver, dissimilarity_functions


class UPGMASolver(DistanceSolver.DistanceSolver):
    """
    UPGMA CassiopeiaSolver.

    Implements the UPGMA algorithm described as a derived class of
    DistanceSolver. This class inherits the generic `solve` method, but
    implements its own procedure for finding cherries by minimizing the
    dissimilarity between samples. After joining nodes, the dissimilarities
    are updated by averaging the distances of elements in the new cluster
    with each existing node. Produces a rooted tree that is assumed to be
    ultrametric. If fast is set to True, a fast UPGMA implementation of is used.

    Args:
        dissimilarity_function: A function by which to compute the dissimilarity
            map. Optional if a dissimilarity map is already provided.
        prior_transformation: Function to use when transforming priors into
            weights. Supports the following transformations:
                "negative_log": Transforms each probability by the negative
                    log (default)
                "inverse": Transforms each probability p by taking 1/p
                "square_root_inverse": Transforms each probability by the
                    the square root of 1/p
        fast: Whether to use a fast implementation of UPGMA.
        implementation: Which fast implementation to use. Options are:
            "ccphylo_upgma": Uses the fast UPGMA implementation from CCPhylo.
        threads: Number of threads to use for dissimilarity map computation.

    Attributes
    ----------
        dissimilarity_function: Function used to compute dissimilarity between
            samples.
        add_root: Whether or not to add an implicit root the tree.
        prior_transformation: Function to use when transforming priors into
            weights.
        threads: Number of threads to use for dissimilarity map computation.
    """

    def __init__(
        self,
        dissimilarity_function: Callable[[np.array, np.array, int, dict[int, dict[int, float]]], float] | None = dissimilarity_functions.weighted_hamming_distance,
        prior_transformation: str = "negative_log",
        fast: bool = False,
        implementation: str = "ccphylo_upgma",
        threads: int = 1,
    ):

        if fast:
            if implementation in ["ccphylo_upgma"]:
                self._implementation = implementation
            else:
                raise DistanceSolverError(
                    "Invalid fast implementation of UPGMA. Options are: "
                    "'ccphylo_upgma'"
                )
        else:
            self._implementation = "generic_upgma"

        super().__init__(
            dissimilarity_function=dissimilarity_function,
            add_root=True,
            prior_transformation=prior_transformation,
            threads=threads
        )

        self.__cluster_to_cluster_size = defaultdict(int)

    def root_tree(
        self, tree: nx.Graph, root_sample: str, remaining_samples: list[str]
    ):
        """Roots a tree produced by UPGMA.

        Adds the root at the top of the UPGMA reconstructed tree. By the
        ultrametric assumption, the root is placed as the parent to the last
        two unjoined nodes.

        Args:
            tree: Networkx object representing the tree topology
            root_sample: Ignored in this case, the root is known in this case
            remaining_samples: The last two unjoined nodes in the tree

        Returns
        -------
            A rooted tree.
        """
        tree.add_node("root")
        tree.add_edges_from(
            [("root", remaining_samples[0]), ("root", remaining_samples[1])]
        )

        rooted_tree = nx.DiGraph()
        for e in nx.dfs_edges(tree, source="root"):
            rooted_tree.add_edge(e[0], e[1])

        return rooted_tree

    def find_cherry(self, dissimilarity_matrix: np.array) -> tuple[int, int]:
        """Finds a pair of samples to join into a cherry.

        Finds the pair of samples with the minimum dissimilarity by finding the
        minimum value in the provided dissimilarity matrix

        Args:
            dissimilarity_matrix: A sample x sample dissimilarity matrix

        Returns
        -------
            A tuple of integers representing rows in the dissimilarity matrix
                to join.
        """
        dissimilarity_matrix = dissimilarity_matrix.astype(float)
        np.fill_diagonal(dissimilarity_matrix, np.inf)

        return np.unravel_index(
            np.argmin(dissimilarity_matrix, axis=None),
            dissimilarity_matrix.shape,
        )

    def update_dissimilarity_map(
        self,
        dissimilarity_map: pd.DataFrame,
        cherry: tuple[str, str],
        new_node: str,
    ) -> pd.DataFrame:
        """Update dissimilarity map after finding a cherry.

        Updates the dissimilarity map after joining together two nodes (m1, m2)
        at a cherry m. For all other nodes v, the new dissimilarity map d' is:

        d'(m, v) = (<m1> * d(m1, v) + <m2> * d(m2, v))/(<m1> + <m2>)

        where <m1> is the size of cluster m1, i.e. the number of sample leaves
        under node m1.

        Args:
            dissimilarity_map: A dissimilarity map to update
            cherry: A tuple of indices in the dissimilarity map that are joining
            new_node: New node name, to be added to the new dissimilarity map

        Returns
        -------
            A new dissimilarity map, updated with the new node
        """
        i_size, j_size = (
            max(1, self.__cluster_to_cluster_size[cherry[0]]),
            max(1, self.__cluster_to_cluster_size[cherry[1]]),
        )

        self.__cluster_to_cluster_size[new_node] = i_size + j_size

        i, j = (
            np.where(dissimilarity_map.index == cherry[0])[0][0],
            np.where(dissimilarity_map.index == cherry[1])[0][0],
        )

        dissimilarity_array = self.__update_dissimilarity_map_numba(
            dissimilarity_map.to_numpy(), i, j, i_size, j_size
        )
        sample_names = list(dissimilarity_map.index) + [new_node]

        dissimilarity_map = pd.DataFrame(
            dissimilarity_array, index=sample_names, columns=sample_names
        )

        # drop out cherry from dissimilarity map
        dissimilarity_map.drop(
            columns=[cherry[0], cherry[1]],
            index=[cherry[0], cherry[1]],
            inplace=True,
        )

        return dissimilarity_map

    @staticmethod
    @numba.jit(nopython=True)
    def __update_dissimilarity_map_numba(
        dissimilarity_map: np.array,
        cherry_i: int,
        cherry_j: int,
        size_i: int,
        size_j: int,
    ) -> np.array:
        """A private, optimized function for updating dissimilarities.

        A faster implementation of updating the dissimilarity map for UPGMA,
        invoked by `self.update_dissimilarity_map`.

        Args:
            dissimilarity_map: A matrix of dissimilarities to update
            cherry_i: Index of the first item in the cherry
            cherry_j: Index of the second item in the cherry

        Returns
        -------
            An updated dissimilarity map

        """
        # add new row & column for incoming sample
        N = dissimilarity_map.shape[1]

        new_row = np.array([0.0] * N)
        updated_map = np.vstack((dissimilarity_map, np.atleast_2d(new_row)))
        new_col = np.array([0.0] * (N + 1))
        updated_map = np.hstack((updated_map, np.atleast_2d(new_col).T))

        new_node_index = updated_map.shape[0] - 1
        for v in range(dissimilarity_map.shape[0]):
            if v == cherry_i or v == cherry_j:
                continue
            updated_map[v, new_node_index] = updated_map[new_node_index, v] = (
                size_i * dissimilarity_map[v, cherry_i]
                + size_j * dissimilarity_map[v, cherry_j]
            ) / (size_i + size_j)

        updated_map[new_node_index, new_node_index] = 0

        return updated_map

    def setup_root_finder(self, cassiopeia_tree: CassiopeiaTree) -> None:
        """Defines the implicit rooting strategy for the UPGMASolver.

        By default, the UPGMA algorithm returns an rooted tree. Therefore,
        the implicit root will be placed and specified at the end of the
        solving procedure as the parent of the last two unjoined nodes.

        Args:
            cassiopeia_tree: Input CassiopeiaTree to `solve`
        """
        cassiopeia_tree.root_sample_name = "root"
