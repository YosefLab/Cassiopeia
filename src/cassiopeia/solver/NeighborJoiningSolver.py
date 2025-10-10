"""Module defining a subclass of DistanceSolver, NeighborJoiningSolver.

The inference procedure is the Neighbor-Joining algorithm proposed by Saitou and
Nei (1987) that iteratively joins together samples that minimize the Q-criterion
on the dissimilarity map.
"""

from collections.abc import Callable

import networkx as nx
import numba
import numpy as np
import pandas as pd

from cassiopeia.data import CassiopeiaTree
from cassiopeia.mixins import DistanceSolverError
from cassiopeia.solver import (
    DistanceSolver,
    dissimilarity_functions,
    solver_utilities,
)


class NeighborJoiningSolver(DistanceSolver.DistanceSolver):
    """Neighbor-Joining class for Cassiopeia.

    Implements the Neighbor-Joining algorithm described by Saitou and Nei (1987)
    as a derived class of DistanceSolver. This class inherits the generic
    `solve` method, but implements its own procedure for finding cherries by
    minimizing the Q-criterion between samples. If fast is set to True,
    a fast NJ implementation of is used.

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
        fast: Whether to use a fast implementation of Neighbor-Joining.
        implementation: Which fast implementation to use. Options are:
            "ccphylo_dnj": CCPhylo implementation the Dynamic Neighbor-Joining
                algorithm described by Clausen (2023). Solution in guaranteed
                to be exact.
            "ccphylo_hnj": CCPhylo implementation of the Heuristic
                Neighbor-Joining algorithm described by Clausen (2023).
                Solution is not guaranteed to be exact.
            "ccphylo_nj": CCPhylo implementation of the Neighbor-Joining.
        threads: Number of threads to use for solver.

    Attributes
    ----------
        dissimilarity_function: Function used to compute dissimilarity between
            samples.
        add_root: Whether or not to add an implicit root the tree.
        prior_transformation: Function to use when transforming priors into
            weights.
        threads: Number of threads to use for solver.

    """

    def __init__(
        self,
        dissimilarity_function: Callable[[np.array, np.array, int, dict[int, dict[int, float]]], float]
        | None = dissimilarity_functions.weighted_hamming_distance,
        add_root: bool = False,
        prior_transformation: str = "negative_log",
        fast: bool = False,
        implementation: str = "ccphylo_dnj",
        threads: int = 1,
    ):
        if fast:
            if implementation in ["ccphylo_dnj", "ccphylo_hnj", "ccphylo_nj"]:
                self._implementation = implementation
            else:
                raise DistanceSolverError(
                    "Invalid fast implementation of Neighbor-Joining. Options "
                    "are: 'ccphylo_dnj', 'ccphylo_hnj', 'ccphylo_nj'"
                )
        else:
            self._implementation = "generic_nj"

        super().__init__(
            dissimilarity_function=dissimilarity_function,
            add_root=add_root,
            prior_transformation=prior_transformation,
            threads=threads,
        )

    def root_tree(self, tree: nx.Graph, root_sample: str, remaining_samples: list[str]) -> nx.DiGraph():
        """Roots a tree produced by Neighbor-Joining at the specified root.

        Uses the specified root to root the tree passed in

        Args:
            tree: Networkx object representing the tree topology
            root_sample: Sample to treat as the root
            remaining_samples: The last two unjoined nodes in the tree

        Returns
        -------
            A rooted tree
        """
        tree.add_edge(remaining_samples[0], remaining_samples[1])

        rooted_tree = nx.DiGraph()
        for e in nx.dfs_edges(tree, source=root_sample):
            rooted_tree.add_edge(e[0], e[1])

        return rooted_tree

    def find_cherry(self, dissimilarity_matrix: np.array) -> tuple[int, int]:
        """Finds a pair of samples to join into a cherry.

        Proceeds by minimizing the Q-criterion as in Saitou and Nei (1987) to
        select a pair of samples to join.

        Args:
            dissimilarity_matrix: A sample x sample dissimilarity matrix

        Returns
        -------
            A tuple of intgers representing rows in the dissimilarity matrix
                to join.
        """
        q = self.compute_q(dissimilarity_matrix)
        np.fill_diagonal(q, np.inf)

        return np.unravel_index(np.argmin(q, axis=None), q.shape)

    @staticmethod
    @numba.jit(nopython=True)
    def compute_q(dissimilarity_map: np.array(int)) -> np.array:
        """Computes the Q-criterion for every pair of samples.

        Computes the Q-criterion defined by Saitou and Nei (1987):

            Q(i,j) = d(i, j) - 1/(n-2) (sum(d(i, :)) + sum(d(j,:)))

        Args:
            dissimilarity_map: A sample x sample dissimilarity map

        Returns
        -------
            A matrix storing the Q-criterion for every pair of samples.
        """
        q = np.zeros(dissimilarity_map.shape)
        n = dissimilarity_map.shape[0]
        dissimilarity_map_rowsums = dissimilarity_map.sum(axis=1)
        for i in range(n):
            for j in range(i):
                q[i, j] = q[j, i] = (dissimilarity_map[i, j]) - (
                    1 / (n - 2) * (dissimilarity_map_rowsums[i] + dissimilarity_map_rowsums[j])
                )
        return q

    def update_dissimilarity_map(
        self,
        dissimilarity_map: pd.DataFrame,
        cherry: tuple[str, str],
        new_node: str,
    ) -> pd.DataFrame:
        """Update dissimilarity map after finding a cherry.

        Updates the dissimilarity map after joining together two nodes (m1, m2)
        at a cherry m. For all nodes v, the new dissimilarity map d' is:

        d'(m, v) = 0.5 * (d(v, m1) + d(v, m2) - d(m1, m2))

        Args:
            dissimilarity_map: A dissimilarity map to update
            cherry: A tuple of indices in the dissimilarity map that are joining
            new_node: New node name, to be added to the new dissimilarity map

        Returns
        -------
            A new dissimilarity map, updated with the new node
        """
        i, j = (
            np.where(dissimilarity_map.index == cherry[0])[0][0],
            np.where(dissimilarity_map.index == cherry[1])[0][0],
        )

        dissimilarity_array = self.__update_dissimilarity_map_numba(dissimilarity_map.to_numpy(), i, j)
        sample_names = list(dissimilarity_map.index) + [new_node]

        dissimilarity_map = pd.DataFrame(dissimilarity_array, index=sample_names, columns=sample_names)

        # drop out cherry from dissimilarity map
        dissimilarity_map.drop(
            columns=[cherry[0], cherry[1]],
            index=[cherry[0], cherry[1]],
            inplace=True,
        )

        return dissimilarity_map

    @staticmethod
    @numba.jit(nopython=True)
    def __update_dissimilarity_map_numba(dissimilarity_map: np.array, cherry_i: int, cherry_j: int) -> np.array:
        """A private, optimized function for updating dissimilarities.

        A faster implementation of updating the dissimilarity map for Neighbor
        Joining, invoked by `self.update_dissimilarity_map`.

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
            updated_map[v, new_node_index] = updated_map[new_node_index, v] = 0.5 * (
                dissimilarity_map[v, cherry_i] + dissimilarity_map[v, cherry_j] - dissimilarity_map[cherry_i, cherry_j]
            )

        updated_map[new_node_index, new_node_index] = 0

        return updated_map

    def setup_root_finder(self, cassiopeia_tree: CassiopeiaTree) -> None:
        """Defines the implicit rooting strategy for the NeighborJoiningSolver.

        By default, the NeighborJoining algorithm returns an unrooted tree.
        To root this tree, an implicit root of all zeros is added to the
        character matrix. Then, the dissimilarity map is recalculated using
        the updated character matrix. If the tree already has a computed
        dissimilarity map, only the new dissimilarities are calculated.

        Args:
            cassiopeia_tree: Input CassiopeiaTree to `solve`
        """
        character_matrix = cassiopeia_tree.character_matrix.copy()
        rooted_character_matrix = character_matrix.copy()

        root = [0] * rooted_character_matrix.shape[1]
        rooted_character_matrix.loc["root"] = root
        cassiopeia_tree.root_sample_name = "root"
        cassiopeia_tree.character_matrix = rooted_character_matrix

        if self.dissimilarity_function is None:
            raise DistanceSolver.DistanceSolverError(
                "Please specify a dissimilarity function to add an implicit root, or specify an explicit root"
            )

        dissimilarity_map = cassiopeia_tree.get_dissimilarity_map()
        if dissimilarity_map is None:
            cassiopeia_tree.compute_dissimilarity_map(
                self.dissimilarity_function, self.prior_transformation, threads=self.threads
            )
        else:
            dissimilarity = {"root": 0}
            for leaf in character_matrix.index:
                weights = None
                if cassiopeia_tree.priors:
                    weights = solver_utilities.transform_priors(cassiopeia_tree.priors, self.prior_transformation)
                dissimilarity[leaf] = self.dissimilarity_function(
                    rooted_character_matrix.loc["root"].values,
                    rooted_character_matrix.loc[leaf].values,
                    cassiopeia_tree.missing_state_indicator,
                    weights,
                )
            cassiopeia_tree.set_dissimilarity("root", dissimilarity)

        cassiopeia_tree.character_matrix = character_matrix
