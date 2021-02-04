"""
This file stores a subclass of DistanceSolver, NeighborJoining. The
inference procedure is the Neighbor-Joining algorithm proposed by Saitou and
Nei (1987) that iteratively joins together samples that minimize the Q-criterion
on the dissimilarity map.
"""
import abc
import networkx as nx
import numba
import numpy as np
import pandas as pd
from typing import Callable, Dict, Optional, Tuple, Union

from cassiopeia.solver import DistanceSolver


class NeighborJoiningSolver(DistanceSolver.DistanceSolver):
    """
    Implements the Neighbor-Joining algorithm described by Saitou and Nei (1987)
    as a derived class of DistanceSolver. This class inherits the generic
    `solve` method, but implements its own procedure for finding cherries by
    minimizing the Q-criterion between samples.

    Args:
        character_matrix: A character matrix of observed character states for
            all samples.
        meta_data: Any meta data associated with the samples
        priors: Prior probabilities of observing a transition from 0 to any
            character state
        prior_transformation: A function defining a transformation on the priors
            in forming weights
        dissimilarity_map: A dissimilarity map describing the distances between
            samples.
        dissimilarity_function: A function by which to compute the dissimilarity
            map. Optional if a dissimilarity map is already provided.
        root_sample: A sample in the character matrix to treat as the root. If
            not provided, a root of (0,...,0) is added to the character matrix.
            Throws an error is a root_sample is not provided and no dissimilarity
            function is provided.

    Attributes:
        character_matrix: The character matrix describing the samples
        meta_data: Data table storing meta data for each sample
        priors: Prior probabilities of character state transitions
        weights: Weights on character/mutation pairs, derived from priors
        dissimilarity_map: Dissimilarity map describing distances between
            samples
        dissimilarity_function: Function to compute the dissimilarity between
            samples.
        root_sample: Sample to treat as a root, an index in the dissimilarity
            map and character matrix.
        tree: The tree returned by `self.solve()`. None if `solve` has not been
            called yet.

    """

    def __init__(
        self,
        character_matrix: pd.DataFrame,
        missing_char: int = -1,
        meta_data: Optional[pd.DataFrame] = None,
        priors: Optional[Dict[int, str]] = None,
        prior_transformation: Optional[
            Callable[[float], float]
        ] = "negative_log",
        dissimilarity_map: Optional[pd.DataFrame] = None,
        dissimilarity_function: Optional[
            Callable[
                [
                    int,
                    int,
                    pd.DataFrame,
                    int,
                    Optional[Dict[int, Dict[str, float]]],
                ],
                float,
            ]
        ] = None,
        root_sample: Optional[str] = None,
    ):

        if dissimilarity_function is None and root_sample is None:
            raise DistanceSolver.DistanceSolveError(
                "Please specify a root sample or provide a dissimilarity "
                "function by which to add a root to the dissimilarity map"
            )

        if not root_sample:

            root = [0] * character_matrix.shape[1]
            character_matrix.loc["root"] = root
            root_sample = "root"

            # if root sample is not specified, we'll add the implicit root
            # and recompute the dissimilarity map
            dissimilarity_map = None

        self.root_sample = root_sample

        super().__init__(
            character_matrix,
            missing_char,
            meta_data,
            priors,
            prior_transformation,
            dissimilarity_map=dissimilarity_map,
            dissimilarity_function=dissimilarity_function,
        )

    def find_cherry(self, dissimilarity_matrix: np.array) -> Tuple[int, int]:
        """Finds a pair of samples to join into a cherry.

        Proceeds by minimizing the Q-criterion as in Saitou and Nei (1987) to
        select a pair of samples to join.

        Args:
            dissimilarity_matrix: A sample x sample dissimilarity matrix

        Returns:
            A tuple of intgers representing rows in the dissimilarity matrix
                to join.
        """

        q = self.compute_q(dissimilarity_matrix)
        np.fill_diagonal(q, np.inf)

        _min = np.argmin(q)

        i, j = _min % q.shape[0], _min // q.shape[0]

        return (i, j)

    def root_tree(self, tree):
        """Roots a tree at the inferred ancestral root.

        Uses the root sample stored in self.root_sample to root the
        tree stored in the class instance.
        """

        rooted_tree = nx.DiGraph()

        for e in nx.dfs_edges(tree, source=self.root_sample):

            rooted_tree.add_edge(e[0], e[1])

        return rooted_tree

    @staticmethod
    @numba.jit(nopython=True)
    def compute_q(dissimilarity_map: np.array(int)):
        """Computes the Q-criterion for every pair of samples.

        Computes the Q-criterion defined by Saitou and Nei (1987):

            Q(i,j) = d(i, j) - 1/(n-2) (sum(d(i, :)) + sum(d(j,:)))

        Args:
            dissimilarity_map: A sample x sample dissimilarity map

        Returns:
            A matrix storing the Q-criterion for every pair of samples.
        """

        q = np.zeros(dissimilarity_map.shape)
        n = dissimilarity_map.shape[0]
        for i in range(n):
            for j in range(i):
                q[i, j] = q[j, i] = (dissimilarity_map[i, j]) - (
                    1
                    / (n - 2)
                    * (
                        dissimilarity_map[i, :].sum()
                        + dissimilarity_map[j, :].sum()
                    )
                )
        return q

    def update_dissimilarity_map(
        self,
        dissimilarity_map: pd.DataFrame,
        cherry: Tuple[str, str],
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

        Returns:
            A new dissimilarity map, updated with the new node
        """

        i, j = (
            np.where(dissimilarity_map.index == cherry[0])[0][0],
            np.where(dissimilarity_map.index == cherry[1])[0][0],
        )

        dissimilarity_array = self.update_dissimilarity_map_numba(
            dissimilarity_map.to_numpy(), i, j
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
    def update_dissimilarity_map_numba(
        dissimilarity_map: np.array,
        cherry_i: int,
        cherry_j: int,
    ) -> np.array:
        """An optimized function for updating dissimilarities.

        A faster implementation of updating the dissimilarity map for Neighbor
        Joining, invoked by `self.update_dissimilarity_map`.

        Args:
            dissimilarity_map: A matrix of dissimilarities to update
            cherry_i: Index of the first item in the cherry
            cherry_j: Index of the second item in the cherry


        Returns:
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
            updated_map[v, new_node_index] = updated_map[
                new_node_index, v
            ] = 0.5 * (
                dissimilarity_map[v, cherry_i]
                + dissimilarity_map[v, cherry_j]
                - dissimilarity_map[cherry_i, cherry_j]
            )

        updated_map[new_node_index, new_node_index] = 0

        return updated_map
