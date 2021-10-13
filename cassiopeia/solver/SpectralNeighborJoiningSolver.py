"""
This file stores a subclass of DistanceSolver, SpectralNeighborJoining. This algorithm is based on the one developed by Jaffe, et al. (2020) in their paper titled "Spectral neighbor joining for reconstruction of latent tree models. 
"""

from typing import Callable, Dict, List, Optional, Tuple, Union

import abc
import networkx as nx
import numpy as np
import pandas as pd
from itertools import chain

from cassiopeia.data import CassiopeiaTree
from cassiopeia.solver import (
    NeighborJoiningSolver,
    dissimilarity_functions,
    solver_utilities,
)


class SpectralNeighborJoiningSolver(NeighborJoiningSolver):
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
                [np.ndarray, np.ndarray, int, Dict[int, Dict[int, float]]], float
            ]
        ] = dissimilarity_functions.weighted_hamming_distance,
        add_root: bool = False,
        prior_transformation: str = "negative_log",
    ):
        super().__init__(dissimilarity_function, add_root=add_root, prior_transformation=prior_transformation) #type: ignore


    def find_cherry(self, dissimilarity_matrix: np.ndarray, lambda_indices: List[List[int]] = None) -> Tuple[int, int]:
        """Finds a pair of samples to join into a cherry.

        Proceeds by minimizing the second largest SVD of the RA Matrix of some subset of nodes.

        Args:
            dissimilarity_matrix: A sample x sample dissimilarity matrix

        Returns:
            A tuple of intgers representing rows in the dissimilarity matrix
                to join.
        """
        # temp? to pass tests
        if not lambda_indices:
            lambda_indices = [[i] for i in range(dissimilarity_matrix.shape[0])]

        lambda_matrix = self.__compute_lambda(dissimilarity_matrix, lambda_indices)
        np.fill_diagonal(lambda_matrix, np.inf)

        return np.unravel_index(np.argmin(lambda_matrix, axis=None), lambda_matrix.shape) # TODO: cache with heapq?


    def __compute_lambda(self, dissimilarity_map: np.ndarray, lambda_indices: List[List[int]]) -> np.ndarray:
        """Computes the lambda matrix, whose entries consist of the second SVD of the R^A affinity matrix for every pair of nodes.

        Computes the lambda(i,j) = SVD # 2 for R^(Ai U Aj) where i and j are the susbet indices in lambda_indices.

        Args:
            dissimilarity_map: A sample x sample dissimilarity map

        Returns:
            A matrix storing the Q-criterion for every pair of samples.
        """

        n = len(lambda_indices)
        lambda_matrix = np.zeros([n, n])

        for i_count, i in enumerate(lambda_indices):
            for j_count, j in enumerate(lambda_indices):
                if j_count >= i_count:
                    break
                
                a_subset = [*i, *j]

                # get complement
                a_comp = lambda_indices.copy()
                a_comp.pop(max(i_count, j_count))
                a_comp.pop(min(i_count, j_count))
                a_comp_flat = list(chain.from_iterable(a_comp))

                # create RA matrix
                RA_matrix = dissimilarity_map[np.ix_(a_subset, a_comp_flat)]

                # Calculate SVD2
                lambda_matrix[i_count, j_count] = lambda_matrix[j_count, i_count] = np.linalg.svd(RA_matrix)[1][1]

        return lambda_matrix

    def __update_lambda_indices(
        self,
        lambda_indices: List[List[int]],
        node_names: np.ndarray,
        cherry: Tuple[str, str],
        new_node: str,
        ) -> Tuple[List[List[int]], np.ndarray]:

        lambda_indices_copy = lambda_indices.copy()

        i, j = (
            np.where(node_names == cherry[0])[0][0], # type: ignore
            np.where(node_names == cherry[1])[0][0], # type: ignore
        )

        # modify names
        node_names = np.append(node_names, new_node) # type: ignore
        node_names = np.delete(node_names, [i, j])

        # modify indices
        lambda_indices_copy.append([*lambda_indices_copy[i], *lambda_indices_copy[j]])
        lambda_indices_copy.pop(max(i, j))
        lambda_indices_copy.pop(min(i, j))

        return (lambda_indices_copy, node_names)


    # TODO: document the change: just replace update dissimilarity map into update lambda_index matrix
    def solve(
        self,
        cassiopeia_tree: CassiopeiaTree,
        layer: Optional[str] = None,
        collapse_mutationless_edges: bool = False,
        logfile: str = "stdout.log",
    ) -> None:
        """Solves a tree for a general bottom-up distance-based solver routine.

        The general solver routine proceeds by iteratively finding pairs of
        samples to join together into a "cherry" and then reform the
        dissimilarity matrix with respect to this new cherry. The implementation
        of how to find cherries and update the dissimilarity map is left to
        subclasses of DistanceSolver. The function will update the `tree`
        attribute of the input CassiopeiaTree.

        Args:
            cassiopeia_tree: CassiopeiaTree object to be populated
            layer: Layer storing the character matrix for solving. If None, the
                default character matrix is used in the CassiopeiaTree.
            collapse_mutationless_edges: Indicates if the final reconstructed
                tree should collapse mutationless edges based on internal states
                inferred by Camin-Sokal parsimony. In scoring accuracy, this
                removes artifacts caused by arbitrarily resolving polytomies.
            logfile: File location to log output. Not currently used.
        """
        node_name_generator = solver_utilities.node_name_generator()

        self.setup_dissimilarity_map(cassiopeia_tree, layer)

        dissimilarity_map = cassiopeia_tree.get_dissimilarity_map()

        N = dissimilarity_map.shape[0]

        # instantiate a dissimilarity map that can be updated as we join
        # together nodes.
        _dissimilarity_map = dissimilarity_map.copy()

        # instantiate a tree where all samples appear as leaves.
        tree = nx.Graph()
        tree.add_nodes_from(_dissimilarity_map.index)

        # Construct Lambda Indices matrix
        lambda_indices: List[List[int]] = [[i] for i in range(N)]
        node_names: np.ndarray = _dissimilarity_map.index.values

        while N > 2:
            print(f'# Running with N={N}')
            i, j = self.find_cherry(_dissimilarity_map.to_numpy(), lambda_indices)

            # get node names to join
            node_i, node_j = (
                node_names[i],
                node_names[j],
            ) 

            new_node_name = next(node_name_generator)
            tree.add_node(new_node_name)
            tree.add_edges_from([
                (new_node_name, node_i), 
                (new_node_name, node_j)
            ])

            # Changed for SNJ
            lambda_indices, node_names = self.__update_lambda_indices(
                    lambda_indices, 
                    node_names,
                    (node_i, node_j), 
                    new_node_name
                )

            N = len(lambda_indices)

        tree = self.root_tree(
            tree,
            cassiopeia_tree.root_sample_name,
            node_names,
        )

        # remove root from character matrix before populating tree
        if (
            cassiopeia_tree.root_sample_name
            in cassiopeia_tree.character_matrix.index
        ):
            cassiopeia_tree.character_matrix = (
                cassiopeia_tree.character_matrix.drop(
                    index=cassiopeia_tree.root_sample_name
                )
            )

        cassiopeia_tree.populate_tree(tree, layer=layer)
        cassiopeia_tree.collapse_unifurcations()

        # collapse mutationless edges
        if collapse_mutationless_edges:
            cassiopeia_tree.collapse_mutationless_edges(
                infer_ancestral_characters=True
            )

