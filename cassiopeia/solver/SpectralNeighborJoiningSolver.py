"""
This file stores a subclass of DistanceSolver, SpectralNeighborJoining. This algorithm is based on the one developed by Jaffe, et al. (2020) in their paper titled "Spectral neighbor joining for reconstruction of latent tree models. 
"""

from typing import Callable, Dict, List, Optional, Tuple, Union

import abc
import networkx as nx
import numpy as np
import pandas as pd
from itertools import chain

# Only for debugging purposes
if True:
    import sys
    sys.path.insert(0, '../..')

from cassiopeia.solver.DistanceSolver import DistanceSolver

from cassiopeia.data import CassiopeiaTree
from cassiopeia.solver import (
    dissimilarity_functions,
    solver_utilities,
)


class SpectralNeighborJoiningSolver(DistanceSolver):
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

    def root_tree(
        self, tree: nx.Graph, root_sample: str, remaining_samples: List[str]
    ) -> nx.DiGraph():
        """Roots a tree produced by Neighbor-Joining at the specified root.

        Uses the specified root to root the tree passed in

        Args:
            tree: Networkx object representing the tree topology
            root_sample: Sample to treat as the root
            remaining_samples: The last two unjoined nodes in the tree

        Returns:
            A rooted tree
        """
        tree.add_edge(remaining_samples[0], remaining_samples[1])

        rooted_tree = nx.DiGraph()
        for e in nx.dfs_edges(tree, source=root_sample):
            rooted_tree.add_edge(e[0], e[1])

        return rooted_tree

    def get_dissimilarity_map(
        self, 
        cassiopeia_tree: CassiopeiaTree,
        layer: Optional[str] = None
    ) -> pd.DataFrame: 
        """Outputs the first lambda matrix, where every subset is an individual node.

        Args:
            cassiopeia_tree: the CassiopeiaTree object passed into solve()
            layer: Layer storing the character matrix for solving. If None, the
                default character matrix is used in the CassiopeiaTree.

        Returns:
            DataFrame object of the lamba matrix, but similar in structure to DistanceSolver's dissimilarity_map.
        """

        # get similarity map and save it as private instance variable
        self.__similarity_map = super().get_dissimilarity_map(cassiopeia_tree, layer)

        # generate first (pairwise) lambda matrix
        N = self.__similarity_map.shape[0]
        self.lambda_indices: List[List[int]] = [[i] for i in range(N)]
        node_names: np.ndarray = self.__similarity_map.index.values

        lambda_matrix_arr = self.__compute_lambda(self.lambda_indices)

        lambda_matrix_df = pd.DataFrame(
            lambda_matrix_arr, 
            index=node_names,
            columns=node_names
        )        

        return lambda_matrix_df


    def find_cherry(self, dissimilarity_map: np.ndarray) -> Tuple[int, int]:
        """Finds a pair of samples to join into a cherry.

        With dissimilarity_map being the lambda matrix, this method finds the argmin pair of subsets of the lambda matrix. 

        Args:
            dissimilarity_matrix: Lambda matrix

        Returns:
            A tuple of intgers representing rows in the dissimilarity matrix
                to join.
        """

        return np.unravel_index(np.argmin(dissimilarity_map, axis=None), dissimilarity_map.shape) # type: ignore


    def __compute_lambda(self, lambda_indices: List[List[int]]) -> np.ndarray:
        """Computes the lambda matrix, whose entries consist of the second SVD of the R^A affinity matrix for every pair of nodes.

        Computes the lambda(i,j) = SVD # 2 for R^(Ai U Aj) where i and j are the susbet indices in lambda_indices.

        Args:
            dissimilarity_map: A sample x sample dissimilarity map

        Returns:
            A matrix storing the Q-criterion for every pair of samples.
        """

        n = len(lambda_indices)
        lambda_matrix_arr = np.zeros([n, n])

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
                RA_matrix = self.__similarity_map.values[np.ix_(a_subset, a_comp_flat)]

                # Calculate SVD2
                lambda_matrix_arr[i_count, j_count] = lambda_matrix_arr[j_count, i_count] = np.linalg.svd(RA_matrix)[1][1]

        np.fill_diagonal(lambda_matrix_arr, np.inf)

        return lambda_matrix_arr

    def update_dissimilarity_map(
        self,
        dissimilarity_map: pd.DataFrame, # lambda matrix
        cherry: Tuple[str, str],
        new_node: str,
        ) -> pd.DataFrame:
        """Updates the lambda matrix using the pair of subsets from find_cherry.

        Args:
            dissimilarity_map: lambda matrix to update
            cherry1: One of the children to join.
            cherry2: One of the children to join.
            new_node: New node name to add to the dissimilarity map

        Returns:
            An updated lambda matrix.
        """

        lambda_indices_copy = self.lambda_indices.copy()

        # get cherry nodes in index of lambda matrix
        i, j = (
            np.where(dissimilarity_map.index == cherry[0])[0][0],
            np.where(dissimilarity_map.index == cherry[1])[0][0],
        )

        # modify names
        node_names = dissimilarity_map.index.values
        node_names = np.append(node_names, new_node) # type: ignore
        node_names = np.delete(node_names, [i, j])

        # modify indices
        lambda_indices_copy.append([*lambda_indices_copy[i], *lambda_indices_copy[j]])
        lambda_indices_copy.pop(max(i, j))
        lambda_indices_copy.pop(min(i, j))

        if len(lambda_indices_copy) > 2:
            # compute new lambda matrix
            lambda_matrix_arr = self.__compute_lambda(lambda_indices_copy)
        else:
            n = len(lambda_indices_copy)
            lambda_matrix_arr = np.zeros((n, n))

        # regenerate lambda matrix
        lambda_matrix_df = pd.DataFrame(
            lambda_matrix_arr,
            index=node_names,
            columns=node_names
        )

        # update lambda indices
        self.lambda_indices = lambda_indices_copy
        

        return lambda_matrix_df

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
                "Please specify a dissimilarity function to add an implicit "
                "root, or specify an explicit root"
            )

        dissimilarity_map = cassiopeia_tree.get_dissimilarity_map()
        if dissimilarity_map is None:
            cassiopeia_tree.compute_dissimilarity_map(
                self.dissimilarity_function, self.prior_transformation
            )
        else:
            dissimilarity = {"root": 0}
            for leaf in character_matrix.index:
                weights = None
                if cassiopeia_tree.priors:
                    weights = solver_utilities.transform_priors(
                        cassiopeia_tree.priors, self.prior_transformation
                    )
                dissimilarity[leaf] = self.dissimilarity_function(
                    rooted_character_matrix.loc["root"].values,
                    rooted_character_matrix.loc[leaf].values,
                    cassiopeia_tree.missing_state_indicator,
                    weights,
                )
            cassiopeia_tree.set_dissimilarity("root", dissimilarity)

        cassiopeia_tree.character_matrix = character_matrix