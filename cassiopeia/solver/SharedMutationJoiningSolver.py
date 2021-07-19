"""
This file stores the SharedMutationJoiningSolver. The inference procedure is
an agglomerative clustering procedure that joins samples that share the most
identical character/state mutations.
"""
import abc
import warnings
from typing import Callable, Dict, List, Optional, Tuple, Union

import networkx as nx
import numba
import numpy as np
import pandas as pd
import scipy

from cassiopeia.data import CassiopeiaTree
from cassiopeia.data import utilities as data_utilities
from cassiopeia.mixins import (
    SharedMutationJoiningSolverError,
    SharedMutationJoiningSolverWarning,
)
from cassiopeia.solver import (
    CassiopeiaSolver,
    dissimilarity_functions,
    solver_utilities,
)


class SharedMutationJoiningSolver(CassiopeiaSolver.CassiopeiaSolver):
    """Shared-Mutation-Joining class for Cassiopeia.

    Implements an iterative, bottom-up agglomerative clustering procedure. The
    algorithm iteratively clusters the samples in the sample pool by the number
    of shared mutations that they have in their character information. The
    algorithm has theoretical guarantees on correctness given a sufficiently
    large number of characters and bounds on edge lengths in the tree generative
    process.

    TODO(mgjones, rzhang): Make the solver work with similarity maps as
        flattened arrays

    Args:
        similarity_function: Function that can be used to compute the
            similarity between samples.
        prior_transformation: Function to use when transforming priors into
            weights. Supports the following transformations:
                "negative_log": Transforms each probability by the negative
                    log (default)
                "inverse": Transforms each probability p by taking 1/p
                "square_root_inverse": Transforms each probability by the
                    the square root of 1/p

    Attributes:
        similarity_function: Function used to compute similarity between
            samples.
        prior_transformation: Function to use when transforming priors into
            weights.
    """

    def __init__(
        self,
        similarity_function: Optional[
            Callable[
                [
                    np.array,
                    np.array,
                    int,
                    Optional[Dict[int, Dict[int, float]]],
                ],
                float,
            ]
        ] = dissimilarity_functions.hamming_similarity_without_missing,
        prior_transformation: str = "negative_log",
    ):

        super().__init__(prior_transformation)

        # Attempt to numbaize
        self.similarity_function = similarity_function
        self.nb_similarity_function = similarity_function
        numbaize = True
        try:
            self.nb_similarity_function = numba.jit(
                similarity_function, nopython=True
            )
        except TypeError:
            numbaize = False
            warnings.warn(
                "Failed to numbaize dissimilarity function. "
                "Falling back to Python.",
                SharedMutationJoiningSolverWarning,
            )

        if numbaize:
            self.__update_similarity_map = numba.jit(
                self.__update_similarity_map, nopython=True
            )

    def solve(
        self, cassiopeia_tree: CassiopeiaTree, layer: Optional[str] = None
    ) -> None:
        """Solves a tree for the SharedMutationJoiningSolver.

        The solver routine calculates an n x n similarity matrix of all
        pairwise sample similarities based on a provided similarity function on
        the character vectors. The general solver routine proceeds by
        iteratively finding pairs of samples to join together into a "cherry"
        until all samples are joined. At each iterative step, the two samples
        with the most shared character/state mutations are joined. Then, an LCA
        node with a character vector containing only the mutations shared by the
        joined samples is added to the sample pool, and the similarity matrix is
        updated with respect to the new LCA node. The function will update the
        `tree` attribute of the input CassiopeiaTree.

        Args:
            cassiopeia_tree: CassiopeiaTree object to be populated
            layer: Layer storing the character matrix for solving. If None, the
                default character matrix is used in the CassiopeiaTree.
        """

        node_name_generator = solver_utilities.node_name_generator()

        if layer:
            character_matrix = cassiopeia_tree.layers[layer].copy()
        else:
            character_matrix = cassiopeia_tree.character_matrix.copy()

        weights = None
        if cassiopeia_tree.priors:
            weights = solver_utilities.transform_priors(
                cassiopeia_tree.priors, self.prior_transformation
            )

        similarity_map = data_utilities.compute_dissimilarity_map(
            character_matrix.to_numpy(),
            character_matrix.shape[0],
            self.similarity_function,
            weights,
            cassiopeia_tree.missing_state_indicator,
        )

        similarity_map = scipy.spatial.distance.squareform(similarity_map)

        similarity_map = pd.DataFrame(
            similarity_map,
            index=character_matrix.index,
            columns=character_matrix.index,
        )

        N = similarity_map.shape[0]

        # Numba-ize the similarity function and weights
        nb_weights = numba.typed.Dict.empty(
            numba.types.int64,
            numba.types.DictType(numba.types.int64, numba.types.float64),
        )
        if weights:
            for k, v in weights.items():
                nb_char_weights = numba.typed.Dict.empty(
                    numba.types.int64, numba.types.float64
                )
                for state, prior in v.items():
                    nb_char_weights[state] = prior
                nb_weights[k] = nb_char_weights

        # instantiate a tree where all samples appear as leaves.
        tree = nx.DiGraph()
        tree.add_nodes_from(similarity_map.index)

        while N > 1:

            i, j = self.find_cherry(similarity_map.values)

            # get indices in the similarity matrix to join
            node_i, node_j = (similarity_map.index[i], similarity_map.index[j])

            new_node_name = next(node_name_generator)
            tree.add_node(new_node_name)
            tree.add_edges_from(
                [(new_node_name, node_i), (new_node_name, node_j)]
            )

            similarity_map = self.update_similarity_map_and_character_matrix(
                character_matrix,
                self.nb_similarity_function,
                similarity_map,
                (node_i, node_j),
                new_node_name,
                cassiopeia_tree.missing_state_indicator,
                nb_weights,
            )

            N = similarity_map.shape[0]

        cassiopeia_tree.populate_tree(tree)

    def find_cherry(self, similarity_matrix: np.array) -> Tuple[int, int]:
        """Finds a pair of samples to join into a cherry.

        Finds the pair of samples with the highest pairwise similarity to join.

        Args:
            similarity_matrix: A sample x sample similarity matrix

        Returns:
            A tuple of integers representing rows in the similarity matrix
            to join.
        """
        similarity_matrix = similarity_matrix.astype(float)
        np.fill_diagonal(similarity_matrix, -np.inf)

        return np.unravel_index(
            np.argmax(similarity_matrix, axis=None), similarity_matrix.shape
        )

    def update_similarity_map_and_character_matrix(
        self,
        character_matrix: pd.DataFrame,
        similarity_function: Callable[
            [np.array, np.array, int, Dict[int, Dict[int, float]]], float
        ],
        similarity_map: pd.DataFrame,
        cherry: Tuple[str, str],
        new_node: str,
        missing_state_indicator: int = -1,
        weights=None,
    ) -> pd.DataFrame:
        """Update similarity map after finding a cherry.

        Adds the new LCA node into the character matrix with the mutations
        shared by the joined nodes as its character vector. Then, updates the
        similarity matrix by calculating the pairwise similarity between the
        new LCA node and all existing nodes.

        Args:
            character_matrix: Contains the character information for all nodes,
                updated as nodes are joined and new internal LCA nodes are added
            similarity_function: A similarity function
            similarity_map: A similarity map to update
            cherry: A tuple of indices in the similarity map that are joining
            new_node: New node name, to be added to the updated similarity map
            missing_state_indicator: Character representing missing data
            weights: Weighting of each (character, state) pair. Typically a
                transformation of the priors.

        Returns:
            A new similarity map, updated with the new node
        """

        character_i, character_j = (
            np.where(character_matrix.index == cherry[0])[0][0],
            np.where(character_matrix.index == cherry[1])[0][0],
        )

        character_array = character_matrix.to_numpy(copy=True)
        similarity_array = similarity_map.to_numpy()
        i_characters = character_array[character_i, :]
        j_characters = character_array[character_j, :]
        lca = data_utilities.get_lca_characters(
            [i_characters, j_characters], missing_state_indicator
        )
        character_matrix.loc[new_node] = lca

        similarity_array_updated = self.__update_similarity_map(
            character_array,
            similarity_array,
            np.array(lca),
            similarity_function,
            missing_state_indicator,
            weights,
        )

        sample_names = list(similarity_map.index) + [new_node]

        similarity_map = pd.DataFrame(
            similarity_array_updated, index=sample_names, columns=sample_names
        )

        # drop out cherry from similarity map and character matrix
        similarity_map.drop(
            columns=[cherry[0], cherry[1]],
            index=[cherry[0], cherry[1]],
            inplace=True,
        )

        character_matrix.drop(index=[cherry[0], cherry[1]], inplace=True)

        return similarity_map

    @staticmethod
    def __update_similarity_map(
        character_matrix: np.array,
        similarity_map: np.array,
        lca: np.array,
        similarity_function: Callable[
            [np.array, np.array, int, Dict[int, Dict[int, float]]], float
        ],
        missing_state_indicator: int = -1,
        weights=None,
    ) -> np.array:
        """A private, optimized function for updating similarities.

        A faster implementation of updating the similarity map for the
        SharedMutationJoiner, invoked by
        `self.update_similarity_map_and_character_matrix`.

        Args:
            character_matrix: The character information for all nodes
            similarity_map: A matrix of similarities to update
            lca: The character vector of the new LCA node
            similarity_function: A similarity function
            missing_state_indicator: Character representing missing data
            weights: Weighting of each (character, state) pair. Typically a
                transformation of the priors.

        Returns:
            An updated similarity map
        """

        C = similarity_map.shape[0]
        new_row = np.zeros(C, dtype=np.float64)
        k = 0
        for i in range(C):
            s1 = character_matrix[i, :]
            new_row[k] = similarity_function(
                s1, lca, missing_state_indicator, weights
            )
            k += 1

        updated_map = np.vstack((similarity_map, np.atleast_2d(new_row)))
        new_col = np.append(new_row, np.array([0]))
        updated_map = np.hstack((updated_map, np.atleast_2d(new_col).T))
        return updated_map
