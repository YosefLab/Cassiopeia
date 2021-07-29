"""
This file stores a subclass of CassiopeiaSolver, the GreedySolver. This class
represents the structure of top-down algorithms that build the reconstructed
tree by recursively splitting the set of samples based on some split criterion.
"""
from typing import Callable, Dict, Generator, List, Optional, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd

from cassiopeia.data import CassiopeiaTree
from cassiopeia.mixins import GreedySolverError, is_ambiguous_state
from cassiopeia.solver import CassiopeiaSolver, solver_utilities


class GreedySolver(CassiopeiaSolver.CassiopeiaSolver):
    """
    A Greedy Cassiopeia solver.
    
    GreedySolver is an abstract class representing the structure of top-down
    inference algorithms. The solver procedure contains logic to build a tree
    from the root by recursively partitioning the set of samples. Each subclass
    will implement "perform_split", which is the procedure for successively
    partioning the sample set.

    Args:
        prior_transformation: Function to use when transforming priors into
            weights. Supports the following transformations:
                "negative_log": Transforms each probability by the negative
                    log (default)
                "inverse": Transforms each probability p by taking 1/p
                "square_root_inverse": Transforms each probability by the
                    the square root of 1/p

    Attributes:
        prior_transformation: Function to use when transforming priors into
            weights.
    """

    def __init__(self, prior_transformation: str = "negative_log"):

        super().__init__(prior_transformation)

    def perform_split(
        self,
        character_matrix: pd.DataFrame,
        samples: List[int],
        weights: Optional[Dict[int, Dict[int, float]]] = None,
        missing_state_indicator: int = -1,
    ) -> Tuple[List[Union[int, str]], List[Union[int, str]]]:
        """Performs a partition of the samples.

        Args:
            character_matrix: Character matrix
            samples: A list of samples to partition
            weights: Weighting of each (character, state) pair. Typically a
                transformation of the priors.
            missing_state_indicator: Character representing missing data.

        Returns:
            A tuple of lists, representing the left and right partition groups
        """
        pass

    def solve(
        self,
        cassiopeia_tree: CassiopeiaTree,
        layer: Optional[str] = None,
        collapse_mutationless_edges: bool = False,
        logfile: str = "stdout.log",
    ):
        """Implements a top-down greedy solving procedure.

        The procedure recursively splits a set of samples to build a tree. At
        each partition of the samples, an ancestral node is created and each
        side of the partition is placed as a daughter clade of that node. This
        continues until each side of the partition is comprised only of single
        samples. If an algorithm cannot produce a split on a set of samples,
        then those samples are placed as sister nodes and the procedure
        terminates, generating a polytomy in the tree. This function will
        populate a tree inside the input CassiopeiaTree.

        Args:
            cassiopeia_tree: CassiopeiaTree storing a character matrix and
                priors.
            layer: Layer storing the character matrix for solving. If None, the
                default character matrix is used in the CassiopeiaTree.
            collapse_mutationless_edges: Indicates if the final reconstructed
                tree should collapse mutationless edges based on internal states
                inferred by Camin-Sokal parsimony. In scoring accuracy, this
                removes artifacts caused by arbitrarily resolving polytomies.
            logfile: File location to log output.
        """

        # A helper function that builds the subtree given a set of samples
        def _solve(
            samples: List[Union[str, int]],
            tree: nx.DiGraph,
            unique_character_matrix: pd.DataFrame,
            weights: Dict[int, Dict[int, float]],
            missing_state_indicator: int,
        ):
            if len(samples) == 1:
                return samples[0]
            # Finds the best partition of the set given the split criteria
            clades = list(
                self.perform_split(
                    unique_character_matrix,
                    samples,
                    weights,
                    missing_state_indicator,
                )
            )
            # Generates a root for this subtree with a unique int identifier
            root = next(node_name_generator)
            tree.add_node(root)

            for clade in clades:
                if len(clade) == 0:
                    clades.remove(clade)

            # If unable to return a split, generate a polytomy and return
            if len(clades) == 1:
                for clade in clades[0]:
                    tree.add_edge(root, clade)
                return root
            # Recursively generate the subtrees for each daughter clade
            for clade in clades:
                child = _solve(
                    clade,
                    tree,
                    unique_character_matrix,
                    weights,
                    missing_state_indicator,
                )
                tree.add_edge(root, child)
            return root

        node_name_generator = solver_utilities.node_name_generator()

        weights = None
        if cassiopeia_tree.priors:
            weights = solver_utilities.transform_priors(
                cassiopeia_tree.priors, self.prior_transformation
            )

        # extract character matrix
        if layer:
            character_matrix = cassiopeia_tree.layers[layer].copy()
        else:
            character_matrix = cassiopeia_tree.character_matrix.copy()

        # Raise exception if the character matrix has ambiguous states.
        if any(
            is_ambiguous_state(state)
            for state in character_matrix.values.flatten()
        ):
            raise GreedySolverError("Solver does not support ambiguous states.")

        unique_character_matrix = character_matrix.drop_duplicates()

        tree = nx.DiGraph()
        tree.add_nodes_from(list(unique_character_matrix.index))

        _solve(
            list(unique_character_matrix.index),
            tree,
            unique_character_matrix,
            weights,
            cassiopeia_tree.missing_state_indicator,
        )

        # Append duplicate samples
        duplicates_tree = self.__add_duplicates_to_tree(
            tree, character_matrix, node_name_generator
        )
        cassiopeia_tree.populate_tree(duplicates_tree, layer=layer)

        # Collapse mutationless edges
        if collapse_mutationless_edges:
            cassiopeia_tree.collapse_mutationless_edges(
                infer_ancestral_characters=True
            )

    def compute_mutation_frequencies(
        self,
        samples: List[str],
        unique_character_matrix: pd.DataFrame,
        missing_state_indicator: int = -1,
    ) -> Dict[int, Dict[int, int]]:
        """Computes character/state mutation frequencies.

        Generates a dictionary that maps each character to a dictionary of state/
        sample frequency pairs, allowing quick lookup. Subsets the character matrix
        to only include the samples in the sample set.
        Args:
            samples: The set of relevant samples in calculating frequencies
            unique_character_matrix: The character matrix from which to calculate frequencies
            missing_state_indicator: The character representing missing values
        Returns:
            A dictionary containing frequency information for each character/state
            pair
        """

        subset_cm = unique_character_matrix.loc[samples, :].to_numpy()
        freq_dict = {}
        for char in range(subset_cm.shape[1]):
            char_dict = {}
            state_counts = np.unique(subset_cm[:, char], return_counts=True)
            for i in range(len(state_counts[0])):
                state = state_counts[0][i]
                count = state_counts[1][i]
                char_dict[state] = count
            if missing_state_indicator not in char_dict:
                char_dict[missing_state_indicator] = 0
            freq_dict[char] = char_dict

        return freq_dict

    def __add_duplicates_to_tree(
        self,
        tree: nx.DiGraph,
        character_matrix: pd.DataFrame,
        node_name_generator: Generator[str, None, None],
    ) -> nx.DiGraph:
        """Takes duplicate samples and places them in the tree.

        Places samples removed in removing duplicates in the tree as sisters
        to the corresponding cells that share the same mutations.

        Args:
            tree: The tree to have duplicates added to
            character_matrix: Character matrix

        Returns:
            The tree with duplicates added
        """

        character_matrix.index.name = "index"
        duplicate_groups = (
            character_matrix[character_matrix.duplicated(keep=False) == True]
            .reset_index()
            .groupby(character_matrix.columns.tolist())["index"]
            .agg(["first", tuple])
            .set_index("first")["tuple"]
            .to_dict()
        )

        for i in duplicate_groups:
            new_internal_node = next(node_name_generator)
            nx.relabel_nodes(tree, {i: new_internal_node}, copy=False)
            for duplicate in duplicate_groups[i]:
                tree.add_edge(new_internal_node, duplicate)

        return tree
