"""Module defining a subclass of CassiopeiaSolver, the GreedySolver.

This class represents the structure of top-down algorithms that build the reconstructed
tree by recursively splitting the set of samples based on some split criterion.
"""

from collections.abc import Generator

import networkx as nx
import numpy as np
import pandas as pd

from cassiopeia.data import CassiopeiaTree
from cassiopeia.mixins import (
    GreedySolverError,
    find_duplicate_groups,
    is_ambiguous_state,
    unravel_ambiguous_states,
)
from cassiopeia.solver import CassiopeiaSolver, solver_utilities


class GreedySolver(CassiopeiaSolver.CassiopeiaSolver):
    """A Greedy Cassiopeia solver.

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

    Attributes
    ----------
        prior_transformation: Function to use when transforming priors into
            weights.
    """

    def __init__(self, prior_transformation: str = "negative_log"):
        super().__init__(prior_transformation)
        self.allow_ambiguous = False

    def perform_split(
        self,
        character_matrix: pd.DataFrame,
        samples: list[int],
        weights: dict[int, dict[int, float]] | None = None,
        missing_state_indicator: int = -1,
    ) -> tuple[list[int | str], list[int | str]]:
        """Performs a partition of the samples.

        Args:
            character_matrix: Character matrix
            samples: A list of samples to partition
            weights: Weighting of each (character, state) pair. Typically a
                transformation of the priors.
            missing_state_indicator: Character representing missing data.

        Returns
        -------
            A tuple of lists, representing the left and right partition groups
        """
        pass

    def solve(
        self,
        cassiopeia_tree: CassiopeiaTree,
        layer: str | None = None,
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
            logfile: File location to log output. Not currently used.
        """

        # A helper function that builds the subtree given a set of samples
        def _solve(
            samples: list[str | int],
            tree: nx.DiGraph,
            unique_character_matrix: pd.DataFrame,
            weights: dict[int, dict[int, float]],
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
            weights = solver_utilities.transform_priors(cassiopeia_tree.priors, self.prior_transformation)

        # extract character matrix
        if layer:
            character_matrix = cassiopeia_tree.layers[layer].copy()
        else:
            character_matrix = cassiopeia_tree.character_matrix.copy()

        # Raise exception if the character matrix has ambiguous states.
        if any(is_ambiguous_state(state) for state in character_matrix.values.flatten()) and not self.allow_ambiguous:
            raise GreedySolverError("Ambiguous states are not currently supported with this solver.")

        keep_rows = (
            character_matrix.apply(
                lambda x: [set(s) if is_ambiguous_state(s) else {s} for s in x.values],
                axis=0,
            )
            .apply(tuple, axis=1)
            .drop_duplicates()
            .index.values
        )
        unique_character_matrix = character_matrix.loc[keep_rows].copy()

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
        duplicates_tree = self.__add_duplicates_to_tree(tree, character_matrix, node_name_generator)
        cassiopeia_tree.populate_tree(duplicates_tree, layer=layer)

        # Collapse mutationless edges
        if collapse_mutationless_edges:
            cassiopeia_tree.collapse_mutationless_edges(infer_ancestral_characters=True)

    def compute_mutation_frequencies(
        self,
        samples: list[str],
        unique_character_matrix: pd.DataFrame,
        missing_state_indicator: int = -1,
    ) -> dict[int, dict[int, int]]:
        """Computes character/state mutation frequencies.

        Generates a dictionary that maps each character to a dictionary of state/
        sample frequency pairs, allowing quick lookup. Subsets the character matrix
        to only include the samples in the sample set.

        This currently supports ambiguous states, for the GreedySolvers that
        support ambiguous states during inference.

        Args:
            samples: The set of relevant samples in calculating frequencies
            unique_character_matrix: The character matrix from which to
                calculate frequencies
            missing_state_indicator: The character representing missing values
        Returns:
            A dictionary containing frequency information for each character/state
            pair
        """
        subset_cm = unique_character_matrix.loc[samples, :].to_numpy()
        freq_dict = {}
        for char in range(subset_cm.shape[1]):
            char_dict = {}
            all_states = unravel_ambiguous_states(subset_cm[:, char])
            state_counts = np.unique(all_states, return_counts=True)

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

        Returns
        -------
            The tree with duplicates added
        """
        duplicate_mappings = find_duplicate_groups(character_matrix)

        for i in duplicate_mappings:
            new_internal_node = next(node_name_generator)
            nx.relabel_nodes(tree, {i: new_internal_node}, copy=False)
            for duplicate in duplicate_mappings[i]:
                tree.add_edge(new_internal_node, duplicate)

        return tree
