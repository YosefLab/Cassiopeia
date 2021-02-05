"""
This file stores a subclass of CassiopeiaSolver, the GreedySolver. This class
represents the structure of top-down algorithms that build the reconstructed 
tree by recursively splitting the set of samples based on some split criterion.
"""
import logging

import abc
import networkx as nx
import numpy as np
import pandas as pd
from typing import Callable, Dict, List, Optional, Tuple, Union

from cassiopeia.data import CassiopeiaTree
from cassiopeia.solver import CassiopeiaSolver
from cassiopeia.solver import solver_utilities


class GreedySolver(CassiopeiaSolver.CassiopeiaSolver):
    """
    GreedySolver is an abstract class representing the structure of top-down
    inference algorithms. The solver procedure contains logic to build a tree
    from the root by recursively paritioning the set of samples. Each subclass
    will implement "perform_split", which is the procedure for successively
    partioning the sample set.

    Args:
        character_matrix: A character matrix of observed character states for
            all samples
        missing_char: The character representing missing values
        meta_data: Any meta data associated with the samples
        priors: Prior probabilities of observing a transition from 0 to any
            state for each character
        prior_function: A function defining a transformation on the priors
            in forming weights

    Attributes:
        character_matrix: The character matrix describing the samples
        missing_char: The character representing missing values
        meta_data: Data table storing meta data for each sample
        priors: Prior probabilities of character state transitions
        weights: Weights on character/mutation pairs, derived from priors
        tree: The tree built by `self.solve()`. None if `solve` has not been
            called yet
        unique_character_matrix: A character matrix with duplicate rows filtered out
    """

    def __init__(
        self, prior_function: Optional[Callable[[float], float]] = None
    ):

        super().__init__(None, None)
        self.prior_function = prior_function

    def perform_split(
        self,
        character_matrix: pd.DataFrame,
        mutation_frequencies: Dict[int, Dict[int, int]],
        samples: List[int],
        weights: Optional[Dict[int, Dict[int, float]]] = None,
        missing_state_indicator: int = -1,
    ) -> Tuple[List[Union[int, str]], List[Union[int, str]]]:
        """Performs a partition of the samples.

        Args:
            character_matrix: Character matrix
            mutation_frequencies: A dictionary containing the frequencies of
                each character/state pair that appear in the character matrix
                restricted to the sample set
            samples: A list of samples to partition
            weights: Weighting of each (character, state) pair. Typically a
                transformation of the priors.
            missing_state_indicator: Character representing missing data.

        Returns:
            A tuple of lists, representing the left and right partitions
        """
        pass

    def solve(self, cassiopeia_tree: CassiopeiaTree):
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
        """

        # A helper function that builds the subtree given a set of samples
        def _solve(
            samples: List[Union[str, int]],
            tree: nx.DiGraph,
            character_matrix: pd.DataFrame,
            unique_character_matrix: pd.DataFrame,
            weights: Dict[int, Dict[int, float]],
            missing_state_indicator: int,
        ):

            if len(samples) == 1:
                return samples[0]
            mutation_frequencies = self.compute_mutation_frequencies(
                samples, character_matrix, missing_state_indicator
            )
            # Finds the best partition of the set given the split criteria
            clades = list(
                self.perform_split(
                    character_matrix,
                    mutation_frequencies,
                    samples,
                    weights,
                    missing_state_indicator,
                )
            )
            # Generates a root for this subtree with a unique int identifier
            root = (
                len(tree.nodes)
                - unique_character_matrix.shape[0]
                + character_matrix.shape[0]
            )
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
                    character_matrix,
                    unique_character_matrix,
                    weights,
                    missing_state_indicator,
                )
                tree.add_edge(root, child)
            return root

        # create weights
        weights = None
        if cassiopeia_tree.priors:
            if self.prior_function:
                weights = solver_utilities.transform_priors(
                    cassiopeia_tree.priors, self.prior_function
                )
            else:
                weights = solver_utilities.transform_priors(
                    cassiopeia_tree.priors, lambda x: -np.log(x)
                )

        # extract character matrix
        character_matrix = cassiopeia_tree.get_original_character_matrix()
        unique_character_matrix = character_matrix.drop_duplicates()

        # instantiate tree
        tree = nx.DiGraph()
        for i in unique_character_matrix.index:
            tree.add_node(i)

        _solve(
            list(unique_character_matrix.index),
            tree,
            character_matrix,
            unique_character_matrix,
            weights,
            cassiopeia_tree.missing_state_indicator,
        )
        # Collapse 0-mutation edges and append duplicate samples
        tree = solver_utilities.collapse_tree(
            tree,
            True,
            unique_character_matrix,
            cassiopeia_tree.missing_state_indicator,
        )
        tree = self.add_duplicates_to_tree(tree, character_matrix)

        cassiopeia_tree.populate_tree(tree)

    def compute_mutation_frequencies(
        self,
        samples: List[Union[int, str]],
        unique_character_matrix: pd.DataFrame,
        missing_state_indicator: int = -1,
    ) -> Dict[int, Dict[int, int]]:
        """Computes the number of samples in a character matrix that have each
        character/state mutation.

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
        cm = unique_character_matrix.loc[samples, :]
        freq_dict = {}
        for char in range(cm.shape[1]):
            char_dict = {}
            state_counts = np.unique(cm.iloc[:, char], return_counts=True)
            for i in range(len(state_counts[0])):
                state = state_counts[0][i]
                count = state_counts[1][i]
                char_dict[state] = count
            if missing_state_indicator not in char_dict:
                char_dict[missing_state_indicator] = 0
            freq_dict[char] = char_dict
        return freq_dict

    def add_duplicates_to_tree(
        self,
        tree: nx.DiGraph,
        character_matrix: pd.DataFrame
    ) -> nx.DiGraph:
        """Takes duplicate samples and places them in the tree.

        Places samples removed in removing duplicates in the tree as sisters
        to the corresponding cells that share the same mutations.

        Args:
            tree: The tree to have duplicates added to
            character_matrix: Character matrix

        Returns:
            A tree with duplicates added
        """
        duplicate_groups = (
            character_matrix[character_matrix.duplicated(keep=False) == True]
            .reset_index()
            .groupby(character_matrix.columns.tolist())["index"]
            .agg(["first", tuple])
            .set_index("first")["tuple"]
            .to_dict()
        )

        for i in duplicate_groups:
            new_internal_node = (
                max([i for i in tree.nodes if type(i) == int]) + 1
            )
            nx.relabel_nodes(tree, {i: new_internal_node}, copy=False)
            for duplicate in duplicate_groups[i]:
                tree.add_node(duplicate)
                tree.add_edge(new_internal_node, duplicate)

        return tree
