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
from typing import Dict, List, Optional, Tuple

from cassiopeia.solver import CassiopeiaSolver
from cassiopeia.solver import solver_utilities as utils


class GreedySolver(CassiopeiaSolver.CassiopeiaSolver):
    def __init__(
        self,
        character_matrix: pd.DataFrame,
        missing_char: str,
        meta_data: Optional[pd.DataFrame] = None,
        priors: Optional[Dict] = None,
    ):

        super().__init__(character_matrix, missing_char, meta_data, priors)
        self.prune_cm = self.character_matrix.drop_duplicates()
        self.tree = nx.DiGraph()
        for i in range(self.prune_cm.shape[0]):
            self.tree.add_node(i)

    @abc.abstractmethod
    def perform_split(
        self, samples: List[int], F
    ) -> Tuple[List[int], List[int]]:
        """Performs a partition of the samples.

        Args:
            samples: A list of samples to partition

        Returns:
            A tuple of lists, representing the left and right partitions
        """
        pass

    def solve(self) -> nx.DiGraph:
        """Implements a top-down greedy solving procedure.

        Returns:
            A networkx directed graph representing the reconstructed tree
        """

        # A helper function that builds the subtree given a set of samples
        def _solve(samples):
            F = utils.compute_mutation_frequencies(samples)
            # Finds the best partition of the set given the split criteria
            left_set, right_set = self.perform_split(samples, F)
            # Generates a root for this subtree with a unique int identifier
            root = len(self.tree.nodes) - len(samples) + self.prune_cm.shape[0]
            self.tree.add_node(root)
            # Recursively generate the left and right subtrees
            left_child = _solve(left_set)
            right_child = _solve(right_set)
            self.tree.add_edge(root, left_child)
            self.tree.add_edge(root, right_child)
            return root

        _solve(range(self.prune_cm.shape[0]))
        # Collapse 0-mutation edges and appends duplicate samples
        utils.collapse_tree(self.tree, self.prune_cm, True, self.missing_char)
        utils.post_process_tree(self.tree, self.character_matrix)
