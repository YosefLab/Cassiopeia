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
from cassiopeia.solver import utils

class GreedySolver(CassiopeiaSolver.CassiopeiaSolver):
    def __init__(
        self,
        character_matrix: pd.DataFrame,
        missing_char: str,
        meta_data: Optional[pd.DataFrame] = None,
        priors: Optional[Dict] = None
    ):

        super().__init__(character_matrix, missing_char, meta_data, priors)
        self.prune_cm = self.character_matrix.drop_duplicates()
        # self.map_cm, self.state_map = utils.map_states(prune_cm, self.unedit_char, self.missing_char)
        self.recon = nx.DiGraph()
        for i in range(self.prune_cm.shape[0]):
            self.recon.add_node(i)

    @abc.abstractmethod
    def perform_split(self, samples: List[int], F) -> Tuple[List[int], List[int]]:
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
        def _solve(samples):
            F = utils.compute_mutation_frequencies(samples)
            left_set, right_set = self.perform_split(samples, F)
            root = len(self.recon.nodes) - len(samples) + self.prune_cm.shape[0]
            self.recon.add_node(root)
            left_child = _solve(left_set)
            right_child = _solve(right_set)
            self.recon.add_edge(root, left_child)
            self.recon.add_edge(root, right_child)
            return root

        _solve(range(self.prune_cm.shape[0]))
        utils.collapse_tree_recon(self.recon, self.prune_cm)
        utils.post_process_tree(self.recon, self.character_matrix)
        return self.recon
