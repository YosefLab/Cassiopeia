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


class GreedySolver(CassiopeiaSolver.CassiopeiaSolver):
    def __init__(
        self,
        character_matrix: pd.DataFrame,
        meta_data: Optional[pd.DataFrame] = None,
        priors: Optional[Dict] = None,
    ):

        super().__init__(character_matrix, meta_data, priors)
        prune_cm = remove_duplicates(self.character_matrix)
        self.map_cm, self.state_map = map_states(prune_cm, self.unedit_char, self.missing_char)

    @abc.abstractmethod
    def perform_split(self, samples: List[int]) -> Tuple[List[int], List[int]]:
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
        T = nx.DiGraph()
        for i in range(self.map_cm.shape[0]):
            T.add_node(i)
        def build_tree(samples):
            left_set, right_set = self.perform_split(samples)
            root = len(T.nodes) - len(samples) + self.map_cm.shape[0]
            T.add_node(root)
            left_child = build_tree(left_set)
            right_child = build_tree(right_set)
            T.add_edge(root, left_child)
            T.add_edge(root, right_child)
            return root
        build_tree(range(self.map_cm.shape[0]))
        collapse_tree_recon(T, self.map_cm)
        return T

    def compute_mutation_frequencies(self, samples: List[int] = None) -> pd.DataFrame:
        """Computes the frequency of character/state pairs in the samples.

        Args:
            samples: A list of samples

        Returns:
            A dataframe mapping character/state pairs to frequencies
        """
        cm = self.character_matrix
        k = cm.shape[1]
        m = max(cm.max()) + 1
        F = np.zeros((k,m), dtype=int)
        if not samples:
            samples = list(range(cm.shape[0]))
        for i in samples:
            for j in range(k):
                F[j][cm.iloc(i, j)] += 1
        return F

def remove_duplicates(cm, verbose = False) -> pd.DataFrame:
    """Removes doublets in a character matrix prior to analysis.
    """
    cm_drop = cm.drop_duplicates()

    if verbose:
        logging.info(f"{cm.shape[0] - cm_drop.shape[0]} duplicate samples removed")

    return cm_drop

def map_states(cm, unedit_char, missing_char) -> Tuple[pd.DataFrame, Dict[int, str]]:
    """Maps the characters in a character matrix to the effective states.
    """
    state_map = {}
    index = 1
    unique_states = sorted(pd.unique(cm.values.ravel()))
    if missing_char in unique_states:
        unique_states.remove(missing_char)
        state_map[-1] = missing_char
        cm = cm.replace(missing_char, -1)
    unique_states.remove(unedit_char)
    state_map[0] = unedit_char
    cm = cm.replace(unedit_char, 0)
    for i in unique_states:
        state_map[index] = i
        cm = cm.replace(i, index)
        index += 1

    return cm, state_map

def consensus_vec(vec1, vec2):
    """Builds a consensus vector from two, obeying Camin-Sokal Parsimony.
    """
    assert(len(vec1) == len(vec2))
    consensus = [0] * len(vec1)
    for i in range(len(vec1)):
        if vec1[i] == -1 and vec2[i] != -1:
            consensus[i] = vec2[i]
        if vec2[i] == -1 and vec1[i] != -1:
            consensus[i] = vec1[i]
        if vec1[i] == vec2[i] and vec1[i] != 0:
            consensus[i] = vec1[i]
    return consensus

def annotate_internal_nodes(network, node, char_map):
    """Annotates the character vectors of the internal nodes of a reconstructed
    network from the samples, obeying Camin-Sokal Parsimony.
    """
    if network.out_degree(node) == 0:
        return
    vecs = []
    for i in network.successors(node):
        annotate_internal_nodes(network, i, char_map)
        vecs.append(char_map[i])
    assert(len(vecs) == 2)
    consensus = consensus_vec(vecs[0], vecs[1])
    char_map[node] = consensus
    return

def collapse_edges_recon(network, node, char_map):
    """A helper function to help collapse edges in a tree.
    """
    if network.out_degree(node) == 0:
        return
    to_remove = []
    to_collapse = []
    for i in network.successors(node):
        to_collapse.append(i)
    for i in to_collapse:
        collapse_edges_recon(network, i, char_map)
        if char_map[i] == char_map[node]:
            for j in network.successors(i):
                network.add_edge(node, j)
            to_remove.append(i)
    for i in to_remove:
        network.remove_node(i)
    return

def collapse_tree_recon(T, cm):
    """Collapses non-informative edges (edges with 0 mutations) in a tree.
    """
    leaves = [n for n in T if T.out_degree(n) == 0 and T.in_degree(n) == 1]
    char_map = {}
    for i in leaves:
        char_map[i] = list(cm.iloc[i,:])
    
    root = [n for n in T if T.in_degree(n) == 0][0]
    annotate_internal_nodes(T, root, char_map)
    collapse_edges_recon(T, root, char_map)
    