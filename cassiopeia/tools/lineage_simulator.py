import abc
from typing import List

import networkx as nx

from .tree import Tree


class LineageSimulator(abc.ABC):
    r"""
    Abstract base class for lineage simulators.
    """
    @abc.abstractmethod
    def simulate_lineage(self) -> Tree:
        r"""Simulates a ground truth lineage"""


class PerfectBinaryTree(LineageSimulator):
    def __init__(
        self,
        generation_branch_lengths: List[float]
    ):
        self.generation_branch_lengths = generation_branch_lengths[:]

    def simulate_lineage(self) -> Tree:
        r"""
        See test for doc.
        """
        generation_branch_lengths = self.generation_branch_lengths
        n_generations = len(generation_branch_lengths)
        T = nx.DiGraph()
        T.add_nodes_from(range(2 ** (n_generations + 1) - 1))
        edges = [(int((child - 1) / 2), child)
                 for child in range(1, 2 ** (n_generations + 1) - 1)]
        node_generation = []
        for i in range(n_generations + 1):
            node_generation += [i] * 2 ** i
        T.add_edges_from(edges)
        for (parent, child) in edges:
            parent_generation = node_generation[parent]
            branch_length = generation_branch_lengths[parent_generation]
            T.edges[parent, child]["length"] = branch_length
        T.nodes[0]["age"] = sum(generation_branch_lengths)
        for child in range(1, 2 ** (n_generations + 1) - 1):
            child_generation = node_generation[child]
            branch_length = generation_branch_lengths[child_generation - 1]
            T.nodes[child]["age"] =\
                T.nodes[int((child - 1) / 2)]["age"] - branch_length
        return Tree(T)


class PerfectBinaryTreeWithRootBranch(LineageSimulator):
    def __init__(
        self,
        generation_branch_lengths: List[float]
    ):
        self.generation_branch_lengths = generation_branch_lengths

    def simulate_lineage(self) -> Tree:
        r"""
        See test for doc.
        """
        # generation_branch_lengths = self.generation_branch_lengths
        generation_branch_lengths = self.generation_branch_lengths
        n_generations = len(generation_branch_lengths)
        T = nx.DiGraph()
        T.add_nodes_from(range(2 ** n_generations))
        edges = [(int(child / 2), child)
                 for child in range(1, 2 ** n_generations)]
        T.add_edges_from(edges)
        node_generation = [0]
        for i in range(n_generations):
            node_generation += [i + 1] * 2 ** i
        for (parent, child) in edges:
            parent_generation = node_generation[parent]
            branch_length = generation_branch_lengths[parent_generation]
            T.edges[parent, child]["length"] = branch_length
        T.nodes[0]["age"] = sum(generation_branch_lengths)
        for child in range(1, 2 ** n_generations):
            child_generation = node_generation[child]
            branch_length = generation_branch_lengths[child_generation - 1]
            T.nodes[child]["age"] =\
                T.nodes[int(child / 2)]["age"] - branch_length
        return Tree(T)
