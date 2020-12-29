import abc
from typing import List

import networkx as nx

from .tree import Tree


class LineageSimulator(abc.ABC):
    r"""
    Abstract base class for lineage simulators.

    A LineageSimulator implements the method simulate_lineage that generates a
    lineage tree (i.e. a phylogeny, in the form of a Tree).
    """

    @abc.abstractmethod
    def simulate_lineage(self) -> Tree:
        r"""
        Simulates a lineage tree, i.e. a Tree with branch lengths and age
        specified for each node. Additional information such as cell fitness,
        etc. might be specified by more complex simulators.
        """


class PerfectBinaryTree(LineageSimulator):
    r"""
    Generates a perfect binary tree with given branch lengths at each depth.

    Args:
        generation_branch_lengths: The branches at depth d in the tree will have
            length generation_branch_lengths[d]
    """

    def __init__(self, generation_branch_lengths: List[float]):
        self.generation_branch_lengths = generation_branch_lengths[:]

    def simulate_lineage(self) -> Tree:
        r"""
        See base class.
        """
        generation_branch_lengths = self.generation_branch_lengths
        n_generations = len(generation_branch_lengths)
        tree = nx.DiGraph()
        tree.add_nodes_from(range(2 ** (n_generations + 1) - 1))
        edges = [
            (int((child - 1) / 2), child)
            for child in range(1, 2 ** (n_generations + 1) - 1)
        ]
        node_generation = []
        for i in range(n_generations + 1):
            node_generation += [i] * 2 ** i
        tree.add_edges_from(edges)
        for (parent, child) in edges:
            parent_generation = node_generation[parent]
            branch_length = generation_branch_lengths[parent_generation]
            tree.edges[parent, child]["length"] = branch_length
        tree.nodes[0]["age"] = sum(generation_branch_lengths)
        for child in range(1, 2 ** (n_generations + 1) - 1):
            child_generation = node_generation[child]
            branch_length = generation_branch_lengths[child_generation - 1]
            tree.nodes[child]["age"] = (
                tree.nodes[int((child - 1) / 2)]["age"] - branch_length
            )
        return Tree(tree)


class PerfectBinaryTreeWithRootBranch(LineageSimulator):
    r"""
    Generates a perfect binary tree *hanging from a branch*, with given branch
    lengths at each depth.

    Args:
        generation_branch_lengths: The branches at depth d in the tree will have
            length generation_branch_lengths[d]
    """

    def __init__(self, generation_branch_lengths: List[float]):
        self.generation_branch_lengths = generation_branch_lengths

    def simulate_lineage(self) -> Tree:
        r"""
        See base class.
        """
        # generation_branch_lengths = self.generation_branch_lengths
        generation_branch_lengths = self.generation_branch_lengths
        n_generations = len(generation_branch_lengths)
        tree = nx.DiGraph()
        tree.add_nodes_from(range(2 ** n_generations))
        edges = [
            (int(child / 2), child) for child in range(1, 2 ** n_generations)
        ]
        tree.add_edges_from(edges)
        node_generation = [0]
        for i in range(n_generations):
            node_generation += [i + 1] * 2 ** i
        for (parent, child) in edges:
            parent_generation = node_generation[parent]
            branch_length = generation_branch_lengths[parent_generation]
            tree.edges[parent, child]["length"] = branch_length
        tree.nodes[0]["age"] = sum(generation_branch_lengths)
        for child in range(1, 2 ** n_generations):
            child_generation = node_generation[child]
            branch_length = generation_branch_lengths[child_generation - 1]
            tree.nodes[child]["age"] = (
                tree.nodes[int(child / 2)]["age"] - branch_length
            )
        return Tree(tree)
