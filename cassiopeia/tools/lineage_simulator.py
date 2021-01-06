import abc
from typing import List

import networkx as nx
import numpy as np

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


class BirthProcess(LineageSimulator):
    r"""
    A Birth Process with exponential holding times.

    Args:
        birth_rate: Birth rate of the process
        tree_depth: Depth of the simulated tree
    """

    def __init__(self, birth_rate: float, tree_depth: float):
        self.birth_rate = birth_rate
        self.tree_depth = tree_depth

    def simulate_lineage(self) -> Tree:
        r"""
        See base class.
        """
        tree_depth = self.tree_depth
        birth_rate = self.birth_rate
        node_age = {}
        node_age[0] = tree_depth
        live_nodes = [1]
        edges = [(0, 1)]
        t = 0
        last_node_id = 1
        while t < tree_depth:
            num_live_nodes = len(live_nodes)
            # Wait till next node divides.
            waiting_time = np.random.exponential(
                1.0 / (birth_rate * num_live_nodes)
            )
            when_node_divides = t + waiting_time
            del waiting_time
            if when_node_divides >= tree_depth:
                # The simulation has ended.
                for node in live_nodes:
                    node_age[node] = 0
                del live_nodes
                break
            # Choose which node divides uniformly at random
            node_that_divides = live_nodes[
                np.random.randint(low=0, high=num_live_nodes)
            ]
            # Remove the node that divided and add its two children
            live_nodes.remove(node_that_divides)
            left_child_id = last_node_id + 1
            right_child_id = last_node_id + 2
            last_node_id += 2
            live_nodes += [left_child_id, right_child_id]
            edges += [
                (node_that_divides, left_child_id),
                (node_that_divides, right_child_id),
            ]
            node_age[node_that_divides] = tree_depth - when_node_divides
            t = when_node_divides
        tree_nx = nx.DiGraph()
        tree_nx.add_nodes_from(range(last_node_id + 1))
        tree_nx.add_edges_from(edges)
        tree = Tree(tree_nx)
        for node in tree.nodes():
            tree.set_age(node, node_age[node])
        tree.set_edge_length_from_node_ages()
        return tree
