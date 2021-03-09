import abc
from typing import List

import networkx as nx
import numpy as np
from queue import Queue

from cassiopeia.data import CassiopeiaTree


class LineageSimulator(abc.ABC):
    r"""
    Abstract base class for lineage simulators.

    A LineageSimulator implements the method simulate_lineage that generates a
    lineage tree (i.e. a phylogeny, in the form of a Tree).
    """

    @abc.abstractmethod
    def simulate_lineage(self) -> CassiopeiaTree:
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

    def simulate_lineage(self) -> CassiopeiaTree:
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
        times = {}
        for node in tree.nodes:
            times[node] = tree.nodes[0]["age"] - tree.nodes[node]["age"]
        res = CassiopeiaTree(tree=tree)
        res.set_times(times)
        return res


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

    def simulate_lineage(self) -> CassiopeiaTree:
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
        times = {}
        for node in tree.nodes:
            times[node] = tree.nodes[0]["age"] - tree.nodes[node]["age"]
        res = CassiopeiaTree(tree=tree)
        res.set_times(times)
        return res


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

    def simulate_lineage(self) -> CassiopeiaTree:
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
        times = {}
        for node in tree_nx.nodes:
            times[node] = node_age[0] - node_age[node]
        tree = CassiopeiaTree(tree=tree_nx)
        tree.set_times(times)
        return tree


class TumorWithAFitSubclone(LineageSimulator):
    r"""
    TODO

    Args:
        branch_length: TODO
        TODO
    """

    def __init__(
        self,
        branch_length: float,
        branch_length_fit: float,
        experiment_duration: float,
        generations_until_fit_subclone: int,
    ):
        self.branch_length = branch_length
        self.branch_length_fit = branch_length_fit
        self.experiment_duration = experiment_duration
        self.generations_until_fit_subclone = generations_until_fit_subclone

    def simulate_lineage(self) -> CassiopeiaTree:
        r"""
        See base class.
        """
        branch_length = self.branch_length
        branch_length_fit = self.branch_length_fit
        experiment_duration = self.experiment_duration
        generations_until_fit_subclone = self.generations_until_fit_subclone

        def node_name_generator():
            i = 0
            while True:
                yield str(i)
                i += 1

        tree = nx.DiGraph()  # This is what will get populated.

        names = node_name_generator()
        q = Queue()  # (node, time, fitness, generation)
        times = {}

        root = next(names) + "_unfit"
        tree.add_node(root)
        times[root] = 0.0

        root_child = next(names) + "_unfit"
        tree.add_edge(root, root_child)
        q.put((root_child, 0.0, "unfit", 0))
        subclone_started = False
        while not q.empty():
            # Pop next node
            (node, time, node_fitness, generation) = q.get()
            time_till_division = (
                branch_length if node_fitness == "unfit" else branch_length_fit
            )
            time_of_division = time + time_till_division
            if time_of_division >= experiment_duration:
                # Not enough time left for the cell to divide.
                times[node] = experiment_duration
                continue
            # Create children, add edges to them, and push children to the
            # queue.
            times[node] = time_of_division
            left_child_fitness = node_fitness
            right_child_fitness = node_fitness
            if (
                not subclone_started
                and generation + 1 == generations_until_fit_subclone
            ):
                # Start the subclone
                subclone_started = True
                left_child_fitness = "fit"
            left_child = next(names) + "_" + left_child_fitness
            right_child = next(names) + "_" + right_child_fitness
            tree.add_nodes_from([left_child, right_child])
            tree.add_edges_from([(node, left_child), (node, right_child)])
            q.put(
                (
                    left_child,
                    time_of_division,
                    left_child_fitness,
                    generation + 1,
                )
            )
            q.put(
                (
                    right_child,
                    time_of_division,
                    right_child_fitness,
                    generation + 1,
                )
            )
        res = CassiopeiaTree(tree=tree)
        res.set_times(times)
        return res
