import abc
import networkx as nx
import numpy as np

from cassiopeia.data import CassiopeiaTree


class CellSubsamplerError(Exception):
    """An Exception class for the CellSubsampler class."""

    pass


class CellSubsampler(abc.ABC):
    r"""
    Abstract base class for all cell samplers.

    A CellSubsampler implements a method 'subsample' which, given a Tree,
    returns a second Tree which is the result of subsampling cells
    (i.e. leafs) of the tree. Only the tree topology will be created for the
    new tree.
    """

    @abc.abstractmethod
    def subsample(self, tree: CassiopeiaTree) -> CassiopeiaTree:
        r"""
        Returns a new CassiopeiaTree which is the result of subsampling
        the cells in the original CassiopeiaTree.

        Args:
            tree: The tree for which to subsample leaves.
        """


class UniformCellSubsampler(CellSubsampler):
    def __init__(self, ratio: float):
        r"""
        Samples 'ratio' of the leaves, rounded down, uniformly at random.
        """
        self.__ratio = ratio

    def subsample(self, tree: CassiopeiaTree) -> CassiopeiaTree:
        ratio = self.__ratio
        n_subsample = int(tree.n_cell * ratio)
        if n_subsample == 0:
            raise CellSubsamplerError(
                "ratio too low: no cells would be " "sampled."
            )

        # First determine which nodes are part of the induced subgraph.
        leaf_keep_idx = np.random.choice(
            range(tree.n_cell), n_subsample, replace=False
        )
        leaves_in_induced_subtree = [tree.leaves[i] for i in leaf_keep_idx]
        induced_subtree_degs = dict(
            [(leaf, 0) for leaf in leaves_in_induced_subtree]
        )

        nodes_in_induced_subtree = set(leaves_in_induced_subtree)
        for node in tree.depth_first_traverse_nodes(postorder=True):
            children = tree.children(node)
            induced_subtree_deg = sum(
                [child in nodes_in_induced_subtree for child in children]
            )
            if induced_subtree_deg > 0:
                nodes_in_induced_subtree.add(node)
                induced_subtree_degs[node] = induced_subtree_deg

        # For debugging:
        # print(f"leaves_in_induced_subtree = {leaves_in_induced_subtree}")
        # print(f"nodes_in_induced_subtree = {nodes_in_induced_subtree}")
        # print(f"induced_subtree_degs = {induced_subtree_degs}")
        nodes = []
        edges = []
        up = {}
        for node in tree.depth_first_traverse_nodes(postorder=False):
            if node == tree.root:
                nodes.append(node)
                up[node] = node
                continue

            if node not in nodes_in_induced_subtree:
                continue

            if induced_subtree_degs[tree.parent(node)] >= 2:
                up[node] = tree.parent(node)
            else:
                up[node] = up[tree.parent(node)]

            if (
                induced_subtree_degs[node] >= 2
                or induced_subtree_degs[node] == 0
            ):
                nodes.append(node)
                edges.append((up[node], node))
        subtree_topology = nx.DiGraph()
        subtree_topology.add_nodes_from(nodes)
        subtree_topology.add_edges_from(edges)
        res = CassiopeiaTree(
            tree=subtree_topology,
        )
        # Copy times over
        res.set_times(dict([(node, tree.get_time(node)) for node in res.nodes]))
        return res
