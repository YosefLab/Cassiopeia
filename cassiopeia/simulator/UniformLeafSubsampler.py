import abc
import networkx as nx
import numpy as np
from typing import Optional

from cassiopeia.data import CassiopeiaTree
from cassiopeia.simulator.LeafSubsampler import LeafSubsampler, LeafSubsamplerError


class UniformLeafSubsampler(LeafSubsampler):
    def __init__(
        self, ratio: Optional[float] = None, n_leaves: Optional[int] = None
    ):
        """
        Uniformly subsample leaf samples.

        If 'ratio' is provided, samples 'ratio' of the leaves, rounded down,
        uniformly at random. If instead 'n_leaves' is provided, 'n_leaves' of
        the leaves are sampled uniformly at random. Only one of the two
        criteria can be provided.

        Args:
            ratio: Specifies the number of leaves to be sampled as a ratio of
                the total number of leaves
            n_leaves: Explicitly specifies the number of leaves to be sampled
        """
        if ratio is None and n_leaves is None:
            raise LeafSubsamplerError(
                "At least one of 'ratio' and 'n_leaves' " "must be specified."
            )
        if ratio is not None and n_leaves is not None:
            raise LeafSubsamplerError(
                "Exactly one of 'ratio' and 'n_leaves'" "must be specified."
            )
        self.__ratio = ratio
        self.__n_leaves = n_leaves

    def subsample_leaves(self, tree: CassiopeiaTree) -> CassiopeiaTree:
        """Uniformly subsample leaf samples of a given tree.

        Note that the input tree must have an implicit root with degree one.
        The first edge then represents the lifetime of the root node before
        division.
        
        Args:
            tree: The tree for which to subsample leaves and generate an
                induced subtree from
        Returns:
            A subtree of the original tree that contains only lineages that
            contain leaves in the sample
        """
        ratio = self.__ratio
        n_leaves = self.__n_leaves
        n_subsample = (
            n_leaves if n_leaves is not None else int(tree.n_cell * ratio)
        )
        if n_subsample <= 0:
            raise LeafSubsamplerError(
                "Specified number of leaves sampled is <= 0."
            )

        n_remove = len(tree.leaves) - n_subsample
        subsampled_tree = CassiopeiaTree(tree=tree.get_tree_topology())
        leaf_remove = np.random.choice(subsampled_tree.leaves, n_remove, replace=False)
        for i in leaf_remove:
            subsampled_tree.remove_and_prune_lineage(i)
        root_child = subsampled_tree.children(subsampled_tree.root)[0]
        subsampled_tree.collapse_unifurcations(source = root_child)

        # Copy and annotate branch lengths and times
        subsampled_tree.set_times(dict([(node, tree.get_time(node)) for node in subsampled_tree.nodes]))
        return subsampled_tree
