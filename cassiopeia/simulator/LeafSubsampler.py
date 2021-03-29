import abc
import networkx as nx
import numpy as np
from typing import Optional

from cassiopeia.data import CassiopeiaTree


class LeafSubsamplerError(Exception):
    """An Exception class for the LeafSubsampler class."""

    pass


class LeafSubsampler(abc.ABC):
    """
    Abstract base class for all leaf samplers.

    A LeafSubsampler implements a method 'subsample_leaves' which, given a 
    tree, generates a sample of the observed leaves in that tree and returns a 
    new tree which is the induced subtree (tree containing only lineages that 
    contain a sampled leaf) of the original tree on that sample.
    """

    @abc.abstractmethod
    def subsample_leaves(self, tree: CassiopeiaTree) -> CassiopeiaTree:
        """
        Returns a new CassiopeiaTree which is the result of subsampling
        the leafs in the original CassiopeiaTree.

        Args:
            tree: The tree for which to subsample leaves and generate an
                induced subtree from
        """