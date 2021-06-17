"""
Abstract class LeafSubsampler. Samples the leaves of CassiopeiaTrees and 
generates a tree that keeps only the lineages pertaining to the sample.

All leaf subsamplers are derived classes of this abstract class, and at a minimum
implement a method called `subsample_leaves`.
"""

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
        Subsamples the leaves of a CassiopeiaTree.

        Returns a new CassiopeiaTree which is the result of subsampling the
        leaves in the original CassiopeiaTree and removing ancestral nodes no
        longer relevant to the sample. All fields on the original character
        matrix persist, but maintains character states, meta data, and the
        dissimilarity map for the sampled cells only.

        Args:
            tree: The CassiopeiaTree for which to subsample leaves

        Returns:
            A new CassiopeiaTree that is the induced subtree on a sample of the
            leaves in the given tree.
        """
