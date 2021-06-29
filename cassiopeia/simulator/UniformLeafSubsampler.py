"""
A subclass of LeafSubsampler, the UniformLeafSubsampler. 

Takes a uniform random sample of the leaves of a CassiopeiaTree and produces a
new CassiopeiaTree that keeps only the lineages pertaining to the sample.
"""

import abc
import copy
import networkx as nx
import numpy as np
from typing import Optional

from cassiopeia.data import CassiopeiaTree
from cassiopeia.simulator.LeafSubsampler import (
    LeafSubsampler,
    LeafSubsamplerError,
    EmptySubtreeError,
)


class UniformLeafSubsampler(LeafSubsampler):
    def __init__(
        self,
        ratio: Optional[float] = None,
        number_of_leaves: Optional[int] = None,
        sampling_probability: Optional[float] = None
    ):
        """
        Uniformly subsample leaf samples of a CassiopeiaTree.

        If 'ratio' is provided, samples 'ratio' of the leaves, rounded down,
        uniformly at random. If instead 'number_of_leaves' is provided, 
        'number_of_leaves' of the leaves are sampled uniformly at random.
        If 'sampling_probability' is provided, each leaf is sampled IID with
        probability p. Only one of the criteria can be provided.

        Args:
            ratio: Specifies the number of leaves to be sampled as a ratio of
                the total number of leaves
            number_of_leaves: Explicitly specifies the number of leaves to be sampled
            sampling_probability: Probability with which to sample each cell.
        """
        if ratio is None and number_of_leaves is None and sampling_probability is None:
            raise LeafSubsamplerError(
                "At least one of 'ratio', 'number_of_leaves' and "
                "'sampling_probability' must be specified."
            )
        if ((ratio is None) + (number_of_leaves is None) + (sampling_probability is None)) <= 1:
            raise LeafSubsamplerError(
                "Exactly one of 'ratio', 'number_of_leaves' and "
                "'sampling_probability' must be specified."
            )
        self.__ratio = ratio
        self.__number_of_leaves = number_of_leaves
        self.__sampling_probability = sampling_probability

    def subsample_leaves(
        self, tree: CassiopeiaTree, collapse_source: str = None
    ) -> CassiopeiaTree:
        """Uniformly subsample leaf samples of a given tree.

        Generates a uniform random sample on the leaves of the given
        CassiopeiaTree and returns a tree pruned to contain lineages relevant
        to only leaves in the sample (the "induced subtree" on the sample).
        All fields on the original character matrix persist, but maintains
        character states, meta data, and the dissimilarity map for the sampled
        cells only.

        Args:
            tree: The CassiopeiaTree for which to subsample leaves
            collapse_source: The source node from which to collapse
                unifurcations

        Returns:
            A new CassiopeiaTree that is the induced subtree on a sample of the
            leaves in the given tree.
        """
        ratio = self.__ratio
        number_of_leaves = self.__number_of_leaves
        sampling_probability = self.__sampling_probability

        subsampled_tree = copy.deepcopy(tree)

        if sampling_probability is not None:
            if sampling_probability < 0 or sampling_probability > 1:
                raise LeafSubsamplerError("sampling_probability must be in [0, 1].")
            number_of_leaves_original = len(tree.leaves)
            leaf_remove_indices =\
                np.random.binomial(
                    n=1,
                    p=1.0 - sampling_probability,
                    size=(number_of_leaves_original)
                )
            if np.sum(leaf_remove_indices) == number_of_leaves_original:
                raise EmptySubtreeError(
                    "No cells were subsampled! "
                    "sampling_probability might be too low."
                )
            leaf_remove = [
                subsampled_tree.leaves[i]
                for i in range(number_of_leaves_original)
                if leaf_remove_indices[i] == 1
            ]
        else:
            n_subsample = (
                number_of_leaves if number_of_leaves is not None else int(tree.n_cell * ratio)
            )
            if n_subsample <= 0:
                raise LeafSubsamplerError(
                    "Specified number of leaves sampled is <= 0."
                )
            if n_subsample > tree.n_cell:
                raise LeafSubsamplerError(
                    "Specified number of leaves sampled is greater than the number"
                    " of leaves in the given tree."
                )

            n_remove = len(tree.leaves) - n_subsample
            leaf_remove = np.random.choice(subsampled_tree.leaves, n_remove, replace=False)

        for i in leaf_remove:
            subsampled_tree.remove_leaf_and_prune_lineage(i)

        if collapse_source is None:
            collapse_source = subsampled_tree.root
        subsampled_tree.collapse_unifurcations(source=collapse_source)

        # Copy and annotate branch lengths and times
        subsampled_tree.set_times(
            dict(
                [(node, tree.get_time(node)) for node in subsampled_tree.nodes]
            )
        )
        return subsampled_tree
