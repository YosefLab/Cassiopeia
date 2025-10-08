"""
A subclass of LeafSubsampler, the UniformLeafSubsampler.

Takes a uniform random sample of the leaves of a CassiopeiaTree and produces a
new CassiopeiaTree that keeps only the lineages pertaining to the sample.
"""

import copy

import numpy as np

from cassiopeia.data import CassiopeiaTree
from cassiopeia.simulator.LeafSubsampler import LeafSubsampler
from cassiopeia.mixins import LeafSubsamplerError


class UniformLeafSubsampler(LeafSubsampler):
    def __init__(
        self,
        ratio: float | None = None,
        number_of_leaves: int | None = None,
    ):
        """
        Uniformly subsample leaf samples of a CassiopeiaTree.

        If 'ratio' is provided, samples 'ratio' of the leaves, rounded down,
        uniformly at random. If instead 'number_of_leaves' is provided,
        'number_of_leaves' of the leaves are sampled uniformly at random. Only
        one of the two criteria can be provided.

        Args:
            ratio: Specifies the number of leaves to be sampled as a ratio of
                the total number of leaves
            number_of_leaves: Explicitly specifies the number of leaves to be sampled
        """
        if ratio is None and number_of_leaves is None:
            raise LeafSubsamplerError(
                "At least one of 'ratio' and 'number_of_leaves' "
                "must be specified."
            )
        if ratio is not None and number_of_leaves is not None:
            raise LeafSubsamplerError(
                "Exactly one of 'ratio' and 'number_of_leaves'"
                "must be specified."
            )
        self.__ratio = ratio
        self.__number_of_leaves = number_of_leaves

    def subsample_leaves(
        self, tree: CassiopeiaTree, keep_singular_root_edge: bool = True
    ) -> CassiopeiaTree:
        """Uniformly subsample leaf samples of a given tree.

        Generates a uniform random sample on the leaves of the given
        CassiopeiaTree and returns a tree pruned to contain lineages relevant
        to only leaves in the sample (the "induced subtree" on the sample).
        All fields on the original character matrix persist, but maintains
        character states, meta data, and the dissimilarity map for the sampled
        cells only.

        Has the option to keep the single edge leading from the root in the
        induced subtree, if it exists. This edge is often used to represent the
        time that the root lives before any divisions occur in the phyologeny,
        and is useful in instances where the branch lengths are critical, like
        simulating ground truth phylogenies or estimating branch lengths.

        Args:
            tree: The CassiopeiaTree for which to subsample leaves
            keep_singular_root_edge: Whether or not to collapse the single edge
                leading from the root in the subsample, if it exists

        Returns
        -------
            A new CassiopeiaTree that is the induced subtree on a sample of the
                leaves in the given tree

        Raises
        ------
            LeafSubsamplerError if the sample size is <= 0, or larger than the
                number of leaves in the tree
        """
        ratio = self.__ratio
        number_of_leaves = self.__number_of_leaves
        n_subsample = (
            number_of_leaves
            if number_of_leaves is not None
            else int(tree.n_cell * ratio)
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
        subsampled_tree = copy.deepcopy(tree)
        leaf_remove = np.random.choice(
            subsampled_tree.leaves, n_remove, replace=False
        )

        subsampled_tree.remove_leaves_and_prune_lineages(leaf_remove)

        # Keep the singular root edge if it exists and is indicated to be kept
        if (
            len(subsampled_tree.children(subsampled_tree.root)) == 1
            and keep_singular_root_edge
        ):
            collapse_source = subsampled_tree.children(subsampled_tree.root)[0]
        else:
            collapse_source = None
        subsampled_tree.collapse_unifurcations(source=collapse_source)

        # Copy and annotate branch lengths and times
        subsampled_tree.set_times(
            {
                node: tree.get_time(node) for node in subsampled_tree.nodes
            }
        )

        return subsampled_tree
