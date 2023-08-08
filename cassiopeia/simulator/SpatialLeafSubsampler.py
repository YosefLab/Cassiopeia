"""
A subclass of LeafSubsampler, the SpatialLeafSubsampler. 

Uniformly samples leaves within a region of interest of a CassiopeiaTree
with spatial information and produces a new CassiopeiaTree containing the
sampled leaves. 
"""

import copy
import numpy as np
from typing import Optional, List

from cassiopeia.data import CassiopeiaTree
from cassiopeia.simulator.LeafSubsampler import (
    LeafSubsampler,
    LeafSubsamplerError,
)
from cassiopeia.mixins import (
    CassiopeiaTreeError
)

class SpatialLeafSubsampler(LeafSubsampler):
    def __init__(
        self,
        bounding_box: List[tuple] = None,
        number_of_leaves: Optional[int] = None,
        attribute_key: str = "spatial"
    ):
        """
        Subsets the leaves of a CassiopeiaTree to those within a region of
        interest. The region of interest is defined by a bounding box, which
        is a list of tuples of the form (min, max) for each dimension. If 
        'number_of_leaves' is provided, 'number_of_leaves' of the leaves are
        sampled uniformly from within the bounding box.

        Args:
            bounding_box: A list of tuples of the form (min, max) for each
                dimension of the bounding box.
            number_of_leaves: Explicitly specifies the number of leaves to be sampled
            attribute_key: The key in the CassiopeiaTree's node attributes
                that contains the spatial coordinates of the leaves.
        """
        if bounding_box is None:
            raise LeafSubsamplerError(
                "bounding_box must be specified"
            )

        self.__bounding_box = bounding_box
        self.__number_of_leaves = number_of_leaves
        self.__attribute_key = attribute_key

    def subsample_leaves(
        self, tree: CassiopeiaTree, keep_singular_root_edge: bool = True
    ) -> CassiopeiaTree:
        """
        Subsets the leaves of a CassiopeiaTree to those within a region of
        interest and returns a tree pruned to contain lineages relevant
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

        Returns:
            A new CassiopeiaTree that is the induced subtree on a sample of the
                leaves in the given tree

        Raises:
            LeafSubsamplerError if the sample size is <= 0, or larger than the
                number of leaves in the tree
        """
        attribute = self.__attribute_key
        bounding_box = self.__bounding_box
        number_of_leaves = self.__number_of_leaves
        # Check that the tree has spatial information
        try:
            coordinates = tree.get_attribute(tree.leaves[0], attribute)
        except CassiopeiaTreeError:
            raise LeafSubsamplerError(
                f"Attribute {attribute} not present in the tree."
            )
        # Check the dimensions of coordinates and bounding box match
        if len(coordinates) != len(bounding_box):
            raise LeafSubsamplerError(
                f"Dimensions of coordinates ({len(coordinates)}) and "
                f"bounding box ({len(bounding_box)}) do not match."
            )
        # Subset the leaves to those within the bounding box
        leaf_keep = []
        for leaf in tree.leaves:
            coordinates = tree.get_attribute(leaf, attribute)
            for i, (min, max) in enumerate(bounding_box):
                if coordinates[i] < min or coordinates[i] > max:
                    break
            else:
                leaf_keep.append(leaf)
        # Check that the number of leaves to keep is > 0
        if len(leaf_keep) == 0:
            raise LeafSubsamplerError(
                "No leaves within the specified bounding box."
            )
        # If number_of_leaves is specified, check value then subsample
        if number_of_leaves is not None:
            # Check number_of_leaves is > 0
            if number_of_leaves <= 0:
                raise LeafSubsamplerError(
                    "Specified number of leaves sampled is <= 0."
                )
            # Check less than leaf_keep
            if number_of_leaves  > len(leaf_keep):
                raise LeafSubsamplerError(
                    f"Specified number of leaves to sample ({number_of_leaves}) is" 
                    f" greater than the number of leaves within the bounding box"
                    f" {len(leaf_keep)}."
                )
            # sample uniformly from leaf_keep
            if number_of_leaves is not None:
                leaf_keep = np.random.choice(leaf_keep, number_of_leaves, replace=False)
        # Find leaves to remove
        leaf_remove = set(tree.leaves) - set(leaf_keep)
        # Remove leaves and prune lineages
        subsampled_tree = copy.deepcopy(tree)
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
            dict(
                [(node, tree.get_time(node)) for node in subsampled_tree.nodes]
            )
        )
        return subsampled_tree
