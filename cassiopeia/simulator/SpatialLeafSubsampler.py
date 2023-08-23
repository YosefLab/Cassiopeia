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
        space: Optional[np.ndarray] = None,
        ratio: Optional[float] = None,
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
            space: Numpy array mask representing the space that cells will be 
                sampled from. For example, to sample cells on a 2D circlular 
                surface, this argument will be a boolean Numpy array where the 
                circular surface is indicated with True.
            ratio: Specifies the number of leaves to be sampled as a ratio of
                the total number of leaves in the region of interest
            number_of_leaves: Explicitly specifies the number of leaves to be 
                sampled
            attribute_key: The key in the CassiopeiaTree's node attributes
                that contains the spatial coordinates of the leaves.
        """
        # check that exactly one of bounding_box or space is provided
        if (bounding_box is None) == (space is None):
            raise LeafSubsamplerError(
                "Exactly one of `bounding_box` or `space` must be provided."
            )
        # check that not both ratio and number_of_leaves are provided
        if ratio is not None and number_of_leaves is not None:
            raise LeafSubsamplerError(
                "Can not specify both `ratio` and `number_of_leaves`"
            )
        if number_of_leaves is not None:
            # Check number_of_leaves is > 0
            if number_of_leaves <= 0:
                raise LeafSubsamplerError(
                    "Specified number of leaves sampled is <= 0."
                )
        if ratio is not None:
            # Check ratio is between 0 and 1
            if ratio <= 0 or ratio > 1:
                raise LeafSubsamplerError(
                    "Specified ratio is <= 0 or > 1."
                )
        self.__bounding_box = bounding_box
        self.__space = space
        self.__ratio = ratio
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
        space = self.__space
        ratio = self.__ratio
        number_of_leaves = self.__number_of_leaves

        # Check that the tree has spatial information
        try:
            coordinates = tree.get_attribute(tree.leaves[0], attribute)
        except CassiopeiaTreeError:
            raise LeafSubsamplerError(
                f"Attribute {attribute} not present in the tree."
            )
        
        if bounding_box is not None:
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
            
        if space is not None:
            # Check the dimensions of coordinates and space match
            if len(coordinates) != len(space.shape):
                raise LeafSubsamplerError(
                    f"Dimensions of coordinates ({len(coordinates)}) and "
                    f"space ({len(space.shape)}) do not match."
                )
            # Subset the leaves to those within the space
            leaf_keep = []
            for leaf in tree.leaves:
                coordinates = tree.get_attribute(leaf, attribute)
                # round coordinates to nearest integer
                coordinates = [int(c) for c in coordinates]
                # check if coordinates are within space size
                if any([c < 0 or c >= s for c, s in 
                        zip(coordinates, space.shape)]):
                    raise LeafSubsamplerError(
                        f"Coordinates {coordinates} are outside the space."
                    )
                # check if coordinates are in space
                if not space[tuple(coordinates)]:
                    continue
                leaf_keep.append(leaf)

        if ratio is not None:
            number_of_leaves = int(len(leaf_keep) * ratio)
        if number_of_leaves is None:
            number_of_leaves = len(leaf_keep)

        # Check that the number of leaves to keep is > 0
        if len(leaf_keep) == 0:
            raise LeafSubsamplerError(
                "No leaves within the specified region."
            )

         # Check less than leaf_keep
        if number_of_leaves  > len(leaf_keep):
            raise LeafSubsamplerError(
                f"Specified number of leaves to sample ({number_of_leaves}) is" 
                f" greater than the number of leaves within the region of "
                f"interest {len(leaf_keep)}."
            )
           
        # sample uniformly from leaf_keep
        if number_of_leaves is not None:
            leaf_keep = np.random.choice(leaf_keep, number_of_leaves, 
                                         replace=False)

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