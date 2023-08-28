"""
A subclass of LeafSubsampler, the SpatialLeafSubsampler. 

Uniformly samples leaves within a region of interest of a CassiopeiaTree
with spatial information and produces a new CassiopeiaTree containing the
sampled leaves. 
"""
from typing import Optional, List
import warnings

from collections import defaultdict
import copy
import numpy as np

from cassiopeia.data import CassiopeiaTree
from cassiopeia.mixins import (
    CassiopeiaTreeError,
)
from cassiopeia.simulator.LeafSubsampler import (
    LeafSubsampler,
    LeafSubsamplerError,
    LeafSubsamplerWarning,
)

class SpatialLeafSubsampler(LeafSubsampler):
    def __init__(
        self,
        bounding_box: Optional[List[tuple]] = None,
        space: Optional[np.ndarray] = None,
        ratio: Optional[float] = None,
        number_of_leaves: Optional[int] = None,
        attribute_key: Optional[str] = "spatial",
        merge_cells: Optional[bool] = False
    ):
        """Subsamples leaves within a region of interest.

        Subsets the leaves of a CassiopeiaTree to those within a region of
        interest. The region of interest is defined by a bounding box or numpy
        mask. Both no downsampling and downsampling defined by a ratio or a
        number_of_leaves is supported. If a mask is provided and merge_cells
        is True, cells contained within the same pixel in the space are merged.

        Args:
            bounding_box: A list of tuples of the form (min, max) for each
                dimension of the bounding box.
            space: Numpy array mask representing the space that cells will be 
                sampled from. For example, to sample cells on a 2D circular 
                surface, this argument will be a boolean Numpy array where the 
                circular surface is indicated with True.
            ratio: Specifies the number of leaves to be sampled as a ratio of
                the total number of leaves in the region of interest
            number_of_leaves: Explicitly specifies the number of leaves to be 
                sampled
            attribute_key: The key in the CassiopeiaTree's node attributes
                that contains the spatial coordinates of the leaves.
            merge_cells: Whether or not to merge cells contained within the 
                same pixel in the space. If True, cells are merged to create a
                new cell with ambiguous character states centered at the pixel.

        Raises:
            LeafSubsamplerError if invalid inputs are provided.
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
        self.__merge_cells = merge_cells

    def subsample_leaves(
        self, 
        tree: CassiopeiaTree, 
        keep_singular_root_edge: Optional[bool] = True,
        collapse_duplicates: bool = True,
    ) -> CassiopeiaTree:
        """Subsamples leaves within a given tree.

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
            collapse_duplicates: Whether or not to collapse duplicated character
                states, so that only unique character states are present in each
                ambiguous state. Defaults to True.

        Returns:
            A new CassiopeiaTree that is the induced subtree on a sample of the
                leaves in the given tree

        Raises:
            LeafSubsamplerError if invalid region of number of leaves
        """
        attribute = self.__attribute_key
        bounding_box = self.__bounding_box
        space = self.__space
        ratio = self.__ratio
        number_of_leaves = self.__number_of_leaves
        merge_cells = self.__merge_cells
        tree = copy.deepcopy(tree)

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
            # check that max coordinate is similar to space shape
            max_coordinate = np.max(coordinates)
            max_dimension = np.max(space.shape)
            if (max_coordinate * 10 < max_dimension):
                warnings.warn(
                    f"Maximum coordinate {max_coordinate} is much smaller than "
                    f"maximum dimension of space {max_dimension}. Consider "
                    f"rescaling coordinates since they are converted to "
                    f"integers for spatial down sampling.",
                    LeafSubsamplerWarning
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
        tree.remove_leaves_and_prune_lineages(leaf_remove)

        # Merge cells
        if merge_cells:
            if space is None:
                raise LeafSubsamplerError(
                    "Can not merge cells without space provided."
                )
            
            # Get the coordinates of the leaves
            coordinate_leaves = defaultdict(list)
            for leaf in tree.leaves:
                coordinates = tree.get_attribute(leaf, attribute)
                # round coordinates to nearest integer
                coordinates = [int(c) for c in coordinates]
                coordinate_leaves[tuple(coordinates)].append(leaf)

            # Set character location and character state
            for coordinates, leaves in coordinate_leaves.items():
                if len(leaves) == 1:
                    tree.set_attribute(leaves[0], attribute, coordinates)
                else:
                    new_leaf = "-".join(leaves)
                    lca = tree.find_lca(*leaves)
                    # set new time to average of leaf times
                    new_time = np.mean([tree.get_time(leaf) for leaf in leaves])
                    # set new character state
                    new_state = []
                    for i in range(len(tree.get_character_states(leaves[0]))):
                        new_char = []
                        for leaf in leaves:
                            new_char.append(tree.get_character_states(leaf)[i])
                        new_state.append(tuple(new_char))
                    # update the tree
                    tree.add_leaf(lca, new_leaf,states=new_state,time=new_time)
                    tree.set_attribute(new_leaf, attribute, coordinates)  
                    tree.remove_leaves_and_prune_lineages(leaves)

            # Collapse duplicates
            if collapse_duplicates and tree.character_matrix is not None:
                tree.collapse_ambiguous_characters()

        # Keep the singular root edge if it exists and is indicated to be kept
        if (
            len(tree.children(tree.root)) == 1
            and keep_singular_root_edge
        ):
            collapse_source = tree.children(tree.root)[0]
        else:
            collapse_source = None
        tree.collapse_unifurcations(source=collapse_source)
        
        # Copy and annotate branch lengths and times
        tree.set_times(
            dict(
                [(node, tree.get_time(node)) for node in tree.nodes]
            )
        )

        return tree
