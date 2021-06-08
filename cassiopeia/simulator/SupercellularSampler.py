"""
A subclass of LeafSubsampler, the SupercellularSampler.

Iteratively, this subsampler randomly merges two leaves to generate a tree with
ambiguous character states. The probability that two leaves will be merged is
proportional to their branch distance.
"""
import copy
from typing import Optional

import numpy as np

from cassiopeia.data import CassiopeiaTree
from cassiopeia.simulator.LeafSubsampler import (
    LeafSubsampler,
    LeafSubsamplerError,
)


class SupercellularSampler(LeafSubsampler):
    def __init__(
        self,
        ratio: Optional[float] = None,
        number_of_merges: Optional[float] = None,
    ):
        """
        Merge leaves in a tree to generate a new tree with ambiguous states.

        Note that according to this procedure, an already merged (and therefore
        ambiguous leaf) may be merged again.

        Only one of ``ratio`` or ``number_of_merges`` must be provided.

        Args:
            ratio: The number of times to merge as a ratio of the total number
                of leaves. A ratio of 0.5 indicates the number of merges will be
                approximately half the number of leaves.
            number_of_merges: Explicit number of merges to perform.
        """
        if (ratio is None) == (number_of_merges is None):
            raise LeafSubsamplerError(
                "Exactly one of 'ratio' and 'number_of_merges' must be specified."
            )

        self.__ratio = ratio
        self.__number_of_merges = number_of_merges

    def subsample_leaves(
        self,
        tree: CassiopeiaTree,
        collapse_source: Optional[str] = None,
        collapse_duplicates: bool = True,
    ):
        """Construct a new CassiopeiaTree by merging leaves.

        Pairs of leaves in the given tree is iteratively merged until a stopping
        condition is met (specified by ``ratio`` or ``number_of_merges`` when
        initializing the sampler). Pairs of leaves are selected by the following
        procedure:
        1) A random leaf `A` is selected.
        2) Its pair `B` is randomly selected with probability inversely
           proportional to the branch distance from the leaf selected in the
           previous step.
        3) The pair is merged into a new leaf with name `A-B` and character
           states merged in to ambiguous states. The new leaf is connected to
           the LCA of the two leaves and at time max(time of LCA, mean time of
           the two leaves).

        Args:
            tree: The CassiopeiaTree for which to subsample leaves
            collapse_source: The source node from which to collapse unifurcations
            collapse_duplicates: Whether or not to collapse duplicated character
                states, so that only unique character states are present in each
                ambiguous state. Defaults to True.

        Raises:
            LeafSubsamplerError if the number of merges exceeds the number of
                leaves in the tree or no merges will be performed.
        """
        n_merges = (
            self.__number_of_merges
            if self.__number_of_merges is not None
            else int(tree.n_cell * self.__ratio)
        )
        if n_merges >= len(tree.leaves):
            raise LeafSubsamplerError(
                "Number of required merges exceeds number of leaves in the tree."
            )
        if n_merges == 0:
            raise LeafSubsamplerError("No merges to be performed.")
        # Tree needs to have character matrix defined
        tree.get_current_character_matrix()

        merged_tree = copy.deepcopy(tree)
        for _ in range(n_merges):
            # Choose first leaf
            leaf1 = np.random.choice(merged_tree.leaves)
            leaf1_state = merged_tree.get_character_states(leaf1)

            # Choose second leaf with weight proportional to inverse distance
            distances = merged_tree.get_distances(leaf1, leaves_only=True)
            leaves = []
            weights = []
            for leaf in sorted(distances.keys()):
                if leaf == leaf1:
                    continue
                leaves.append(leaf)
                weights.append(1 / distances[leaf])
            leaf2 = np.random.choice(
                leaves, p=np.array(weights) / np.sum(weights)
            )

            leaf2_state = merged_tree.get_character_states(leaf2)

            # Merge these two leaves at the mean time of the two leaves.
            # Note that the mean time of the two leaves may never be earlier than
            # the LCA time, because each of the leaf times must be greater than or
            # equal to the LCA time.
            # If the tree is ultrametric, this preserves ultrametricity.
            new_leaf = f"{leaf1}-{leaf2}"
            lca = merged_tree.find_lca(leaf1, leaf2)
            new_time = (
                merged_tree.get_time(leaf1) + merged_tree.get_time(leaf2)
            ) / 2
            new_state = []
            for char1, char2 in zip(leaf1_state, leaf2_state):
                new_char = []
                if not isinstance(char1, tuple):
                    char1 = (char1,)
                if not isinstance(char2, tuple):
                    char2 = (char2,)
                new_state.append(char1 + char2)
            merged_tree.add_leaf(lca, new_leaf, states=new_state, time=new_time)
            merged_tree.remove_leaf_and_prune_lineage(leaf1)
            merged_tree.remove_leaf_and_prune_lineage(leaf2)

        if collapse_source is None:
            collapse_source = merged_tree.root
        merged_tree.collapse_unifurcations(source=collapse_source)

        if collapse_duplicates:
            merged_tree.collapse_ambiguous_characters()

        return merged_tree
