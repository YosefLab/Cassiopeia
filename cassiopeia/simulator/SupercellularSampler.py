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
from cassiopeia.simulator.LeafSubsampler import LeafSubsampler, LeafSubsamplerError


class SupercellularSampler(LeafSubsampler):
    def __init__(self, ratio=None, number_of_merges=None):
        """
        Iteratively merge two leaves in a CassiopeiaTree to generate a new
        CassiopeiaTree with ambiguous character states. Note that according to this
        procedure, an already merged (and therefore ambiguous leaf) may be merged
        again.

        Only one of ``ratio`` or ``number_of_merges`` must be provided.

        Args:
            ratio: The number of times to merge as a ratio of the total number
                of leaves. A ratio of 0.5 indicates the number of merges will be
                approximately half the number of trees.
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
        collapse_duplicates: Optional[bool] = True,
    ):
        """Construct a new CassiopeiaTree by merging leaves in the provided
        tree.

        Args:
            tree: The CassiopeiaTree for which to subsample leaves
            collapse_source: The source node from which to collapse unifurcations
            collapse_duplicates: Whether or not to collapse duplicated character
                states, so that only unique character states are present in each
                ambiguous state. Defaults to True.

        """
        n_merges = self.__number_of_merges if self.__number_of_merges is not None else int(tree.n_cell * self.__ratio)
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
            for leaf, distance in distances.items():
                if leaf == leaf1:
                    continue
                leaves.append(leaf)
                weights.append(1 / distance)
            leaf2 = np.random.choice(leaves, p=np.array(weights) / np.sum(weights))
            leaf2_state = merged_tree.get_character_states(leaf2)

            # Merge these two leaves at the mean time of the two leaves
            # If the tree is ultrametric, this preserves ultrametricity
            new_leaf = f'{leaf1}-{leaf2}'
            new_time = (merged_tree.get_time(leaf1) + merged_tree.get_time(leaf2)) / 2
            new_state = []
            for char1, char2 in zip(leaf1_state, leaf2_state):
                new_char = []
                if not isinstance(char1, list):
                    char1 = [char1]
                if not isinstance(char2, list):
                    char2 = [char2]
                new_state.append(char1 + char2)
            merged_tree.add_leaf(merged_tree.find_lca(leaf1, leaf2), new_leaf)
            merged_tree.set_time(new_leaf, new_time)
            merged_tree.set_character_states(new_leaf, new_state)
            merged_tree.remove_leaf_and_prune_lineage(leaf1)
            merged_tree.remove_leaf_and_prune_lineage(leaf2)

        if collapse_source is None:
            collapse_source = merged_tree.root
        merged_tree.collapse_unifurcations(source=collapse_source)

        if collapse_duplicates:
            merged_tree.collapse_ambiguous_characters()

        return merged_tree
