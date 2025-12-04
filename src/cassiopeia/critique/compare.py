"""A library that stores functions for comparing two trees to one another.

Currently, we'll support a triplets correct function and a Robinson-Foulds
function.
"""

import copy
from collections import defaultdict

import networkx as nx
import numpy as np

from cassiopeia.critique import critique_utilities
from cassiopeia.data import CassiopeiaTree
from cassiopeia.typing import TreeLike
from cassiopeia.utils import _get_digraph, get_leaves


def triplets_correct(
    tree1: CassiopeiaTree,
    tree2: CassiopeiaTree,
    number_of_trials: int = 1000,
    min_triplets_at_depth: int = 1,
) -> tuple[dict[int, float], dict[int, float], dict[int, float], dict[int, float]]:
    """Calculate the triplets correct accuracy between two trees.

    Takes in two newick strings and computes the proportion of triplets in the
    tree (defined as a set of three leaves) that are the same across the two
    trees. This procedure samples the same number of triplets at every depth
    such as to reduce the amount of bias of sampling triplets randomly.

    Args:
        tree1: Input CassiopeiaTree
        tree2: CassiopeiaTree to be compared to the first tree.
        number_of_trials: Number of triplets to sample at each depth
        min_triplets_at_depth: The minimum number of triplets needed with LCA
            at a depth for that depth to be included

    Returns:
            A tuple of four dictionaries storing triplet statistics at each depth:
            all_triplets_correct: Total proportion of triplets correct.
            resolvable_triplets_correct: Proportion correct among resolvable triplets.
            unresolved_triplets_correct: Proportion correct among unresolvable triplets.
            proportion_resolvable: Proportion of triplets that are resolvable at each depth.
    """
    # keep dictionary of triplets correct
    all_triplets_correct = defaultdict(int)
    unresolved_triplets_correct = defaultdict(int)
    resolvable_triplets_correct = defaultdict(int)
    proportion_unresolvable = defaultdict(int)

    # create copies of the trees and collapse process
    T1 = copy.deepcopy(tree1)
    T2 = copy.deepcopy(tree2)

    T1.collapse_unifurcations()
    T2.collapse_unifurcations()

    # set depths in T1 and compute number of triplets that are rooted at
    # ancestors at each depth
    depth_to_nodes = critique_utilities.annotate_tree_depths(T1)

    max_depth = np.max([T1.get_attribute(n, "depth") for n in T1.nodes])
    for depth in range(max_depth):
        score = 0
        number_unresolvable_triplets = 0

        # check that there are enough triplets at this depth
        candidate_nodes = depth_to_nodes[depth]
        total_triplets = sum([T1.get_attribute(v, "number_of_triplets") for v in candidate_nodes])
        if total_triplets < min_triplets_at_depth:
            continue

        for _ in range(number_of_trials):
            (i, j, k), out_group = critique_utilities.sample_triplet_at_depth(
                T1, depth, depth_to_nodes
            )

            reconstructed_outgroup = critique_utilities.get_outgroup(T2, (i, j, k))

            is_resolvable = True
            if out_group == "None":
                number_unresolvable_triplets += 1
                is_resolvable = False

            # increment score if the reconstructed outgroup is the same as the
            # ground truth
            score = int(reconstructed_outgroup == out_group)

            all_triplets_correct[depth] += score
            if is_resolvable:
                resolvable_triplets_correct[depth] += score
            else:
                unresolved_triplets_correct[depth] += score

        all_triplets_correct[depth] /= number_of_trials

        if number_unresolvable_triplets == 0:
            unresolved_triplets_correct[depth] = 1.0
        else:
            unresolved_triplets_correct[depth] /= number_unresolvable_triplets

        proportion_unresolvable[depth] = number_unresolvable_triplets / number_of_trials

        if proportion_unresolvable[depth] < 1:
            resolvable_triplets_correct[depth] /= number_of_trials - number_unresolvable_triplets
        else:
            resolvable_triplets_correct[depth] = 1.0

    return (
        all_triplets_correct,
        resolvable_triplets_correct,
        unresolved_triplets_correct,
        proportion_unresolvable,
    )


def _robinson_foulds_bitset(tree1: nx.DiGraph, tree2: nx.DiGraph):
    """Compute the unrooted Robinson–Foulds distance using bitsets."""
    leaves1 = sorted([n for n in tree1 if tree1.out_degree(n) == 0])
    leaves2 = sorted([n for n in tree2 if tree2.out_degree(n) == 0])
    if set(leaves1) != set(leaves2):
        raise ValueError("Trees must have identical leaf sets.")

    leaf_index = {leaf: i for i, leaf in enumerate(leaves1)}

    def get_splits(tree, leaf_index):
        """Return a set of canonical bitmasks representing bipartitions."""
        topo = list(nx.topological_sort(tree))
        bitset = {}
        # postorder accumulation of leaf bitsets
        for n in reversed(topo):
            if tree.out_degree(n) == 0:
                bitset[n] = (1 << leaf_index[n]) if n in leaf_index else 0
            else:
                m = 0
                for c in tree.successors(n):
                    m |= bitset[c]
                bitset[n] = m

        all_mask = bitset[topo[0]]
        length = all_mask.bit_count()
        # For unrooted splits, each internal edge defines a bipartition.
        # Canonicalize by mapping each side to min(side, complement),
        # so the split is independent of rooting.
        splits = set()
        for _, c in tree.edges:
            bs = bitset[c]
            k = bs.bit_count()
            # exclude trivial: 1 or length-1 leaves
            if 1 < k < length - 0:  # k<length and k>1
                comp = all_mask ^ bs
                # exclude complement-trivial as well; the test above already ensures k<length
                if comp != 0 and comp != all_mask:
                    splits.add(min(bs, comp))
        return splits

    splits1 = get_splits(tree1, leaf_index)
    splits2 = get_splits(tree2, leaf_index)

    rf = len(splits1.symmetric_difference(splits2))
    max_rf = len(splits1) + len(splits2)
    return rf, max_rf


def robinson_foulds(
    tree1: TreeLike,
    tree2: TreeLike | None = None,
    key1: str | None = None,
    key2: str | None = None,
) -> tuple[float, float]:
    """Compute the Robinson–Foulds distance between two trees.

    Args:
        tree1: The tree object.
        tree2: The tree object to compare against. If ``None``, ``key1`` and ``key2``
            are used to select two trees from the `tree1` object.
        key1: If ``tree1`` is a :class:`treedata.TreeData`, specifies the ``obst`` key to use.
            Only required if multiple trees are present.
        key2: The ``obst`` key to compare against. Selects from ``tree2`` if provided,
            otherwise selects from ``tree1``. Only required if multiple trees are present.

    Returns:
        tuple[float, float]: The Robinson–Foulds distance and the maximum
        possible distance for the pair of trees.
    """
    if tree2 is None and (key1 is None or key2 is None):
        raise ValueError("If tree2 is None, both key1 and key2 must be provided.")
    t1, _ = _get_digraph(tree1, tree_key=key1)
    t2, _ = (
        _get_digraph(tree2, tree_key=key2)
        if tree2 is not None
        else _get_digraph(tree1, tree_key=key2)
    )

    if set(get_leaves(t1)) != set(get_leaves(t2)):
        raise ValueError("Trees must have identical leaf sets.")

    return _robinson_foulds_bitset(t1, t2)
