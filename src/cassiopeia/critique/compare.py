"""A library that stores functions for comparing two trees to one another.

Currently, we'll support a triplets correct function and a Robinson-Foulds
function.
"""

from collections import defaultdict

import networkx as nx
import numpy as np

from cassiopeia.critique.critique_utilities import (
    annotate_tree_depths_nx,
    get_outgroup_nx,
    sample_triplet_at_depth,
)
from cassiopeia.typing import TreeLike
from cassiopeia.utils import (
    _get_digraph,
    collapse_unifurcations,
    get_leaves,
)


def triplets_correct(
    tree1: TreeLike,
    tree2: TreeLike | None = None,
    key1: str | None = None,
    key2: str | None = None,
    number_of_trials: int = 1000,
    min_triplets_at_depth: int = 1,
) -> tuple[dict[int, float], dict[int, float], dict[int, float], dict[int, float]]:
    """Calculate triplets-correct accuracy between two trees.

    Accepts inputs as `CassiopeiaTree`, `networkx.DiGraph` (rooted, edges parent→child),
    or string keys referencing trees stored in `tdata.obst`. Computes the proportion of
    triplets (sets of three leaves) that agree between the two trees. Sampling is
    stratified by depth (same number of triplets per depth) to reduce bias from
    purely random triplet sampling. Unifurcations are collapsed before evaluation.
    Trees must have identical leaf sets.

    Args:
        tree1: The tree object.
        tree2: The tree object to compare against. If ``None``, ``key1`` and ``key2``
            are used to select two trees from the `tree1` object.
        key1: If ``tree1`` is a :class:`treedata.TreeData`, specifies the ``obst`` key to use.
            Only required if multiple trees are present.
        key2: The ``obst`` key to compare against. Selects from ``tree2`` if provided,
            otherwise selects from ``tree1``. Only required if multiple trees are present.
        number_of_trials: Number of triplets to sample at each depth.
        min_triplets_at_depth: Minimum number of triplets with LCA at a depth for that
            depth to be included.

    Returns:
            A tuple of four dictionaries storing triplet statistics at each depth:
            all_triplets_correct: Total proportion of triplets correct.
            resolvable_triplets_correct: Proportion correct among resolvable triplets.
            unresolved_triplets_correct: Proportion correct among unresolvable triplets.
            proportion_unresolvable: Proportion of triplets that are unresolvable at each depth.
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
    if type(tree1) is not type(tree2):
        raise TypeError("tree1 and tree2 must be the same type. ")

    return run_triplets_correct_nx(
        t1, t2, number_of_trials=number_of_trials, min_triplets_at_depth=min_triplets_at_depth
    )


def run_triplets_correct_nx(
    G1: nx.DiGraph,
    G2: nx.DiGraph,
    *,
    number_of_trials: int,
    min_triplets_at_depth: int,
) -> tuple[dict[int, float], dict[int, float], dict[int, float], dict[int, float]]:
    """Compute depth-stratified triplets-correct metrics on two rooted nx.DiGraph trees.

    This is the NX backend used by `triplets_correct`.

    Args:
        G1: First rooted tree as an `nx.DiGraph` (edges parent→child).
        G2: Second rooted tree as an `nx.DiGraph` (edges parent→child).
        number_of_trials: Number of triplets to sample at each depth.
        min_triplets_at_depth: Minimum number of triplets with LCA at a depth for that
            depth to be included.

    Returns:
        A tuple of four dicts keyed by depth:
            all_triplets_correct: Total proportion of triplets correct.
            resolvable_triplets_correct: Proportion correct among resolvable triplets.
            unresolved_triplets_correct: Proportion correct among unresolvable triplets.
            proportion_unresolvable: Proportion of triplets that are unresolvable at each depth.
    """
    # collapse unifurcations
    collapse_unifurcations(G1)
    collapse_unifurcations(G2)

    # require identical leaf sets
    leaves1 = {n for n in G1.nodes if G1.out_degree(n) == 0}
    leaves2 = {n for n in G2.nodes if G2.out_degree(n) == 0}
    if leaves1 != leaves2:
        raise ValueError("Trees must have identical leaf sets to compare triplets.")

    # annotate depths and per-node triplet counts (on G1)
    depth_to_nodes = annotate_tree_depths_nx(G1)

    # max depth from G1
    depths = [d for _, d in G1.nodes(data="depth") if d is not None]
    if not depths:
        return ({}, {}, {}, {})

    max_depth = int(np.max(depths))

    all_triplets_correct: dict[int, float] = defaultdict(float)
    unresolved_triplets_correct: dict[int, float] = defaultdict(float)
    resolvable_triplets_correct: dict[int, float] = defaultdict(float)
    proportion_unresolvable: dict[int, float] = defaultdict(float)

    for depth in range(max_depth + 1):
        candidate_nodes = depth_to_nodes.get(depth, [])
        total_triplets = int(sum(G1.nodes[v].get("number_of_triplets", 0) for v in candidate_nodes))
        if total_triplets < min_triplets_at_depth:
            continue

        score_sum = 0
        res_sum = 0
        unres_sum = 0
        num_unres = 0

        for _ in range(number_of_trials):
            (i, j, k), out_group = sample_triplet_at_depth(G1, depth, depth_to_nodes)

            is_resolvable = out_group != "None"
            if not is_resolvable:
                num_unres += 1

            reconstructed = get_outgroup_nx(G2, i, j, k)
            score = int(reconstructed == out_group)
            score_sum += score
            if is_resolvable:
                res_sum += score
            else:
                unres_sum += score

        all_triplets_correct[depth] = score_sum / number_of_trials
        proportion_unresolvable[depth] = num_unres / number_of_trials

        if num_unres == 0:
            unresolved_triplets_correct[depth] = 1.0
        else:
            unresolved_triplets_correct[depth] = unres_sum / num_unres

        if proportion_unresolvable[depth] < 1.0:
            denom = number_of_trials - num_unres
            resolvable_triplets_correct[depth] = res_sum / denom
        else:
            resolvable_triplets_correct[depth] = 1.0

    return (
        dict(all_triplets_correct),
        dict(resolvable_triplets_correct),
        dict(unresolved_triplets_correct),
        dict(proportion_unresolvable),
    )


def _robinson_foulds_bitset(tree1: nx.DiGraph, tree2: nx.DiGraph):
    """Compute the unrooted Robinson–Foulds distance using bitsets."""
    leaves1 = sorted([n for n in tree1 if tree1.degree[n] == 1])
    leaves2 = sorted([n for n in tree2 if tree2.degree[n] == 1])
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
