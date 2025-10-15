"""
A library that stores functions for comparing two trees to one another.

Currently, we'll support a triplets correct function and a Robinson-Foulds
function.
"""

import copy
from collections import defaultdict

import networkx as nx
import numpy as np
from treedata import TreeData

from cassiopeia.critique import critique_utilities
from cassiopeia.data import CassiopeiaTree


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

    Returns
    -------
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
            (i, j, k), out_group = critique_utilities.sample_triplet_at_depth(T1, depth, depth_to_nodes)

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
    """
    Compute the unrooted Robinson–Foulds distance between two trees using bitset encoding.

    Trees must be connected, acyclic, and have the same leaf labels.

    Args:
        tree1: An nx.DiGraph representing the first tree
        tree2: An nx.DiGraph representing the second tree
    """
    tree1_undirected = tree1.to_undirected()
    tree2_undirected = tree2.to_undirected()
    # --- Helper: map leaves to bit positions ---
    leaves = sorted([n for n in tree1_undirected if tree1_undirected.degree[n] == 1])
    leaves2 = sorted([n for n in tree2_undirected if tree2_undirected.degree[n] == 1])
    if set(leaves) != set(leaves2):
        raise ValueError("Trees must have identical leaf sets.")

    leaf_index = {leaf: i for i, leaf in enumerate(leaves)}
    n_leaves = len(leaves)

    def get_splits(tree):
        """Return a set of canonical bitmasks representing bipartitions."""
        splits = set()

        root = next(n for n in tree.nodes() if tree.degree[n] > 1)

        # Post-order DFS to compute leaf sets for each subtree
        visited = set()
        leaf_sets = {}  # Maps each node to its descendant leaf bitmask

        def dfs(node, parent=None):
            visited.add(node)

            # If it's a leaf, return its bitmask
            if node in leaf_index:
                mask = 1 << leaf_index[node]
                leaf_sets[node] = mask
                return mask

            # Otherwise, combine children's leaf sets
            mask = 0
            for neighbor in tree.neighbors(node):
                if neighbor != parent and neighbor not in visited:
                    child_mask = dfs(neighbor, node)
                    mask |= child_mask

            leaf_sets[node] = mask

            # This edge creates a split: mask vs its complement
            if parent is not None and mask != 0:
                complement = ((1 << n_leaves) - 1) ^ mask

                # Skip trivial splits
                if bin(mask).count("1") > 1 and bin(complement).count("1") > 1:
                    # Canonicalize
                    part = min(mask, complement)
                    splits.add(part)

            return mask

        dfs(root)
        return splits

    splits1 = get_splits(tree1_undirected)
    splits2 = get_splits(tree2_undirected)

    rf = len(splits1.symmetric_difference(splits2))
    return rf, splits1, splits2


def robinson_foulds(
    tree1: CassiopeiaTree | str | nx.DiGraph, tree2: CassiopeiaTree | str | nx.DiGraph, tdata: TreeData | None = None
) -> tuple[float, float]:
    """Compares two trees with Robinson-Foulds distance.

    Computes the Robinsons-Foulds distance between two trees. Currently, this
    is the unweighted variant as most of the algorithms we use are maximum-
    parsimony based and do not use edge weights.
    Args:
        tree1: The first tree. Can be one of:
            - CassiopeiaTree: A Cassiopeia tree object
            - str: Key to look up tree in tdata.obst
            - nx.DiGraph: A NetworkX directed graph
        tree2: The second tree. Must be the same type as tree1.
        tdata: TreeData object containing trees in obst attribute. Required when tree1 and tree2 are strings.

    Returns
    -------
        The Robinson-Foulds distance between the two trees and the maximum
            Robinson-Foulds distance for downstream normalization
    """
    # argument logic
    if type(tree1) is not type(tree2):
        raise TypeError("tree1 and tree2 must be the same type. ")

    if isinstance(tree1, CassiopeiaTree):
        T1 = tree1.get_tree_topology()
        T2 = tree2.get_tree_topology()

    elif isinstance(tree1, str):
        if tdata is None:
            raise ValueError("When tree1 and tree2 are strings, tdata must be provided")
        if not hasattr(tdata, "obst") or tdata.obst is None:
            raise ValueError("tdata does not have an 'obst' attribute")
        if tree1 not in tdata.obst or tree2 not in tdata.obst:
            raise ValueError(
                f"Tree keys must exist in tdata.obst. Missing: {[k for k in [tree1, tree2] if k not in tdata.obst]}"
            )

        T1 = tdata.obst[tree1]
        T2 = tdata.obst[tree2]

    elif isinstance(tree1, nx.DiGraph):
        T1 = tree1
        T2 = tree2

    else:
        raise TypeError("Unsupported tree type")

    rf, splits1, splits2 = _robinson_foulds_bitset(T1, T2)
    max_rf = len(splits1) + len(splits2)  # Maximum possible RF distance

    return rf, max_rf
