"""A library that stores functions for comparing two trees to one another.

Currently, we'll support a triplets correct function and a Robinson-Foulds
function.
"""

import copy
from collections import defaultdict

import ete3
import networkx as nx
import numpy as np
from treedata import TreeData

from cassiopeia.critique.critique_utilities import (
    annotate_tree_depths_nx,
    collapse_unifurcations_nx,
    get_outgroup_nx,
    sample_triplet_at_depth_nx,
    to_nx_tree,
)
from cassiopeia.data import CassiopeiaTree


def triplets_correct(
    tree1: CassiopeiaTree | str | nx.DiGraph,
    tree2: CassiopeiaTree | str | nx.DiGraph,
    tdata: TreeData | None = None,
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
        tree1: First tree (`CassiopeiaTree`, `nx.DiGraph`, or str key into `tdata.obst`).
        tree2: Second tree of the same type as `tree1`.
        tdata: Required only when `tree1` and `tree2` are string keys; must provide `.obst`.
        number_of_trials: Number of triplets to sample at each depth.
        min_triplets_at_depth: Minimum number of triplets with LCA at a depth for that
            depth to be included.

    Returns
    -------
        A tuple of four dicts keyed by depth:
            all_triplets_correct: Total proportion of triplets correct.
            resolvable_triplets_correct: Proportion correct among resolvable triplets.
            unresolved_triplets_correct: Proportion correct among unresolvable triplets.
            proportion_unresolvable: Proportion of triplets that are unresolvable at each depth.
    """
    if type(tree1) is not type(tree2):
        raise TypeError("tree1 and tree2 must be the same type. ")

    if isinstance(tree1, CassiopeiaTree):
        G1 = to_nx_tree(copy.deepcopy(tree1))
        G2 = to_nx_tree(copy.deepcopy(tree2))

    elif isinstance(tree1, str):
        if tdata is None:
            raise ValueError("When tree1 and tree2 are strings, tdata must be provided")
        if not hasattr(tdata, "obst") or tdata.obst is None or len(tdata.obst) == 0:
            raise ValueError("tdata does not have an 'obst' attribute")
        if tree1 not in tdata.obst or tree2 not in tdata.obst:
            raise ValueError(
                f"Tree keys must exist in tdata.obst. Missing: {[k for k in [tree1, tree2] if k not in tdata.obst]}"
            )

        # tdata key triplets correct
        raw1 = tdata.obst[tree1]
        raw2 = tdata.obst[tree2]
        G1 = to_nx_tree(copy.deepcopy(raw1))
        G2 = to_nx_tree(copy.deepcopy(raw2))

    elif isinstance(tree1, nx.DiGraph):
        # nx.DiGraph triplets correct
        G1 = to_nx_tree(copy.deepcopy(tree1))
        G2 = to_nx_tree(copy.deepcopy(tree2))

    else:
        raise TypeError("Unsupported input type. Expected CassiopeiaTree, str (key into tdata.obst), or nx.DiGraph.")

    return run_triplets_correct_nx(
        G1, G2, number_of_trials=number_of_trials, min_triplets_at_depth=min_triplets_at_depth
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

    Returns
    -------
        A tuple of four dicts keyed by depth:
            all_triplets_correct: Total proportion of triplets correct.
            resolvable_triplets_correct: Proportion correct among resolvable triplets.
            unresolved_triplets_correct: Proportion correct among unresolvable triplets.
            proportion_unresolvable: Proportion of triplets that are unresolvable at each depth.
    """
    # collapse unifurcations
    collapse_unifurcations_nx(G1)
    collapse_unifurcations_nx(G2)

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
            (i, j, k), out_group = sample_triplet_at_depth_nx(G1, depth, depth_to_nodes)
            reconstructed = get_outgroup_nx(G2, (i, j, k))

            is_resolvable = out_group != "None"
            if not is_resolvable:
                num_unres += 1

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


# def triplets_correct(
#     tree1: CassiopeiaTree,
#     tree2: CassiopeiaTree,
#     number_of_trials: int = 1000,
#     min_triplets_at_depth: int = 1,
# ) -> tuple[dict[int, float], dict[int, float], dict[int, float], dict[int, float]]:
#     """Calculate the triplets correct accuracy between two trees.

#     Takes in two newick strings and computes the proportion of triplets in the
#     tree (defined as a set of three leaves) that are the same across the two
#     trees. This procedure samples the same number of triplets at every depth
#     such as to reduce the amount of bias of sampling triplets randomly.

#     Args:
#         tree1: Input CassiopeiaTree
#         tree2: CassiopeiaTree to be compared to the first tree.
#         number_of_trials: Number of triplets to sample at each depth
#         min_triplets_at_depth: The minimum number of triplets needed with LCA
#             at a depth for that depth to be included

#     Returns
#     -------
#         A tuple of four dictionaries storing triplet statistics at each depth:
#             all_triplets_correct: Total proportion of triplets correct.
#             resolvable_triplets_correct: Proportion correct among resolvable triplets.
#             unresolved_triplets_correct: Proportion correct among unresolvable triplets.
#             proportion_resolvable: Proportion of triplets that are resolvable at each depth.
#     """
#     # keep dictionary of triplets correct
#     all_triplets_correct = defaultdict(int)
#     unresolved_triplets_correct = defaultdict(int)
#     resolvable_triplets_correct = defaultdict(int)
#     proportion_unresolvable = defaultdict(int)

#     # create copies of the trees and collapse process
#     T1 = copy.deepcopy(tree1)
#     T2 = copy.deepcopy(tree2)

#     T1.collapse_unifurcations()
#     T2.collapse_unifurcations()

#     # set depths in T1 and compute number of triplets that are rooted at
#     # ancestors at each depth
#     depth_to_nodes = critique_utilities.annotate_tree_depths(T1)

#     max_depth = np.max([T1.get_attribute(n, "depth") for n in T1.nodes])
#     for depth in range(max_depth):
#         score = 0
#         number_unresolvable_triplets = 0

#         # check that there are enough triplets at this depth
#         candidate_nodes = depth_to_nodes[depth]
#         total_triplets = sum([T1.get_attribute(v, "number_of_triplets") for v in candidate_nodes])
#         if total_triplets < min_triplets_at_depth:
#             continue

#         for _ in range(number_of_trials):
#             (i, j, k), out_group = critique_utilities.sample_triplet_at_depth(T1, depth, depth_to_nodes)

#             reconstructed_outgroup = critique_utilities.get_outgroup(T2, (i, j, k))

#             is_resolvable = True
#             if out_group == "None":
#                 number_unresolvable_triplets += 1
#                 is_resolvable = False

#             # increment score if the reconstructed outgroup is the same as the
#             # ground truth
#             score = int(reconstructed_outgroup == out_group)

#             all_triplets_correct[depth] += score
#             if is_resolvable:
#                 resolvable_triplets_correct[depth] += score
#             else:
#                 unresolved_triplets_correct[depth] += score

#         all_triplets_correct[depth] /= number_of_trials

#         if number_unresolvable_triplets == 0:
#             unresolved_triplets_correct[depth] = 1.0
#         else:
#             unresolved_triplets_correct[depth] /= number_unresolvable_triplets

#         proportion_unresolvable[depth] = number_unresolvable_triplets / number_of_trials

#         if proportion_unresolvable[depth] < 1:
#             resolvable_triplets_correct[depth] /= number_of_trials - number_unresolvable_triplets
#         else:
#             resolvable_triplets_correct[depth] = 1.0

#     return (
#         all_triplets_correct,
#         resolvable_triplets_correct,
#         unresolved_triplets_correct,
#         proportion_unresolvable,
#     )


def robinson_foulds(tree1: CassiopeiaTree, tree2: CassiopeiaTree) -> tuple[float, float]:
    """Compares two trees with Robinson-Foulds distance.

    Computes the Robinsons-Foulds distance between two trees. Currently, this
    is the unweighted variant as most of the algorithms we use are maximum-
    parsimony based and do not use edge weights. This is mostly just a wrapper
    around the `robinson_foulds` method from Ete3.

    Args:
        tree1: A CassiopeiaTree representing the first tree
        tree2: A CassiopeiaTree representing the second tree

    Returns
    -------
        The Robinson-Foulds distance between the two trees and the maximum
            Robinson-Foulds distance for downstream normalization
    """
    # create copies of the trees and collapse process
    T1 = copy.deepcopy(tree1)
    T2 = copy.deepcopy(tree2)

    # convert to Ete3 trees and collapse unifurcations
    T1.collapse_unifurcations()
    T2.collapse_unifurcations()

    T1 = ete3.Tree(T1.get_newick(), format=1)
    T2 = ete3.Tree(T2.get_newick(), format=1)

    (
        rf,
        rf_max,
        names,
        edges_t1,
        edges_t2,
        discarded_edges_t1,
        discarded_edges_t2,
    ) = T1.robinson_foulds(T2, unrooted_trees=True)

    return rf, rf_max
