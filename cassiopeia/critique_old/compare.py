"""
A library that stores functions for comparing two trees to one another.
Currently, we'll support a triplets correct function and a Robinson-Foulds
function.
"""
from collections import defaultdict
import copy
import ete3
import networkx as nx
import numpy as np
from typing import Dict, Tuple

from cassiopeia.critique_old import critique_utilities
from cassiopeia.data import CassiopeiaTree

def triplets_correct(
    tree1: nx.DiGraph, tree2: nx.DiGraph, number_of_trials: int = 1000
) -> Tuple[
    Dict[int, float], Dict[int, float], Dict[int, float], Dict[int, float]
]:
    """Calculate the triplets correct accuracy between two trees.
    Takes in two newick strings and computes the proportion of triplets in the
    tree (defined as a set of three leaves) that are the same across the two
    trees. This procedure samples the same number of triplets at every depth
    such as to reduce the amount of bias of sampling triplets randomly.
    Args:
        tree1: A graph representing the first tree
        tree2: A graph representing the second tree
        number_of_trials: Number of triplets to sample at each depth
    Returns:
        Four dictionaries storing triplet information at each depth:
            all_triplets_correct: the total triplets correct
            resolvable_triplets_correct: the triplets correct for resolvable
                triplets
            unresolved_triplets_correct: the triplets correct for unresolvable
                triplets
            proportion_resolvable: the proportion of unresolvable triplets per
                depth
    """

    # keep dictionary of triplets correct
    all_triplets_correct = defaultdict(int)
    unresolved_triplets_correct = defaultdict(int)
    resolvable_triplets_correct = defaultdict(int)
    proportion_unresolvable = defaultdict(int)

    # create copies of the trees and collapse process
    T1 = copy.copy(tree1)
    T2 = copy.copy(tree2)

    T1.collapse_unifurcations()
    T2.collapse_unifurcations()

    T1 = T1.get_tree_topology()
    T2 = T2.get_tree_topology()

    # set depths in T1 and T2 and compute number of triplets that are rooted at
    # ancestors at each depth
    leaf_children, nodes_at_depth = critique_utilities.annotate_tree(T1)

    for depth in range(max(nodes_at_depth)):

        number_unresolvable_triplets = 0

        # check that there are triplets at this depth

        candidate_nodes = nodes_at_depth[depth]
        total_triplets = sum([T1.nodes[n]["number_of_triplets"] for n in candidate_nodes])
        if total_triplets == 0:
            continue

        for _ in range(number_of_trials):
            (i, j, k), out_group_T1 = critique_utilities.sample_triplet_at_depth(
                T1, candidate_nodes, leaf_children, total_triplets
            )

            out_group_T2 = critique_utilities.get_outgroup((i, j, k), T2)

            is_resolvable = True
            if out_group_T1 == "None":
                number_unresolvable_triplets += 1
                is_resolvable = False

            # find outgroup based on the depth of the latest-common-ancestors
            # of each pair of items. The pair with the deepest LCA is the
            # ingroup and the remaining leaf is the outgroup. The score is
            # incremented if the compared tree (T2) has the same outgroup as
            # T1.
            score = int(out_group_T1 == out_group_T2)

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

        proportion_unresolvable[depth] = (
            number_unresolvable_triplets / number_of_trials
        )

        if proportion_unresolvable[depth] < 1:
            resolvable_triplets_correct[depth] /= (
                number_of_trials - number_unresolvable_triplets
            )
        else:
            resolvable_triplets_correct[depth] = 1.0

    return (
        all_triplets_correct,
        resolvable_triplets_correct,
        unresolved_triplets_correct,
        proportion_unresolvable,
    )


def robinson_foulds(
    tree1: nx.DiGraph, tree2: nx.DiGraph
) -> Tuple[float, float]:
    """Compares two trees with Robinson-Foulds distance.

    Computes the Robinsons-Foulds distance between two trees. Currently, this
    is the unweighted variant as most of the algorithms we use are maximum-
    parsimony based and do not use edge weights. This is mostly just a wrapper
    around the `robinson_foulds` method from Ete3.

    Args:
        tree1: A graph representing the first tree
        tree2: A graph representing the second tree

    Returns:
        The Robinson-Foulds distance between the two trees and the maximum
            Robinson-Foulds distance for downstream normalization
    """

    # convert to Ete3 trees and collapse unifurcations
    tree1.collapse_unifurcations()
    tree2.collapse_unifurcations()

    T1 = ete3.Tree(tree1.get_newick(), format=1)
    T2 = ete3.Tree(tree2.get_newick(), format=1)

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
