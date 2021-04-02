"""
A library that stores functions for comparing two trees to one another.
Currently, we'll support a triplets correct function and a Robinson-Foulds
function.
"""
from collections import defaultdict
import copy
import ete3
import itertools
import networkx as nx
import numpy as np
from typing import Dict, Tuple

from cassiopeia.critique import critique_utilities
from cassiopeia.data import CassiopeiaTree


def triplets_correct(
    tree1: CassiopeiaTree, tree2: CassiopeiaTree, number_of_trials: int = 1000
) -> Tuple[
    Dict[int, float], Dict[int, float], Dict[int, float], Dict[int, float]
]:
    """Calculate the triplets correct accuracy between two trees.

    Takes in two newick strings and computes the proportion of triplets in the 
    tree (defined as a set of three leaves) that are the same across the two
    trees. This procedure samples the same number of triplets at every depth
    such as to reduce the amount of bias of sampling triplets randomly.

    Args:
        tree1: Input CassiopeiaTree
        tree2: CassiopeiaTree to be compared to the first tree.
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

    # set depths in T1 and T2 and compute number of triplets that are rooted at
    # ancestors at each depth
    critique_utilities.annotate_tree_depths(T1)
    critique_utilities.annotate_tree_depths(T2)

    max_depth = np.max([T1.get_attribute(n, "depth") for n in T1.nodes])
    for depth in range(max_depth):

        score = 0
        number_unresolvable_triplets = 0

        # check that there are triplets at this depth
        candidate_nodes = T1.filter_nodes(lambda x: T1.get_attribute(x, "depth") == depth)
        total_triplets = sum([T1.get_attribute(v, "number_of_triplets") for v in candidate_nodes])
        if total_triplets == 0:
            continue

        # precompute all LCAs for T2
        lca_dictionary = {}
        lcas = T2.find_lcas_of_pairs(itertools.combinations(T2.leaves, 2))

        for lca in lcas:
            lca_dictionary[tuple(sorted(lca[0]))] = lca[1]

        for _ in range(number_of_trials):

            (i, j, k), out_group = critique_utilities.sample_triplet_at_depth(
                T1, depth
            )

            ij_lca = lca_dictionary[tuple(sorted((i, j)))]
            ik_lca = lca_dictionary[tuple(sorted((i, k)))]
            jk_lca = lca_dictionary[tuple(sorted((j, k)))]

            is_resolvable = True
            if out_group == "None":
                number_unresolvable_triplets += 1
                is_resolvable = False

            # find outgroup based on the depth of the latest-common-ancestors
            # of each pair of items. The pair with the deepest LCA is the
            # ingroup and the remaining leaf is the outgroup. The score is
            # incremented if the compared tree (T2) has the same outgroup as
            # T1.
            score = 0

            ij_lca_depth = T2.get_attribute(ij_lca, "depth")
            jk_lca_depth = T2.get_attribute(jk_lca, "depth")
            ik_lca_depth = T2.get_attribute(ik_lca, "depth")

            if ij_lca_depth > jk_lca_depth and ij_lca_depth > ik_lca_depth:
                score = int(k == out_group)
            elif ik_lca_depth > ij_lca_depth and ik_lca_depth > jk_lca_depth:
                score = int(j == out_group)
            elif jk_lca_depth > ik_lca_depth and jk_lca_depth > ij_lca_depth:
                score = int(i == out_group)
            else:
                score = int("None" == out_group)

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
