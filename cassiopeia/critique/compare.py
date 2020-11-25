"""
A library that stores functions for comparing two trees to one another.
Currently, we'll support a triplets correct function and a Robinson-Foulds
function.
"""
from collections import defaultdict
import ete3
import networkx as nx
import numpy as np
from typing import Dict, Tuple

from cassiopeia.critique import critique_utilities
from cassiopeia.solver import solver_utilities


def triplets_correct(
    tree1: nx.DiGraph, tree2: nx.DiGraph, number_of_trials: int = 1000
) -> Tuple[Dict[int, float], Dict[int, float], Dict[int, float], Dict[int,float]]:
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

    # convert to Ete3 trees and collapse unifurcations
    T1 = ete3.Tree(solver_utilities.to_newick(tree1), format=1)
    T2 = ete3.Tree(solver_utilities.to_newick(tree2), format=1)

    T1 = solver_utilities.collapse_unifurcations(T1)
    T2 = solver_utilities.collapse_unifurcations(T2)

    # set depths in T1 and T2 and compute number of triplets that are rooted at
    # ancestors at each depth
    critique_utilities.annotate_tree_depths(T1)
    critique_utilities.annotate_tree_depths(T2)

    max_depth = np.max([n.depth for n in T1])
    for depth in range(max_depth):

        score = 0
        number_unresolvable_triplets = 0

        # check that there are triplets at this depth
        candidate_nodes = T1.search_nodes(depth=depth)
        total_triplets = sum([v.number_of_triplets for v in candidate_nodes])
        if total_triplets == 0:
            continue
   

        for _ in range(number_of_trials):

            (i, j, k), out_group = critique_utilities.sample_triplet_at_depth(
                T1, depth
            )
            ij_lca = T2.get_common_ancestor(i, j)
            ik_lca = T2.get_common_ancestor(i, k)
            jk_lca = T2.get_common_ancestor(j, k)

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
            if ij_lca.depth > jk_lca.depth and ij_lca.depth > ik_lca.depth:
                score = int(k == out_group)
            elif ik_lca.depth > ij_lca.depth and ik_lca.depth > jk_lca.depth:
                score = int(j == out_group)
            elif jk_lca.depth > ik_lca.depth and jk_lca.depth > ij_lca.depth:
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
