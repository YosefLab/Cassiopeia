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

from cassiopeia.critique import critique_utilities
from cassiopeia.data import CassiopeiaTree


def triplets_correct(
    tree1: CassiopeiaTree,
    tree2: CassiopeiaTree,
    number_of_trials: int = 1000,
    min_triplets_at_depth: int = 1,
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
        min_triplets_at_depth: The minimum number of triplets needed with LCA
            at a depth for that depth to be included

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
        total_triplets = sum(
            [T1.get_attribute(v, "number_of_triplets") for v in candidate_nodes]
        )
        if total_triplets < min_triplets_at_depth:
            continue

        for _ in range(number_of_trials):

            (i, j, k), out_group = critique_utilities.sample_triplet_at_depth(
                T1, depth, depth_to_nodes
            )

            reconstructed_outgroup = critique_utilities.get_outgroup(
                T2, (i, j, k)
            )

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
    tree1: CassiopeiaTree, tree2: CassiopeiaTree
) -> Tuple[float, float]:
    """Compares two trees with Robinson-Foulds distance.

    Computes the Robinsons-Foulds distance between two trees. Currently, this
    is the unweighted variant as most of the algorithms we use are maximum-
    parsimony based and do not use edge weights. This is mostly just a wrapper
    around the `robinson_foulds` method from Ete3.

    Args:
        tree1: A CassiopeiaTree representing the first tree
        tree2: A CassiopeiaTree representing the second tree

    Returns:
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
