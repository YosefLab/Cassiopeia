"""
Utilities for the critique module.
"""

import ete3
import itertools
import numpy as np
from scipy import special
from typing import List, Tuple


def annotate_tree_depths(tree: ete3.Tree) -> None:
    """Annotates tree depth at every node.

    Adds two attributes to the tree: how far away each node is from the root of
    the tree and how many triplets are rooted at that node. Modifies the tree
    in place.

    Args:
        tree: An ete3 Tree
    """
    for n in tree.traverse():
        if n.is_root():
            n.depth = 0
        else:
            n.depth = n.up.depth + 1

        number_of_leaves = 0
        correction = 0
        for child in n.children:
            number_of_leaves += len(child)
            correction += special.comb(len(child), 3)

        n.number_of_triplets = special.comb(number_of_leaves, 3) - correction


def sample_triplet_at_depth(
    tree: ete3.Tree, depth: int
) -> Tuple[List[int], str]:
    """Samples a triplet at a given depth.

    Samples a triplet of leaves such that the depth of the LCA of the triplet
    is at the specified depth. 

    Args:
        tree: An ete3 Tree object
        depth: Depth at which to sample the triplet

    Returns:
        A list of three leaves corresponding to the triplet name of the outgroup
            of the triplet.
    """

    candidate_nodes = tree.search_nodes(depth=depth)
    total_triplets = sum([v.number_of_triplets for v in candidate_nodes])

    # sample a  node from this depth with probability proportional to the number
    # of triplets underneath it
    probs = [v.number_of_triplets / total_triplets for v in candidate_nodes]
    node = np.random.choice(candidate_nodes, size=1, replace=False, p=probs)[0]

    # Generate the probilities to sample each combination of 3 daughter clades
    # to sample from, proportional to the number of triplets in each daughter
    # clade. Choices include all ways to choose 3 different daughter clades
    # or 2 from one daughter clade and one from another
    probs = []
    combos = []
    denom = 0
    for (i, j, k) in itertools.combinations_with_replacement(
        list(node.children), 3
    ):

        if i == j and j == k:
            continue

        combos.append((i, j, k))

        val = 0
        if i == j:
            val = special.comb(len(i), 2) * len(k)
        elif j == k:
            val = special.comb(len(j), 2) * len(i)
        elif i == k:
            val = special.comb(len(k), 2) * len(j)
        else:
            val = len(i) * len(j) * len(k)
        probs.append(val)
        denom += val

    probs = [val / denom for val in probs]

    # choose daughter clades
    ind = np.random.choice(range(len(combos)), size=1, replace=False, p=probs)[0]
    (i, j, k) = combos[ind]

    if i == j:
        in_group = np.random.choice(i.get_leaf_names(), 2, replace=False)
        out_group = np.random.choice(k.get_leaf_names())
    elif j == k:
        in_group = np.random.choice(j.get_leaf_names(), 2, replace=False)
        out_group = np.random.choice(i.get_leaf_names())
    elif i == k:
        in_group = np.random.choice(k.get_leaf_names(), 2, replace=True)
        out_group = np.random.choice(j.get_leaf_names())
    else:

        return (
            (
                str(np.random.choice(i.get_leaf_names())),
                str(np.random.choice(j.get_leaf_names())),
                str(np.random.choice(k.get_leaf_names())),
            ),
            "None",
        )

    return (str(in_group[0]), str(in_group[1]), str(out_group)), out_group
