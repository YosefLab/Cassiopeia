"""
Utilities for the critique module.
"""

import ete3
import itertools
import numpy as np
from scipy import special
from typing import List, Tuple

from cassiopeia.data import CassiopeiaTree

def annotate_tree_depths(tree: CassiopeiaTree) -> None:
    """Annotates tree depth at every node.

    Adds two attributes to the tree: how far away each node is from the root of
    the tree and how many triplets are rooted at that node. Modifies the tree
    in place.

    Args:
        tree: An ete3 Tree
    """
    for n in tree.depth_first_traverse_nodes(source = tree.root, postorder=False):
        if tree.is_root(n):
            tree.set_attribute(n, "depth", 0)
        else:
            tree.set_attribute(n, "depth", tree.get_attribute(tree.parent(n), "depth") + 1)

        number_of_leaves = 0
        correction = 0
        for child in tree.children(n):
            number_of_leaves += len(tree.leaves_in_subtree(child))
            correction += special.comb(len(tree.leaves_in_subtree(child)), 3)

        tree.set_attribute(n, "number_of_triplets", special.comb(number_of_leaves, 3) - correction)

def sample_triplet_at_depth(
    tree: CassiopeiaTree, depth: int
) -> Tuple[List[int], str]:
    """Samples a triplet at a given depth.

    Samples a triplet of leaves such that the depth of the LCA of the triplet
    is at the specified depth. 

    Args:
        tree: CassiopeiaTree
        depth: Depth at which to sample the triplet

    Returns:
        A list of three leaves corresponding to the triplet name of the outgroup
            of the triplet.
    """

    candidate_nodes = tree.filter_nodes(lambda x: tree.get_attribute(x, "depth") == depth)
    total_triplets = sum([tree.get_attribute(v, "number_of_triplets") for v in candidate_nodes])

    # sample a  node from this depth with probability proportional to the number
    # of triplets underneath it
    probs = [tree.get_attribute(v, "number_of_triplets") / total_triplets for v in candidate_nodes]
    node = np.random.choice(candidate_nodes, size=1, replace=False, p=probs)[0]

    # Generate the probilities to sample each combination of 3 daughter clades
    # to sample from, proportional to the number of triplets in each daughter
    # clade. Choices include all ways to choose 3 different daughter clades
    # or 2 from one daughter clade and one from another
    probs = []
    combos = []
    denom = 0
    for (i, j, k) in itertools.combinations_with_replacement(
        list(tree.children(node)), 3
    ):

        if i == j and j == k:
            continue

        combos.append((i, j, k))

        size_of_i = len(tree.leaves_in_subtree(i))
        size_of_j = len(tree.leaves_in_subtree(j))
        size_of_k = len(tree.leaves_in_subtree(k))

        val = 0
        if i == j:
            val = special.comb(size_of_i, 2) * size_of_k
        elif j == k:
            val = special.comb(size_of_j, 2) * size_of_i
        elif i == k:
            val = special.comb(size_of_k, 2) * size_of_j
        else:
            val = size_of_i * size_of_j * size_of_k
        probs.append(val)
        denom += val

    probs = [val / denom for val in probs]

    # choose daughter clades
    ind = np.random.choice(range(len(combos)), size=1, replace=False, p=probs)[
        0
    ]
    (i, j, k) = combos[ind]

    if i == j:
        in_group = np.random.choice(tree.leaves_in_subtree(i), 2, replace=False)
        out_group = np.random.choice(tree.leaves_in_subtree(k))
    elif j == k:
        in_group = np.random.choice(tree.leaves_in_subtree(j), 2, replace=False)
        out_group = np.random.choice(tree.leaves_in_subtree(i))
    elif i == k:
        in_group = np.random.choice(tree.leaves_in_subtree(k), 2, replace=True)
        out_group = np.random.choice(tree.leaves_in_subtree(j))
    else:

        return (
            (
                str(np.random.choice(tree.leaves_in_subtree(i))),
                str(np.random.choice(tree.leaves_in_subtree(j))),
                str(np.random.choice(tree.leaves_in_subtree(k))),
            ),
            "None",
        )

    return (str(in_group[0]), str(in_group[1]), str(out_group)), out_group
