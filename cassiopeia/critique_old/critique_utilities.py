"""
Utilities for the critique module.
"""

from collections import defaultdict
import ete3
import itertools
import numpy as np
from scipy import special
from typing import List, Tuple, Union


def annotate_tree(tree):
    leaf_children = defaultdict(list)
    nodes_at_depth = defaultdict(list)

    def annotate_node(n, depth):
        n.depth = depth
        if n.is_leaf():
            leaf_children[n].append(n)
            n.number_of_triplets = 0
            return

        number_of_leaves = 0
        correction = 0

        for child in n.children:
            annotate_node(child, depth + 1)
            leaf_children[n] += leaf_children[child]
            number_of_leaves += len(leaf_children[child])
            correction += special.comb(len(leaf_children[child]), 3)

        n.number_of_triplets = special.comb(number_of_leaves, 3) - correction
        if n.number_of_triplets > 0:
            nodes_at_depth[depth].append(n)

    root = [n for n in tree.traverse() if n.is_root()][0]
    annotate_node(root, 0)
    return leaf_children, nodes_at_depth


def sample_triplet_at_depth(
    tree: ete3.Tree, candidate_nodes: List[Union[int, str]], leaf_children
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
    total_triplets = sum([v.number_of_triplets for v in candidate_nodes])

    # sample a  node from this depth with probability proportional to the number
    # of triplets underneath it
    probs = [v.number_of_triplets / total_triplets for v in candidate_nodes]
    node = np.random.choice(candidate_nodes, size=1, replace=False, p=probs)[0]

    # Generate the probabilities to sample each combination of 3 daughter clades
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
            val = special.comb(len(leaf_children[i]), 2) * len(leaf_children[k])
        elif j == k:
            val = special.comb(len(leaf_children[j]), 2) * len(leaf_children[i])
        elif i == k:
            val = special.comb(len(leaf_children[k]), 2) * len(leaf_children[j])
        else:
            val = (
                len(leaf_children[i])
                * len(leaf_children[j])
                * len(leaf_children[k])
            )
        probs.append(val)
        denom += val

    probs = [val / denom for val in probs]

    # choose daughter clades
    ind = np.random.choice(range(len(combos)), size=1, replace=False, p=probs)[
        0
    ]
    (i, j, k) = combos[ind]

    if i == j:
        in_group = np.random.choice(leaf_children[i], 2, replace=False)
        out_group = np.random.choice(leaf_children[k])
    elif j == k:
        in_group = np.random.choice(leaf_children[j], 2, replace=False)
        out_group = np.random.choice(leaf_children[i])
    elif i == k:
        in_group = np.random.choice(leaf_children[k], 2, replace=True)
        out_group = np.random.choice(leaf_children[j])
    else:

        return (
            (
                str(np.random.choice(leaf_children[i]).name),
                str(np.random.choice(leaf_children[j]).name),
                str(np.random.choice(leaf_children[k]).name),
            ),
            "None",
        )
    return (
        str(in_group[0].name),
        str(in_group[1].name),
        str(out_group.name),
    ), str(out_group.name)
