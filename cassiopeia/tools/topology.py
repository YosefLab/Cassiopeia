"""
Utilities to assess topological properties of a phylogeny, such as balance
and expansion.
"""
from typing import Union

import math
import numpy as np
import pandas as pd

from cassiopeia.data import CassiopeiaTree


def compute_expansion_probabilities(
    tree: CassiopeiaTree,
    min_clade_size: int = 10,
    min_depth: int = 1,
    copy=True,
) -> Union[CassiopeiaTree, None]:
    """Call expansions on a tree.

    Uses the methodology described in Yang, Jones et al, BioRxiv (2021) to
    assess the expansion probability of a given subclade of a phylogeny.

    This function will add an attribute "expansion_probability" to the tree.

    Args:
        tree: CassiopeiaTree
        min_clade_size: Minimum number of leaves in a subtree to be considered.
        min_depth: Minimum depth of clade to be considered.
        copy: Return copy.

    Returns:
        None. Adds attributes to the tree. 
    """

    tree = tree.copy() if copy else tree

    # instantiate attributes
    tree.set_attribute(tree.root, "depth", 0)
    for node in tree.depth_first_traverse_nodes(postorder=False):
        tree.set_attribute(node, "expansion_probability", 1.0)
        tree.set_attribute(
            node, "depth", tree.get_attribute(tree.parent(node, "depth")) + 1
        )

    for node in tree.depth_first_traverse_nodes(postorder=False):

        n = len(tree.leaves_in_subtree(node))
        depth = tree.get_attribute(node, "depth")
        if depth >= min_depth:

            k = len(tree.children(node))
            for c in tree.children(node):

                if len(tree.leaves_in_subtree(c)) < min_clade_size:
                    continue

                b = len(tree.leaves_in_subtree(c))
                p = np.sum(
                    [
                        simple_coalescent_probability(n, b2, k)
                        for b2 in range(b, n - k + 2)
                    ]
                )
                tree.set_attribute(c, "expansion_probability", p)

    return tree if copy else None


def simple_coalescent_probability(n: int, b: int, k: int) -> float:
    """Simple coalescent probability of imbalance.
    
    Assuming a simple coalescent model, compute the probability that a given
    lineage has exactly b samples, given there are n cells and k lineages
    overall.
 
    Args:
        n: Number of leaves in subtree
        b: Number of leaves in one lineage
        k: Number of lineages
    Returns:
        Probability of observing b leaves on one lineage in a tree of n total 
            leaves
    """
    return nCr(n - b - 1, k - 2) / nCr(n - 1, k - 1)


def nCr(n: int, k: int) -> float:
    """Compute the quantity n choose k.

    Args:
        n: Number of items total.
        k: Number of items to choose.

    Returns:
        The number of ways to choose k items from n.
    """
    f = math.factorial
    return f(n) // f(k) // f(n - k)
