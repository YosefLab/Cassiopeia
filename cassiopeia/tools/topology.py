"""
Utilities to assess topological properties of a phylogeny, such as balance
and expansion.
"""
from typing import Union

import math
import numpy as np
import pandas as pd

from cassiopeia.data import CassiopeiaTree
from cassiopeia.mixins import CassiopeiaError


def compute_expansion_probabilities(
    tree: CassiopeiaTree,
    min_clade_size: int = 10,
    min_depth: int = 1,
    copy: bool = False,
) -> Union[CassiopeiaTree, None]:
    """Call expansion probabilities on a tree.

    Uses the methodology described in Yang, Jones et al, BioRxiv (2021) to
    assess the expansion probability of a given subclade of a phylogeny.
    Mathematical treatment of the coalescent probability is described in
    Griffiths and Tavare, Stochastic Models (1998).

    The probability corresponds to the probability that, under a simple neutral
    coalescent model, a given subclade contains the observed number of cells; in
    other words, a one-sided p-value. Often, if the probability is less than
    some threshold (e.g., 0.05), this might indicate that there exists some
    subclade under this node that to which this expansion probability can be
    attributed (i.e. the null hypothesis that the subclade is undergoing 
    neutral drift can be rejected).

    This function will add an attribute "expansion_probability" to the tree, and
    return None unless :param:`copy` is set to True.

    Args:
        tree: CassiopeiaTree
        min_clade_size: Minimum number of leaves in a subtree to be considered.
        min_depth: Minimum depth of clade to be considered. Depth is measured
            in number of nodes from the root, not branch lengths.
        copy: Return copy.

    Returns:
        If copy is set to False, returns the tree with attributes added
            in place. Else, returns a new CassiopeiaTree.
    """

    tree = tree.copy() if copy else tree

    # instantiate attributes
    _depths = {} 
    for node in tree.depth_first_traverse_nodes(postorder=False):
        tree.set_attribute(node, "expansion_probability", 1.0)

        if tree.is_root(node):
            _depths[node] = 0
        else:
            _depths[node] = (_depths[tree.parent(node)] + 1)

    for node in tree.depth_first_traverse_nodes(postorder=False):

        n = len(tree.leaves_in_subtree(node))

        k = len(tree.children(node))
        for c in tree.children(node):

            if len(tree.leaves_in_subtree(c)) < min_clade_size:
                continue
            
            depth = _depths[c]
            if depth < min_depth:
                continue

            b = len(tree.leaves_in_subtree(c))

            # this value below is a simplification of the quantity:
            # sum[simple_coalescent_probability(n, b2, k) for \
            #   b2 in range(b, n - k + 2)]
            p = nCk(n-b, k-1) / nCk(n-1, k-1)

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
    return nCk(n - b - 1, k - 2) / nCk(n - 1, k - 1)


def nCk(n: int, k: int) -> float:
    """Compute the quantity n choose k.

    Args:
        n: Number of items total.
        k: Number of items to choose.

    Returns:
        The number of ways to choose k items from n.
    """

    if k > n:
        raise CassiopeiaError("Argument k cannot be larger than n.")

    f = math.factorial
    return f(n) // f(k) // f(n - k)
