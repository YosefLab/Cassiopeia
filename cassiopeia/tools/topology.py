"""
Utilities to assess topological properties of a phylogeny, such as balance
and expansion.
"""
from typing import Callable, Dict, Union, Optional

import math
import numpy as np
import pandas as pd
from scipy import spatial, stats


from cassiopeia.data import CassiopeiaTree, compute_phylogenetic_weight_matrix
from cassiopeia.mixins import CassiopeiaError
from cassiopeia.solver import dissimilarity_functions


def compute_expansion_pvalues(
    tree: CassiopeiaTree,
    min_clade_size: int = 10,
    min_depth: int = 1,
    copy: bool = False,
) -> Union[CassiopeiaTree, None]:
    """Call expansion pvalues on a tree.

    Uses the methodology described in Yang, Jones et al, BioRxiv (2021) to
    assess the expansion probability of a given subclade of a phylogeny.
    Mathematical treatment of the coalescent probability is described in
    Griffiths and Tavare, Stochastic Models (1998).

    The probability computed corresponds to the probability that, under a simple
    neutral coalescent model, a given subclade contains the observed number of
    cells; in other words, a one-sided p-value. Often, if the probability is
    less than some threshold (e.g., 0.05), this might indicate that there exists
    some subclade under this node that to which this expansion probability can
    be attributed (i.e. the null hypothesis that the subclade is undergoing 
    neutral drift can be rejected).

    This function will add an attribute "expansion_pvalue" to the tree, and
    return None unless :param:`copy` is set to True.

    On a typical balanced tree, this function will perform in O(n log n) time, 
    but can be up to O(n^3) on highly unbalanced trees. A future endeavor may 
    be to impelement the function in O(n) time.

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
        tree.set_attribute(node, "expansion_pvalue", 1.0)

        if tree.is_root(node):
            _depths[node] = 0
        else:
            _depths[node] = _depths[tree.parent(node)] + 1

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
            p = nCk(n - b, k - 1) / nCk(n - 1, k - 1)

            tree.set_attribute(c, "expansion_pvalue", p)

    return tree if copy else None


def compute_cophenetic_correlation(
    tree: CassiopeiaTree,
    weights: Optional[pd.DataFrame] = None,
    dissimilarity_map: Optional[pd.DataFrame] = None,
    dissimilarity_function: Optional[
        Callable[[np.array, np.array, int, Dict[int, Dict[int, float]]], float]
    ] = dissimilarity_functions.weighted_hamming_distance,
) -> float:
    """Computes the cophenetic correlation of a lineage.

    Computes the cophenetic correlation of a lineage, which is defined as the
    Pearson correlation between the phylogenetic distance and dissimilarity
    between characters.

    If neither weight matrix nor the dissimilarity map are precomputed, then 
    this function will run in O(n^3 + n^2logn) time, as the dissimilarity map
    will take O(n^3) time and the phylogenetic distance will take O(n^2 logn)
    time.

    Args:
        tree: CassiopeiaTree
        weights: Phylogenetic weights matrix. If this is not specified, invokes
            `cas.data.compute_phylogenetic_weight_matrix`
        dissimilarity_map: Dissimilarity matrix between samples. If this is not
            specified, then `tree.compute_dissimilarity_map` will be called.
        dissimilarity_function: Dissimilarity function to use. If dissimilarity
            map is not passed in, and one does not already exist in the
            CassiopeiaTree, then this function will be used to compute the
            dissimilarities between samples.
    
    Returns:
        The cophenetic correlation of the tree.
    """

    # set phylogenetic weight matrix
    W = (
        compute_phylogenetic_weight_matrix(tree)
        if (weights is None)
        else weights
    )

    # set dissimilarity map
    D = (
        tree.get_dissimilarity_map()
        if (dissimilarity_map is None)
        else dissimilarity_map
    )
    if D is None:
        D = tree.compute_dissimilarity_map(
            dissimilarity_function=dissimilarity_function
        )

    # convert to condensed distance matrices
    Wp = spatial.distance.squareform(W)
    Dp = spatial.distance.squareform(D)

    return stats.pearsonr(Wp, Dp)


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
