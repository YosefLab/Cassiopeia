"""Utilities to assess topological properties of a phylogeny, such as balance and expansion."""

import math
from collections.abc import Callable

import networkx as nx
import numpy as np
import pandas as pd
from scipy import spatial, stats
from treedata import TreeData

from cassiopeia import dissimilarity as dissimilarity_functions
from cassiopeia.data import CassiopeiaTree, compute_phylogenetic_weight_matrix
from cassiopeia.mixins import CassiopeiaError
from cassiopeia.typing import TreeLike
from cassiopeia.utils import (
    _check_tree_has_key,
    _collapse_unifurcations,
    _combine_edge_data,
    _get_digraph,
    get_leaves,
)


def mean_depth(
    tree: CassiopeiaTree | TreeData | nx.DiGraph,
    depth_key: str,
    tree_key: str | None = None,
) -> float:
    """Compute the mean depth of a tree's leaves.

    Calculates the average depth across all leaf nodes in the tree. Depth is
    retrieved from the node attribute specified by ``depth_key``. This can
    represent either discrete generations (e.g., number of divisions) or
    continuous time (e.g., evolutionary time).

    Args:
        tree: Tree object (CassiopeiaTree, TreeData, or nx.DiGraph).
        depth_key: Node attribute key containing depth values (e.g., ``"depth"``,
            ``"time"``).
        tree_key: Tree key to use if ``tree`` is a TreeData object with multiple
            trees.

    Returns:
        Mean depth of the tree's leaves.
    """
    t, _ = _get_digraph(tree, tree_key=tree_key)
    _check_tree_has_key(t, depth_key)
    leaves = get_leaves(tree, tree_key=tree_key)
    depths = [t.nodes[leaf][depth_key] for leaf in leaves]
    return float(np.mean(depths))


def _mutationless_criteria(parent_states: list, child_states: list) -> bool:
    """Return ``True`` when a parent and child share identical character states.

    The default edge-collapse criterion: an edge carries no mutation when the
    parent and child have identical inferred character states (introducing a
    missing-data event counts as a mutation, so unequal states are retained).
    """
    return parent_states == child_states


# Registry of edge-collapse criteria. Maps a criterion name to a predicate
# ``(parent_states, child_states) -> bool`` that is ``True`` when the edge
# between them should be collapsed. The structural ``"unifurcation"`` criterion
# is handled separately (it needs no character states). Extend this to add
# future state-based criteria.
_COLLAPSE_CRITERIA: dict[str, Callable[[list, list], bool]] = {
    "mutationless": _mutationless_criteria,
}


def collapse_edges(
    tdata: TreeLike,
    tree_key: str | None = None,
    characters_key: str = "characters",
    criteria: str = "mutationless",
    collapse_root: bool = True,
    copy: bool = False,
) -> TreeLike | None:
    """Collapse edges of a tree according to a collapse criterion.

    For each internal node, any non-leaf child satisfying the collapse
    *criteria* is spliced out and its children are reattached to the node.
    Leaves are never removed. Numeric edge attributes (e.g. branch lengths) are
    summed across the removed edges so additive quantities are preserved;
    non-numeric attributes take the child edge's value.

    Two criteria are supported:

    * ``'mutationless'`` (default): collapse an edge when the parent and child
      have identical inferred character states. Ancestral character states must
      already be present on every node under the ``characters_key`` node
      attribute; call :func:`cassiopeia.tl.ancestral_characters` first if they
      are not.
    * ``'unifurcation'``: collapse every internal node with exactly one child
      (a structural criterion needing no character states).

    Only :class:`~treedata.TreeData` is supported.

    Args:
        tdata: TreeData object to operate on.
        tree_key: The ``obst`` key of the tree to use.
        characters_key: Node attribute holding character states (the same name
            as the obsm character matrix and the output of
            :func:`cassiopeia.tl.ancestral_characters`). Only used by the
            ``'mutationless'`` criterion.
        criteria: Name of the edge-collapse criterion to apply. Supports
            ``'mutationless'`` and ``'unifurcation'``.
        collapse_root: For ``criteria='unifurcation'``, whether to also collapse
            the root's single child into the root. Ignored otherwise.
        copy: If ``True``, operate on and return a copy of *tdata*; otherwise
            modify in place and return ``None``.

    Returns:
        A modified copy of *tdata* if ``copy=True``, else ``None``.

    Raises:
        TypeError: If *tdata* is not a TreeData object.
        ValueError: If *criteria* is not a recognized criterion.
        CassiopeiaError: If a node is missing character states (mutationless).
    """
    from treedata import TreeData

    if not isinstance(tdata, TreeData):
        raise TypeError(
            "collapse_edges() operates on TreeData. For a CassiopeiaTree, convert "
            "with CassiopeiaTree.to_treedata()."
        )
    if criteria != "unifurcation" and criteria not in _COLLAPSE_CRITERIA:
        raise ValueError(
            f"Unknown collapse criteria {criteria!r}. "
            f"Available: {sorted([*_COLLAPSE_CRITERIA, 'unifurcation'])}"
        )

    tdata = tdata.copy() if copy else tdata
    # TreeData stores frozen graphs; operate on a copy and write back.
    g, tree_key = _get_digraph(tdata, tree_key, copy=True)

    if criteria == "unifurcation":
        g = _collapse_unifurcations(g, collapse_root=collapse_root)
        tdata.obst[tree_key] = g
        return tdata if copy else None

    predicate = _COLLAPSE_CRITERIA[criteria]

    for node in g.nodes:
        if characters_key not in g.nodes[node]:
            raise CassiopeiaError(
                f"Node {node!r} has no character states under {characters_key!r}. "
                "Call cassiopeia.tl.ancestral_characters first."
            )

    for node in list(nx.dfs_postorder_nodes(g)):
        if g.out_degree(node) == 0:
            continue
        for child in list(g.successors(node)):
            if g.out_degree(child) == 0:
                continue
            if predicate(g.nodes[node][characters_key], g.nodes[child][characters_key]):
                parent_edge = dict(g.get_edge_data(node, child, default={}))
                for grandchild in list(g.successors(child)):
                    child_edge = dict(g.get_edge_data(child, grandchild, default={}))
                    g.add_edge(node, grandchild, **_combine_edge_data(parent_edge, child_edge))
                g.remove_node(child)

    tdata.obst[tree_key] = g

    return tdata if copy else None


def compute_expansion_pvalues(
    tree: CassiopeiaTree,
    min_clade_size: int = 10,
    min_depth: int = 1,
    copy: bool = False,
) -> CassiopeiaTree | None:
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
    weights: pd.DataFrame | None = None,
    dissimilarity_map: pd.DataFrame | None = None,
    dissimilarity_function: Callable[[np.array, np.array, int, dict[int, dict[int, float]]], float]
    | None = dissimilarity_functions.weighted_hamming,
) -> tuple[float, float]:
    """Computes the cophenetic correlation of a lineage.

    Computes the cophenetic correlation of a lineage, which is defined as the
    Pearson correlation between the phylogenetic distance and dissimilarity
    between characters.

    If neither weight matrix nor the dissimilarity map are precomputed, then
    this function will run in O(mn^2 + n^2logn + n^2) time, as the dissimilarity
    map will take O(mn^2) time, the phylogenetic distance will take O(n^2 logn)
    time, and the Pearson correlation will take O(n^2) time since it must
    compare n^2 entries (n = number of leaves; m = number of characters).

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
            The cophenetic correlation value and significance for the tree.
    """
    # set phylogenetic weight matrix
    W = compute_phylogenetic_weight_matrix(tree) if (weights is None) else weights

    # set dissimilarity map
    D = tree.get_dissimilarity_map() if (dissimilarity_map is None) else dissimilarity_map
    if D is None:
        tree.compute_dissimilarity_map(dissimilarity_function=dissimilarity_function)
        D = tree.get_dissimilarity_map()

    # align matrices
    cells = tree.leaves
    W = W.loc[cells, cells]
    D = D.loc[cells, cells]

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
