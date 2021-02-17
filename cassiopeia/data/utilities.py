"""
General utilities for the datasets encountered in Cassiopeia.
"""
import copy
from queue import PriorityQueue
from typing import Callable, Dict, List, Optional

import ete3
import networkx as nx
import numba
import numpy as np


def get_lca_characters(
    vecs: List[List[int]], missing_state_indicator: int
) -> List[int]:
    """Builds the character vector of the LCA of a list of character vectors,
    obeying Camin-Sokal Parsimony.

    For each index in the reconstructed vector, imputes the non-missing
    character if only one of the constituent vectors has a missing value at that
    index, and imputes missing value if all have a missing value at that index.

    Args:
        vecs: A list of character vectors to generate an LCA for
        missing_state_indicator: The character representing missing values

    Returns:
        A list representing the character vector of the LCA

    """
    k = len(vecs[0])
    for i in vecs:
        assert len(i) == k
    lca_vec = [0] * len(vecs[0])
    for i in range(k):
        chars = set([vec[i] for vec in vecs])
        if len(chars) == 1:
            lca_vec[i] = list(chars)[0]
        else:
            if missing_state_indicator in chars:
                chars.remove(missing_state_indicator)
                if len(chars) == 1:
                    lca_vec[i] = list(chars)[0]
    return lca_vec


def newick_to_networkx(newick_string: str) -> nx.DiGraph:
    """Converts a newick string to a networkx DiGraph.

    Args:
        newick_string: A newick string.

    Returns:
        A networkx DiGraph.
    """

    tree = ete3.Tree(newick_string, 1)
    return ete3_to_networkx(tree)


def ete3_to_networkx(tree: ete3.Tree) -> nx.DiGraph:
    """Converts an ete3 Tree to a networkx DiGraph.

    Args:
        tree: an ete3 Tree object

    Returns:
        a networkx DiGraph
    """

    g = nx.DiGraph()
    internal_node_iter = 0
    for n in tree.traverse():

        if n.is_root():
            if n.name == "":
                n.name = f"node{internal_node_iter}"
                internal_node_iter += 1
            continue

        if n.name == "":
            n.name = f"node{internal_node_iter}"
            internal_node_iter += 1

        g.add_edge(n.up.name, n.name)

    return g


def to_newick(tree: nx.DiGraph) -> str:
    """Converts a networkx graph to a newick string.

    Args:
        tree: A networkx tree

    Returns:
        A newick string representing the topology of the tree
    """

    def _to_newick_str(g, node):
        is_leaf = g.out_degree(node) == 0
        _name = str(node)
        return (
            "%s" % (_name,)
            if is_leaf
            else (
                "("
                + ",".join(
                    _to_newick_str(g, child) for child in g.successors(node)
                )
                + ")"
            )
        )

    root = [node for node in tree if tree.in_degree(node) == 0][0]
    return _to_newick_str(tree, root) + ";"


def compute_dissimilarity_map(
    cm: np.array,
    C: int,
    dissimilarity_function: Callable,
    weights: Optional[Dict[int, Dict[int, float]]] = None,
    missing_state_indicator: int = -1,
) -> np.array:
    """Compute the dissimilarity between all samples

    An optimized function for computing pairwise dissimilarities between
    samples in a character matrix according to the dissimilarity function.

    Args:
        cm: Character matrix
        C: Number of samples
        weights: Weights to use for comparing states.
        missing_state_indicator: State indicating missing data

    Returns:
        A dissimilarity mapping as a flattened array.
    """

    nb_dissimilarity = numba.jit(dissimilarity_function, nopython=True)

    nb_weights = numba.typed.Dict.empty(numba.types.int64, numba.types.DictType(numba.types.int64, numba.types.float64))
    if weights:

        for k, v in weights.items():
            nb_char_weights = numba.typed.Dict.empty(numba.types.int64, numba.types.float64)
            for state, prior in v.items():
                nb_char_weights[state] = prior
            nb_weights[k] = nb_char_weights

    @numba.jit(nopython=True)
    def _compute_dissimilarity_map(cm, C, missing_state_indicator, nb_weights):

        dm = np.zeros(C * (C - 1) // 2, dtype=numba.float64)
        k = 0
        for i in range(C - 1):
            for j in range(i + 1, C):

                s1 = cm[i, :]
                s2 = cm[j, :]
                dm[k] = nb_dissimilarity(
                    s1, s2, missing_state_indicator, nb_weights
                )
                k += 1

        return dm

    return _compute_dissimilarity_map(cm, C, missing_state_indicator, nb_weights)


def resolve_multifurcations_networkx(tree: nx.DiGraph) -> nx.DiGraph:
    r"""
    Given a tree represented by a networkx DiGraph, it resolves
    multifurcations. The tree is NOT modified in-place.
    The root is made to have only one children, as in a real-life tumor
    (the founding cell never divides immediately!)
    """
    tree = copy.deepcopy(tree)
    node_names = set([n for n in tree])
    root = [n for n in tree if tree.in_degree(n) == 0][0]
    subtree_sizes = {}
    _dfs_subtree_sizes(tree, subtree_sizes, root)
    assert len(subtree_sizes) == len([n for n in tree])

    # First make the root have degree 1.
    if tree.out_degree(root) >= 2:
        children = list(tree.successors(root))
        assert len(children) == tree.out_degree(root)
        # First remove the edges from the root
        tree.remove_edges_from([(root, child) for child in children])
        # Now create the intermediate node and add edges back
        root_child = f"{root}-child"
        if root_child in node_names:
            raise RuntimeError("Node name already exists!")
        tree.add_edge(root, root_child)
        tree.add_edges_from([(root_child, child) for child in children])

    def _dfs_resolve_multifurcations(tree, v):
        children = list(tree.successors(v))
        if len(children) >= 3:
            # Must resolve the multifurcation
            _resolve_multifurcation(tree, v, subtree_sizes, node_names)
        for child in children:
            _dfs_resolve_multifurcations(tree, child)

    _dfs_resolve_multifurcations(tree, root)
    # Check that the tree is binary
    if not (len(tree.nodes) == len(tree.edges) + 1):
        raise RuntimeError("Failed to binarize tree")
    return tree


def _resolve_multifurcation(tree, v, subtree_sizes, node_names):
    r"""
    node_names is used to make sure we don't create a node name that already
    exists.
    """
    children = list(tree.successors(v))
    n_children = len(children)
    assert n_children >= 3

    # Remove all edges from v to its children
    tree.remove_edges_from([(v, child) for child in children])

    # Create the new binary structure
    queue = PriorityQueue()
    for child in children:
        queue.put((subtree_sizes[child], child))

    for i in range(n_children - 2):
        # Coalesce two smallest subtrees
        subtree_1_size, subtree_1_root = queue.get()
        subtree_2_size, subtree_2_root = queue.get()
        assert subtree_1_size <= subtree_2_size
        coalesced_tree_size = subtree_1_size + subtree_2_size + 1
        coalesced_tree_root = f"{v}-coalesce-{i}"
        if coalesced_tree_root in node_names:
            raise RuntimeError("Node name already exists!")
        # For debugging:
        # print(f"Coalescing {subtree_1_root} (sz {subtree_1_size}) and"
        #       f" {subtree_2_root} (sz {subtree_2_size})")
        tree.add_edges_from(
            [
                (coalesced_tree_root, subtree_1_root),
                (coalesced_tree_root, subtree_2_root),
            ]
        )
        queue.put((coalesced_tree_size, coalesced_tree_root))
    # Hang the two subtrees obtained to v
    subtree_1_size, subtree_1_root = queue.get()
    subtree_2_size, subtree_2_root = queue.get()
    assert subtree_1_size <= subtree_2_size
    tree.add_edges_from([(v, subtree_1_root), (v, subtree_2_root)])


def _dfs_subtree_sizes(tree, subtree_sizes, v) -> int:
    res = 1
    for child in tree.successors(v):
        res += _dfs_subtree_sizes(tree, subtree_sizes, child)
    subtree_sizes[v] = res
    return res
