"""
General utilities for the datasets encountered in Cassiopeia.
"""
import ete3
import networkx as nx
from typing import List


def get_lca_characters(vecs: List[List[int]], missing_char: int) -> List[int]:
    """Builds the character vector of the LCA of a list of character vectors,
    obeying Camin-Sokal Parsimony.

    For each index in the reconstructed vector, imputes the non-missing
    character if only one of the constituent vectors has a missing value at that
    index, and imputes missing value if all have a missing value at that index.

    Args:
        vecs: A list of character vectors to generate an LCA for
        missing_char: The character representing missing values

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
            if missing_char in chars:
                chars.remove(missing_char)
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
