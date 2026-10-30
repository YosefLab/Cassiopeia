"""Utilities for the critique module."""

import math
from collections import defaultdict
from typing import Any

import networkx as nx
import numpy as np


def nCr(n: int, r: int) -> float:
    """Computes nCr.

    Args:
        n: Total number
        r: Number to sample

    Returns:
            nCr
    """
    if r > n or n < 0 or r < 0:
        return 0
    return math.comb(n, r)


def annotate_tree_depths(G: nx.DiGraph) -> dict[int, list[Any]]:
    """Annotates tree depth at every node for a networkx DiGraph.

    Adds three attributes to each node of the tree: "depth", "subtree_leaf_size"
    (the number of leaves in the subtree rooted at that node), and "number_of_triplets"
    (how many triplets are rooted at that node.)
    Modifies the tree in place.

    Args:
        G: nx.DiGraph

    Returns:
            A dictionary mapping depth to the list of nodes at that depth.
    """
    root = G.graph.get("root") or next((n for n in G.nodes if G.in_degree(n) == 0), None)
    if root is None:
        raise ValueError("No root found for depth annotation.")

    # depth
    depths = nx.single_source_shortest_path_length(G, root)
    depth_to_nodes = defaultdict(list)
    for node, depth in depths.items():
        G.nodes[node]["depth"] = depth
        depth_to_nodes[depth].append(node)

    # leaf sizes
    leaf_sz = {}
    for u in reversed(list(nx.topological_sort(G))):
        kids = list(G.successors(u))
        leaf_sz[u] = 1 if not kids else sum(leaf_sz[c] for c in kids)
        G.nodes[u]["subtree_leaf_size"] = leaf_sz[u]

    # triplet count
    for u in G.nodes:
        kids = list(G.successors(u))
        total = sum(leaf_sz[c] for c in kids)
        correction = sum(nCr(leaf_sz[c], 3) for c in kids)
        G.nodes[u]["number_of_triplets"] = nCr(total, 3) - correction

    return dict(depth_to_nodes)


def get_outgroup(G: nx.DiGraph, triplet: tuple[Any, Any, Any]) -> str:
    """Determine outgroup via #shared ancestors per pair; return 'None' on ties."""
    if nx is None:
        raise ImportError("networkx is required for NX helpers.")

    i, j, k = triplet
    i_anc = set(nx.ancestors(G, i))
    j_anc = set(nx.ancestors(G, j))
    k_anc = set(nx.ancestors(G, k))

    ij_common = len(i_anc & j_anc)
    ik_common = len(i_anc & k_anc)
    jk_common = len(j_anc & k_anc)

    if ij_common > jk_common and ij_common > ik_common:
        return str(k)
    if ik_common > jk_common and ik_common > ij_common:
        return str(j)
    if jk_common > ij_common and jk_common > ik_common:
        return str(i)
    return "None"


def _leaves_below(G: nx.DiGraph, u: Any) -> list[Any]:
    """Leaves of the subtree rooted at *u*."""
    leaves: list[Any] = []
    stack = [u]
    while stack:
        x = stack.pop()
        children = list(G.successors(x))
        if children:
            stack.extend(children)
        else:
            leaves.append(x)
    return leaves


def sample_triplet_at_depth(
    G: "nx.DiGraph",
    depth: int,
    depth_to_nodes: dict[int, list[Any]] | None = None,
) -> tuple[tuple[str, str, str], str]:
    """Samples a triplet at a given depth for a networkx DiGraph.

    Samples a triplet of leaves such that the depth of the LCA of the triplet
    is at the specified depth. The LCA is drawn in proportion to the number of
    triplets it is the LCA of, and the triplet uniformly among those. A triplet
    with two leaves in one daughter clade of the LCA is resolvable (the third
    leaf is the outgroup); one with its leaves in three daughter clades is not.

    Args:
        G: nx.DiGraph
        depth: Depth at which to sample the triplet
        depth_to_nodes: An optional dictionary that maps a depth to the nodes
            that appear at that depth, as returned by
            :func:`annotate_tree_depths` (which must have been run on *G*).
            This speeds up the function considerably.

    Returns:
            A list of three leaves corresponding to the triplet name of the outgroup
            of the triplet.
    """
    if depth_to_nodes is None:
        depth_to_nodes = annotate_tree_depths(G)  # edits G in place

    candidates = depth_to_nodes.get(depth, [])
    if not candidates:
        raise ValueError(f"No nodes at depth {depth}.")

    total_triplets = int(sum(G.nodes[v].get("number_of_triplets", 0) for v in candidates))
    if total_triplets > 0:
        probs = [G.nodes[v].get("number_of_triplets", 0) / total_triplets for v in candidates]
        node = candidates[np.random.choice(len(candidates), p=probs)]
    else:
        node = candidates[0]

    kids = list(G.successors(node))
    sizes = np.array([G.nodes[k]["subtree_leaf_size"] for k in kids], dtype=float)
    total = sizes.sum()
    # Triplets with two leaves in one clade (resolvable) and with three clades.
    resolvable = np.array([nCr(int(s), 2) for s in sizes], dtype=float) * (total - sizes)
    n_resolvable = resolvable.sum()
    n_unresolvable = (total**3 - 3 * total * (sizes**2).sum() + 2 * (sizes**3).sum()) / 6

    if n_resolvable + n_unresolvable <= 0:
        # not enough structure; sample any 3 leaves under node and mark unresolved
        leaves = _leaves_below(G, node)
        if len(leaves) < 3:
            raise ValueError("Not enough leaves to form a triplet at this depth.")
        pick = np.random.choice(len(leaves), 3, replace=False)
        return tuple(str(leaves[i]) for i in pick), "None"

    if np.random.random() < n_resolvable / (n_resolvable + n_unresolvable):
        a = np.random.choice(len(kids), p=resolvable / n_resolvable)
        others = sizes.copy()
        others[a] = 0
        b = np.random.choice(len(kids), p=others / others.sum())
        in_leaves = _leaves_below(G, kids[a])
        in_group = [in_leaves[i] for i in np.random.choice(len(in_leaves), 2, replace=False)]
        out_leaves = _leaves_below(G, kids[b])
        out_group = out_leaves[np.random.randint(len(out_leaves))]
        return (str(in_group[0]), str(in_group[1]), str(out_group)), str(out_group)

    # Three distinct clades, each drawn in proportion to its number of leaves.
    while True:
        a, b, c = np.random.choice(len(kids), 3, p=sizes / total)
        if a != b and b != c and a != c:
            break
    picks = []
    for k in (a, b, c):
        leaves = _leaves_below(G, kids[k])
        picks.append(str(leaves[np.random.randint(len(leaves))]))
    return tuple(picks), "None"
