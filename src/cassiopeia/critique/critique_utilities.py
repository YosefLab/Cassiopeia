"""Utilities for the critique module."""

import itertools
import math
from collections import defaultdict
from typing import Any

import networkx as nx
import numpy as np

from cassiopeia.data import CassiopeiaTree


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
    f = math.factorial
    return f(n) // f(r) // f(n - r)


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


def sample_triplet_at_depth(
    G: "nx.DiGraph",
    depth: int,
    depth_to_nodes: dict[int, list[Any]] | None = None,
) -> tuple[tuple[str, str, str], str]:
    """Samples a triplet at a given depth for a networkx DiGraph.

    Samples a triplet of leaves such that the depth of the LCA of the triplet
    is at the specified depth.

    Args:
        G: nx.DiGraph
        depth: Depth at which to sample the triplet
        depth_to_nodes: An optional dictionary that maps a depth to the nodes
            that appear at that depth. This speeds up the function considerably.

    Returns:
            A list of three leaves corresponding to the triplet name of the outgroup
            of the triplet.
    """
    annotated_depth_to_nodes = annotate_tree_depths(
        G
    )  # edits G in place so nodes have all three attributes
    if depth_to_nodes is not None:
        if depth_to_nodes != annotated_depth_to_nodes:
            raise ValueError("Provided depth_to_nodes does not match the graph structure.")
    else:
        depth_to_nodes = annotated_depth_to_nodes

    candidates = depth_to_nodes.get(depth, [])
    if not candidates:
        raise ValueError(f"No nodes at depth {depth}.")

    total_triplets = int(sum(G.nodes[v].get("number_of_triplets", 0) for v in candidates))
    if total_triplets > 0:
        probs = [G.nodes[v].get("number_of_triplets", 0) / total_triplets for v in candidates]
        node = np.random.choice(candidates, size=1, replace=False, p=probs)[0]
    else:
        node = candidates[0]

    subtree_leaf_size = {u: G.nodes[u]["subtree_leaf_size"] for u in G.nodes}

    # daughter clade combos
    kids = list(G.successors(node))
    combos: list[tuple[Any, Any, Any]] = []
    weights: list[int] = []
    denom = 0

    for a, b, c in itertools.combinations_with_replacement(kids, 3):
        if a == b == c:
            continue
        combos.append((a, b, c))
        sa, sb, sc = (subtree_leaf_size[a], subtree_leaf_size[b], subtree_leaf_size[c])
        if a == b:
            w = nCr(sa, 2) * sc
        elif b == c:
            w = nCr(sb, 2) * sa
        elif a == c:
            w = nCr(sc, 2) * sb
        else:
            w = sa * sb * sc
        weights.append(w)
        denom += w

    if denom == 0 or not combos:
        # not enough structure; sample any 3 leaves under node and mark unresolved
        leaves: list[Any] = []
        stack = [node]
        while stack:
            x = stack.pop()
            ch = list(G.successors(x))
            if not ch:
                leaves.append(x)
            else:
                stack.extend(ch)
        if len(leaves) < 3:
            raise ValueError("Not enough leaves to form a triplet at this depth.")
        pick = np.random.choice(leaves, 3, replace=False)
        return (str(pick[0]), str(pick[1]), str(pick[2])), "None"

    probs = [w / denom for w in weights]
    idx = int(np.random.choice(len(combos), size=1, replace=False, p=probs)[0])
    a, b, c = combos[idx]

    # draw leaves according to combo
    def leaves_in_subtree(u: Any) -> list[Any]:
        leaves: list[Any] = []
        st = [u]
        while st:
            x = st.pop()
            ch = list(G.successors(x))
            if not ch:
                leaves.append(x)
            else:
                st.extend(ch)
        return leaves

    if a == b:
        in_group = np.random.choice(leaves_in_subtree(a), 2, replace=False)
        out_group = np.random.choice(leaves_in_subtree(c))
    elif b == c:
        in_group = np.random.choice(leaves_in_subtree(b), 2, replace=False)
        out_group = np.random.choice(leaves_in_subtree(a))
    elif a == c:
        in_group = np.random.choice(leaves_in_subtree(c), 2, replace=False)  # no replacement
        out_group = np.random.choice(leaves_in_subtree(b))
    else:
        return (
            (
                str(np.random.choice(leaves_in_subtree(a))),
                str(np.random.choice(leaves_in_subtree(b))),
                str(np.random.choice(leaves_in_subtree(c))),
            ),
            "None",
        )

    return (str(in_group[0]), str(in_group[1]), str(out_group)), str(out_group)
