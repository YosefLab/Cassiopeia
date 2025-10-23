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


def annotate_tree_depths(tree: CassiopeiaTree) -> None:
    """Annotates tree depth at every node.

    Adds two attributes to the tree: how far away each node is from the root of
    the tree and how many triplets are rooted at that node. Modifies the tree
    in place.

    Args:
        tree: An ete3 Tree

    Returns:
            A dictionary mapping depth to the list of nodes at that depth.
    """
    depth_to_nodes = defaultdict(list)
    for n in tree.depth_first_traverse_nodes(source=tree.root, postorder=False):
        if tree.is_root(n):
            tree.set_attribute(n, "depth", 0)
        else:
            tree.set_attribute(n, "depth", tree.get_attribute(tree.parent(n), "depth") + 1)

        depth_to_nodes[tree.get_attribute(n, "depth")].append(n)

        number_of_leaves = 0
        correction = 0
        for child in tree.children(n):
            number_of_leaves += len(tree.leaves_in_subtree(child))
            correction += nCr(len(tree.leaves_in_subtree(child)), 3)

        tree.set_attribute(n, "number_of_triplets", nCr(number_of_leaves, 3) - correction)

    return depth_to_nodes


def get_outgroup(tree: CassiopeiaTree, triplet: tuple[str, str, str]) -> str:
    """Infers the outgroup of a triplet from a CassiopeiaTree.

    Finds the outgroup based on the depth of the latest-common-ancestors
    of each pair of items. The pair with the deepest LCA is the
    ingroup and the remaining leaf is the outgroup. We infer the depth
    of the LCA from the number of shared ancestors.

    Args:
        tree: CassiopeiaTree
        triplet: A tuple of three leaves constituting a triplet.

    Returns:
            The outgroup (i.e. the most distal leaf in the triplet.)
    """
    i, j, k = triplet[0], triplet[1], triplet[2]

    i_ancestors = tree.get_all_ancestors(i)
    j_ancestors = tree.get_all_ancestors(j)
    k_ancestors = tree.get_all_ancestors(k)

    ij_common = len(set(i_ancestors) & set(j_ancestors))
    ik_common = len(set(i_ancestors) & set(k_ancestors))
    jk_common = len(set(j_ancestors) & set(k_ancestors))
    out_group = "None"
    if ij_common > jk_common and ij_common > ik_common:
        out_group = k
    elif ik_common > jk_common and ik_common > ij_common:
        out_group = j
    elif jk_common > ij_common and jk_common > ik_common:
        out_group = i
    return out_group


def sample_triplet_at_depth(
    tree: CassiopeiaTree,
    depth: int,
    depth_to_nodes: dict[int, list[str]] | None = None,
) -> tuple[list[int], str]:
    """Samples a triplet at a given depth.

    Samples a triplet of leaves such that the depth of the LCA of the triplet
    is at the specified depth.

    Args:
        tree: CassiopeiaTree
        depth: Depth at which to sample the triplet
        depth_to_nodes: An optional dictionary that maps a depth to the nodes
            that appear at that depth. This speeds up the function considerably.

    Returns:
            A list of three leaves corresponding to the triplet name of the outgroup
            of the triplet.
    """
    if depth_to_nodes is None:
        candidate_nodes = tree.filter_nodes(lambda x: tree.get_attribute(x, "depth") == depth)
    else:
        candidate_nodes = depth_to_nodes[depth]

    total_triplets = sum([tree.get_attribute(v, "number_of_triplets") for v in candidate_nodes])

    # sample a  node from this depth with probability proportional to the number
    # of triplets underneath it
    probs = [tree.get_attribute(v, "number_of_triplets") / total_triplets for v in candidate_nodes]
    node = np.random.choice(candidate_nodes, size=1, replace=False, p=probs)[0]

    # Generate the probabilities to sample each combination of 3 daughter clades
    # to sample from, proportional to the number of triplets in each daughter
    # clade. Choices include all ways to choose 3 different daughter clades
    # or 2 from one daughter clade and one from another
    probs = []
    combos = []
    denom = 0
    for i, j, k in itertools.combinations_with_replacement(list(tree.children(node)), 3):
        if i == j and j == k:
            continue

        combos.append((i, j, k))

        size_of_i = len(tree.leaves_in_subtree(i))
        size_of_j = len(tree.leaves_in_subtree(j))
        size_of_k = len(tree.leaves_in_subtree(k))

        val = 0
        if i == j:
            val = nCr(size_of_i, 2) * size_of_k
        elif j == k:
            val = nCr(size_of_j, 2) * size_of_i
        elif i == k:
            val = nCr(size_of_k, 2) * size_of_j
        else:
            val = size_of_i * size_of_j * size_of_k
        probs.append(val)
        denom += val

    probs = [val / denom for val in probs]

    # choose daughter clades
    ind = np.random.choice(range(len(combos)), size=1, replace=False, p=probs)[0]
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


def annotate_tree_depths_nx(G: nx.DiGraph) -> dict[int, list[Any]]:
    """Annotates tree depth at every node for a networkx DiGraph.

    Adds two attributes to the tree: how far away each node is from the root of
    the tree and how many triplets are rooted at that node. Modifies the tree
    in place.

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

    # triplet count
    for u in G.nodes:
        kids = list(G.successors(u))
        total = sum(leaf_sz[c] for c in kids)
        correction = sum(nCr(leaf_sz[c], 3) for c in kids)
        G.nodes[u]["number_of_triplets"] = nCr(total, 3) - correction

    return dict(depth_to_nodes)


def get_outgroup_nx(G: "nx.DiGraph", triplet: tuple[Any, Any, Any]) -> str:
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


def sample_triplet_at_depth_nx(
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
    # candidates at this depth
    if depth_to_nodes is None:
        candidates = [n for n, d in G.nodes(data=True) if d.get("depth") == depth]
    else:
        candidates = depth_to_nodes.get(depth, [])
    if not candidates:
        raise ValueError(f"No nodes at depth {depth}.")

    total_triplets = int(sum(G.nodes[v].get("number_of_triplets", 0) for v in candidates))
    if total_triplets > 0:
        probs = [G.nodes[v].get("number_of_triplets", 0) / total_triplets for v in candidates]
        node = np.random.choice(candidates, size=1, replace=False, p=probs)[0]
    else:
        node = candidates[0]  # fallback; likely unresolved

    # Precomputed subtree leaf sizes from annotate_tree_depths_nx:
    # If user calls this after annotate_tree_depths_nx (like in your main), we can reuse.
    # But for safety, compute on the fly if missing (cheap with topo DP).
    if "subtree_leaf_size" not in next(iter(G.nodes(data=True)))[1]:
        # compute and stash (optional; we won't store, we just derive sizes locally)
        leaf_sz: dict[Any, int] = {}
        for u in reversed(list(nx.topological_sort(G))):
            kids = list(G.successors(u))
            if not kids:
                leaf_sz[u] = 1
            else:
                leaf_sz[u] = sum(leaf_sz[c] for c in kids)
    else:
        # not used here; left as a hook if you later store it
        leaf_sz = {u: d["subtree_leaf_size"] for u, d in G.nodes(data=True)}

    def subtree_leaf_size(u: Any) -> int:
        # Prefer cached leaf_sz if present, else derive quickly
        if u in leaf_sz:
            return leaf_sz[u]
        kids = list(G.successors(u))
        if not kids:
            leaf_sz[u] = 1
        else:
            leaf_sz[u] = sum(subtree_leaf_size(c) for c in kids)
        return leaf_sz[u]

    # daughter clade combos
    kids = list(G.successors(node))
    combos: list[tuple[Any, Any, Any]] = []
    weights: list[int] = []
    denom = 0

    for a, b, c in itertools.combinations_with_replacement(kids, 3):
        if a == b == c:
            continue
        combos.append((a, b, c))
        sa, sb, sc = (subtree_leaf_size(a), subtree_leaf_size(b), subtree_leaf_size(c))
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
