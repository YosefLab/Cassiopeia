"""Rooting procedures for trees.

Procedures are registered by name and operate on an (unrooted) tree graph,
returning a rooted :class:`~networkx.DiGraph`.  They are used both by
:func:`cassiopeia.solver.nj` (via :func:`apply`, on the complete undirected NJ
graph) and by :func:`reroot` (to re-root an already-built tree).  Custom
procedures can be added via :func:`register`.

Available procedures: ``outgroup``, ``midpoint``, ``centroid``, ``shared_mutation``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import networkx as nx

from cassiopeia.mixins import DistanceSolverError

if TYPE_CHECKING:
    import pandas as pd
    from treedata import TreeData

    from cassiopeia.data import CassiopeiaTree

# Registry: name → callable(graph, **kwargs) → nx.DiGraph
_PROCEDURES: dict[str, Callable] = {}


def register(name: str) -> Callable:
    """Register a rooting procedure under *name*.

    The decorated function must have the signature::

        fn(graph, **kwargs) -> nx.DiGraph

    where *graph* is an (unrooted) tree graph and the return value is a rooted
    directed tree.
    """

    def decorator(fn: Callable) -> Callable:
        _PROCEDURES[name] = fn
        return fn

    return decorator


def _default_root(data: CassiopeiaTree | TreeData) -> str:
    from treedata import TreeData

    if isinstance(data, TreeData):
        return list(data.obs_names)[0]
    if data.root_sample_name is None:
        raise DistanceSolverError(
            "No root is set on the CassiopeiaTree. Set root_sample_name or pass root='outgroup'."
        )
    return data.root_sample_name


# ── Shared helpers ────────────────────────────────────────────────────────────


def _leaves(graph: nx.Graph) -> list:
    """Leaf nodes of an undirected tree (degree 1)."""
    return [n for n in graph if graph.degree[n] == 1]


def _leaf_side_counts(undirected: nx.Graph, leaf_attr: dict[str, int] | None = None):
    """Compute per-edge leaf-side counts on an undirected tree.

    Runs a single DFS from an arbitrary node and returns a function
    ``side(u, v) -> (down_count, total)`` giving, for the edge ``{u, v}``, the
    number of (attributed) leaves on the side away from the DFS root and the
    overall total.  When *leaf_attr* is given, counts that attribute instead of
    raw leaf counts.

    Returns ``(parent, edge_side_fn, total)``.
    """
    root = next(iter(undirected.nodes))
    parent = {root: None}
    order = []
    stack = [root]
    while stack:
        u = stack.pop()
        order.append(u)
        for v in undirected[u]:
            if v == parent.get(u):
                continue
            parent[v] = u
            stack.append(v)

    is_leaf = {u: (undirected.degree[u] == 1) for u in undirected}
    if leaf_attr is None:
        value = {u: (1 if is_leaf[u] else 0) for u in undirected}
    else:
        value = {u: (leaf_attr.get(u, 0) if is_leaf[u] else 0) for u in undirected}

    down = dict(value)
    for u in reversed(order):
        p = parent.get(u)
        if p is not None:
            down[p] += down[u]
    total = sum(value.values())

    def side(u, v):
        # (count on the v-away-from-root side, total)
        if parent.get(v) == u:
            return down[v], total
        if parent.get(u) == v:
            return total - down[u], total
        return down[v], total

    return parent, side, total


def reroot_on_edge(graph: nx.DiGraph, new_root, best_edge) -> nx.DiGraph:
    """Insert *new_root* on *best_edge* and orient all edges away from it.

    Node attributes are preserved; the previous root and any resulting
    unifurcations are collapsed.
    """
    from cassiopeia.utils import _collapse_unifurcations

    undirected = graph.to_undirected()
    u, v = best_edge

    u2 = undirected.copy()
    if u2.has_edge(u, v):
        u2.remove_edge(u, v)
    u2.add_node(new_root)
    u2.add_edge(new_root, u)
    u2.add_edge(new_root, v)

    rooted = nx.DiGraph()
    rooted.add_nodes_from(graph.nodes(data=True))
    rooted.add_node(new_root)
    for p, c in nx.bfs_edges(u2, new_root):
        attrs = graph.get_edge_data(p, c) or graph.get_edge_data(c, p) or {}
        rooted.add_edge(p, c, **attrs)

    return _collapse_unifurcations(rooted, collapse_root=True)


# ── Procedures ────────────────────────────────────────────────────────────────


@register("outgroup")
def outgroup(graph: nx.Graph, outgroup: str | None = None, **kwargs) -> nx.DiGraph:
    """Root the tree using an outgroup leaf.

    When *outgroup* is ``None`` (NJ synthetic-root case), a synthetic all-zero
    leaf named ``"root"`` is expected in *graph* and used as the root.  When
    *outgroup* is a sample name, a new internal root is inserted on the branch
    connecting that sample to the rest of the tree.

    Args:
        graph: Complete undirected tree (including the synthetic leaf if
            *outgroup* is ``None``).
        outgroup: Name of an existing leaf to use as outgroup, or ``None`` to use
            the synthetic ``"root"`` leaf.
        kwargs: Keyword arguments.

    Returns:
        A rooted :class:`~networkx.DiGraph`.
    """
    if outgroup is None:
        if "root" not in graph.nodes:
            raise ValueError(
                "Expected synthetic 'root' node in the tree. "
                "Ensure the 'root' leaf was included in the distance matrix."
            )
        rooted = nx.DiGraph()
        for e in nx.dfs_edges(graph, source="root"):
            rooted.add_edge(e[0], e[1])
        return rooted

    if outgroup not in graph.nodes:
        raise ValueError(
            f"Outgroup {outgroup!r} not found in the tree. Available nodes: {sorted(graph.nodes)}"
        )
    undirected = graph.to_undirected()
    neighbors = list(undirected.neighbors(outgroup))
    if len(neighbors) != 1:
        raise ValueError(
            f"Outgroup {outgroup!r} must be a leaf node "
            f"(expected 1 neighbour, found {len(neighbors)})."
        )
    internal = neighbors[0]

    root_name = "root"
    while root_name in graph.nodes:
        root_name = root_name + "_internal"

    return reroot_on_edge(graph, root_name, (outgroup, internal))


@register("midpoint")
def midpoint(
    graph: nx.Graph, time_key: str = "length", new_root: str = "root", **kwargs
) -> nx.DiGraph:
    """Root at the midpoint of the longest leaf-to-leaf path.

    Branch lengths are read from the ``time_key`` edge attribute (defaulting to
    ``1.0`` per edge when absent).  The two most distant leaves are found and a
    new root is placed on the edge straddling the halfway point of the path
    between them.

    Args:
        graph: Tree graph (rooted or unrooted).
        time_key: Edge attribute holding branch lengths.
        new_root: Name for the inserted root node.
        kwargs: Keyword arguments.

    Returns:
        A rooted :class:`~networkx.DiGraph`.
    """
    weighted = nx.Graph()
    weighted.add_nodes_from(graph.nodes())
    for u, v in graph.to_undirected().edges():
        attrs = graph.get_edge_data(u, v) or graph.get_edge_data(v, u) or {}
        weighted.add_edge(u, v, weight=attrs.get(time_key, 1.0))

    leaves = _leaves(weighted)
    if len(leaves) < 2:
        raise ValueError("Midpoint rooting requires at least two leaves.")

    # farthest leaf pair
    best_pair = None
    best_dist = -1.0
    for leaf in leaves:
        dist = nx.single_source_dijkstra_path_length(weighted, leaf, weight="weight")
        for other in leaves:
            if other != leaf and dist.get(other, 0.0) > best_dist:
                best_dist = dist[other]
                best_pair = (leaf, other)

    a, b = best_pair
    path = nx.shortest_path(weighted, a, b, weight="weight")
    half = best_dist / 2.0
    acc = 0.0
    best_edge = (path[0], path[1])
    for i in range(len(path) - 1):
        w = weighted[path[i]][path[i + 1]]["weight"]
        if acc + w >= half:
            best_edge = (path[i], path[i + 1])
            break
        acc += w

    return reroot_on_edge(graph, new_root, best_edge)


@register("centroid")
def centroid(graph: nx.Graph, new_root: str = "root", **kwargs) -> nx.DiGraph:
    """Root on the edge that most evenly splits the leaves.

    Finds the edge whose two sides have the closest leaf counts, inserts a new
    root there, and orients edges away from it.

    Args:
        graph: Tree graph (rooted or unrooted).
        new_root: Name for the inserted root node.
        kwargs: Keyword arguments.

    Returns:
        A rooted :class:`~networkx.DiGraph`.
    """
    undirected = graph.to_undirected()
    if not nx.is_tree(undirected):
        raise ValueError("Input must be a tree.")

    _, side, total = _leaf_side_counts(undirected)

    best = None
    best_edge = None
    for u, v in undirected.edges():
        side_count, _ = side(u, v)
        other = total - side_count
        diff = abs(side_count - other)
        key = (diff, -min(side_count, other), tuple(sorted((u, v), key=str)))
        if best is None or key < best:
            best, best_edge = key, (u, v)

    return reroot_on_edge(graph, new_root, best_edge)


def _get_mutation_outgroup(characters: pd.DataFrame, missing_state=-1, unedited_state=0) -> list:
    """Identify outgroup leaves as those sharing the single most common mutation."""
    best_col = None
    best_value = None
    best_prop = -1
    for col in characters.columns:
        col_values = characters[col][~characters[col].isin([missing_state, unedited_state])]
        if col_values.empty:
            continue
        mode_value = col_values.mode().iloc[0]
        prop = (characters[col] == mode_value).mean()
        if prop > best_prop:
            best_prop = prop
            best_col = col
            best_value = mode_value
    if best_col is None:
        return []
    return characters.index[characters[best_col] == best_value].tolist()


@register("shared_mutation")
def shared_mutation(
    graph: nx.Graph,
    characters: pd.DataFrame = None,
    new_root: str = "root",
    missing_state=-1,
    unedited_state=0,
    **kwargs,
) -> nx.DiGraph:
    """Root using an outgroup defined by the most common shared mutation.

    The outgroup is the set of leaves carrying the single most common
    (non-missing, non-unedited) character state.  The edge whose leaf partition
    best matches the outgroup set (by Jaccard similarity) is chosen as the root
    edge.

    Args:
        graph: Tree graph (rooted or unrooted).
        characters: Character matrix indexed by leaf name.
        new_root: Name for the inserted root node.
        missing_state: Value representing missing data.
        unedited_state: Value representing the unedited state.
        kwargs: Keyword arguments.

    Returns:
        A rooted :class:`~networkx.DiGraph`.
    """
    if characters is None:
        raise ValueError("shared_mutation rooting requires a character matrix.")

    undirected = graph.to_undirected()
    outgroup_leaves = set(_get_mutation_outgroup(characters, missing_state, unedited_state))
    if not outgroup_leaves:
        return centroid(graph, new_root=new_root)

    leaf_attr = {
        n: (1 if n in outgroup_leaves else 0) for n in undirected if undirected.degree[n] == 1
    }
    _, leaf_side, total_leaves = _leaf_side_counts(undirected)
    _, out_side, total_out = _leaf_side_counts(undirected, leaf_attr=leaf_attr)

    def jaccard(side_leaves, side_out):
        union = side_leaves + total_out - side_out
        return side_out / union if union > 0 else 0.0

    best_jaccard = -1.0
    best_edge = None
    for u, v in undirected.edges():
        side_leaves, _ = leaf_side(u, v)
        side_out, _ = out_side(u, v)
        j_child = jaccard(side_leaves, side_out)
        j_parent = jaccard(total_leaves - side_leaves, total_out - side_out)
        j = max(j_child, j_parent)
        if j > best_jaccard:
            best_jaccard = j
            best_edge = (u, v)

    return reroot_on_edge(graph, new_root, best_edge)


# ── Public reroot ─────────────────────────────────────────────────────────────


def reroot(
    tdata: TreeData,
    method: str = "outgroup",
    tree_key: str | None = None,
    characters_key: str = "characters",
    time_key: str = "length",
    key_added: str | None = None,
    copy: bool = False,
    **kwargs,
) -> TreeData | None:
    """Re-root an already-built tree using a registered rooting procedure.

    Only :class:`~treedata.TreeData` is supported.

    Args:
        tdata: TreeData containing the tree to re-root.
        method: Rooting procedure name — one of ``'outgroup'``, ``'midpoint'``,
            ``'centroid'``, ``'shared_mutation'``.
        tree_key: ``obst`` key of the tree to re-root.
        characters_key: ``obsm`` key for the character matrix (used by
            ``shared_mutation``).
        time_key: Edge attribute holding branch lengths (used by ``midpoint``).
        key_added: ``obst`` key to store the re-rooted tree under.  Defaults to
            *tree_key* (overwriting in place).
        copy: If ``True``, operate on and return a copy of *tdata*; otherwise
            modify in place and return ``None``.
        **kwargs: Forwarded to the rooting procedure (e.g. ``outgroup=...``).

    Returns:
        A modified copy of *tdata* if ``copy=True``, else ``None``.

    Raises:
        TypeError: If *tdata* is not a TreeData object.
        ValueError: If *method* is not a registered procedure.
    """
    from treedata import TreeData

    from cassiopeia.utils import _get_characters, _get_digraph

    if not isinstance(tdata, TreeData):
        raise TypeError(
            "reroot() operates on TreeData. For a CassiopeiaTree, convert with "
            "CassiopeiaTree.to_treedata()."
        )
    if method not in _PROCEDURES:
        raise ValueError(f"Unknown rooting procedure {method!r}. Available: {sorted(_PROCEDURES)}")

    tdata = tdata.copy() if copy else tdata
    g, tree_key = _get_digraph(tdata, tree_key, copy=True)

    proc_kwargs = dict(kwargs)
    if method == "shared_mutation" and proc_kwargs.get("characters") is None:
        proc_kwargs["characters"] = _get_characters(tdata, characters_key)
    if method == "midpoint":
        proc_kwargs.setdefault("time_key", time_key)

    rooted = _PROCEDURES[method](g, **proc_kwargs)

    tdata.obst[key_added or tree_key] = rooted

    return tdata if copy else None
