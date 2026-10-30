"""Rooting procedures for trees.

Procedures are registered by name and operate on an (unrooted) tree graph,
returning a rooted :class:`~networkx.DiGraph`.  They are used both by
:func:`cassiopeia.solver.nj` (``root=``, on the complete undirected NJ graph)
and by :func:`reroot` (to re-root an already-built tree).  Custom procedures can
be added via :func:`register`.

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


def _default_root(data: CassiopeiaTree | TreeData) -> str | None:
    """A CassiopeiaTree's ``root_sample_name``; ``None`` for a TreeData."""
    from treedata import TreeData

    if isinstance(data, TreeData):
        return None
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


def reroot_on_edge(
    graph: nx.DiGraph,
    new_root: str | None,
    best_edge,
    length_key: str = "length",
    position: float = 0.5,
) -> nx.DiGraph:
    """Insert *new_root* on *best_edge* and orient all edges away from it.

    Node and edge attributes are preserved; the previous root and any resulting
    unifurcations are collapsed (summing branch lengths). If *best_edge* has a
    numeric ``length_key`` attribute, it is split between the two new root edges,
    with fraction *position* on the side of ``best_edge[0]``. If *new_root* is
    ``None``, the new root is named ``"root"``, or ``"root_<i>"`` if a node of
    the rooted tree is already named ``"root"``.

    Raises:
        ValueError: If *new_root* is already a node of *graph*.
    """
    from cassiopeia.utils import _collapse_unifurcations

    if new_root is not None and new_root in graph:
        raise ValueError(f"new_root {new_root!r} is already a node of the tree.")
    auto_name = new_root is None
    if auto_name:
        new_root, i = "root", 0
        while new_root in graph:
            i += 1
            new_root = f"root_{i}"

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

    split = graph.get_edge_data(u, v) or graph.get_edge_data(v, u) or {}
    length = split.get(length_key)
    if isinstance(length, (int, float)) and not isinstance(length, bool):
        rooted[new_root][u][length_key] = position * length
        rooted[new_root][v][length_key] = (1 - position) * length

    rooted = _collapse_unifurcations(rooted, collapse_root=True)
    if auto_name and new_root != "root" and "root" not in rooted:
        # The old root named "root" (e.g. from nj) was collapsed: reuse its name.
        nx.relabel_nodes(rooted, {new_root: "root"}, copy=False)
    return rooted


# ── Procedures ────────────────────────────────────────────────────────────────


@register("outgroup")
def outgroup(
    graph: nx.Graph, outgroup: str | None = None, length_key: str = "length", **kwargs
) -> nx.DiGraph:
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
        length_key: Edge attribute holding branch lengths.
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

    return reroot_on_edge(graph, None, (outgroup, internal), length_key=length_key)


@register("midpoint")
def midpoint(
    graph: nx.Graph, length_key: str = "length", new_root: str | None = None, **kwargs
) -> nx.DiGraph:
    """Root at the midpoint of the longest leaf-to-leaf path.

    Branch lengths are read from the ``length_key`` edge attribute (defaulting to
    ``1.0`` per edge when absent).  The two most distant leaves are found and a
    new root is placed on the edge straddling the halfway point of the path
    between them.

    Args:
        graph: Tree graph (rooted or unrooted).
        length_key: Edge attribute holding branch lengths.
        new_root: Name for the inserted root node (default ``"root"``, or
            ``"root_<i>"`` if taken).
        kwargs: Keyword arguments.

    Returns:
        A rooted :class:`~networkx.DiGraph`.
    """
    weighted = nx.Graph()
    weighted.add_nodes_from(graph.nodes())
    for u, v in graph.to_undirected().edges():
        attrs = graph.get_edge_data(u, v) or graph.get_edge_data(v, u) or {}
        weighted.add_edge(u, v, weight=attrs.get(length_key, 1.0))

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
    position = 0.5
    for i in range(len(path) - 1):
        w = weighted[path[i]][path[i + 1]]["weight"]
        if acc + w >= half:
            best_edge = (path[i], path[i + 1])
            position = (half - acc) / w if w > 0 else 0.5
            break
        acc += w

    return reroot_on_edge(graph, new_root, best_edge, length_key=length_key, position=position)


@register("centroid")
def centroid(
    graph: nx.Graph, new_root: str | None = None, length_key: str = "length", **kwargs
) -> nx.DiGraph:
    """Root on the edge that most evenly splits the leaves.

    Finds the edge whose two sides have the closest leaf counts, inserts a new
    root there, and orients edges away from it.

    Args:
        graph: Tree graph (rooted or unrooted).
        new_root: Name for the inserted root node (default ``"root"``, or
            ``"root_<i>"`` if taken).
        length_key: Edge attribute holding branch lengths.
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

    return reroot_on_edge(graph, new_root, best_edge, length_key=length_key)


def _get_mutation_outgroup(characters: pd.DataFrame, missing_state=-1, unedited_state=0) -> list:
    """Identify outgroup leaves as those sharing the single most common mutation."""
    from cassiopeia.utils import _normalize_missing

    excluded = list(_normalize_missing(missing_state) | _normalize_missing(unedited_state))
    best_col = None
    best_value = None
    best_prop = -1
    for col in characters.columns:
        col_values = characters[col][~characters[col].isin(excluded)]
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
    new_root: str | None = None,
    missing_state=-1,
    unedited_state=0,
    length_key: str = "length",
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
        new_root: Name for the inserted root node (default ``"root"``, or
            ``"root_<i>"`` if taken).
        missing_state: Value (or sequence of values) representing missing data.
        unedited_state: Value (or sequence of values) representing the unedited
            state.
        length_key: Edge attribute holding branch lengths.
        kwargs: Keyword arguments.

    Returns:
        A rooted :class:`~networkx.DiGraph`.
    """
    if characters is None:
        raise ValueError("shared_mutation rooting requires a character matrix.")

    undirected = graph.to_undirected()
    outgroup_leaves = set(_get_mutation_outgroup(characters, missing_state, unedited_state))
    if not outgroup_leaves:
        return centroid(graph, new_root=new_root, length_key=length_key)

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

    return reroot_on_edge(graph, new_root, best_edge, length_key=length_key)


# ── Public reroot ─────────────────────────────────────────────────────────────


def reroot(
    tdata: TreeData,
    method: str = "centroid",
    tree_key: str | None = None,
    characters_key: str = "characters",
    length_key: str = "length",
    time_key: str = "time",
    key_added: str | None = None,
    copy: bool = False,
    **kwargs,
) -> TreeData | None:
    """Re-root an already-built tree using a registered rooting procedure.

    Branch lengths are read from the ``length_key`` edge attribute; edges
    without it take their length from the ``time_key`` node attribute (only one
    is needed), or unit length if neither is available. Lengths are preserved
    across the re-rooting (the root edge is split, and the old root is collapsed
    by summing its edges). A unifurcating root and its branch are removed first.
    If the input tree has node times, they are recomputed as the distance from
    the new root, offset by the old root's time.

    Args:
        tdata: TreeData containing the tree to re-root.
        method: Rooting procedure name — one of ``'centroid'``, ``'midpoint'``,
            ``'outgroup'``, ``'shared_mutation'``.
        tree_key: Key in ``tdata.obst`` of the tree to re-root.
        characters_key: Key in ``tdata.obsm`` for the character matrix (used by
            ``shared_mutation``).
        length_key: Edge attribute holding branch lengths.
        time_key: Node attribute holding node times.
        key_added: Key in ``tdata.obst`` to store the re-rooted tree under.
            Defaults to *tree_key* (overwriting in place).
        copy: If ``True``, operate on and return a copy of *tdata*; otherwise
            modify in place and return ``None``.
        **kwargs: Forwarded to the rooting procedure (e.g. ``outgroup=...``).

    Returns:
        A modified copy of *tdata* if ``copy=True``, else ``None``.

    Raises:
        TypeError: If *tdata* is not a TreeData object.
        ValueError: If *method* is not a registered procedure, or
            ``method='outgroup'`` without an ``outgroup`` leaf name.
    """
    from treedata import TreeData

    from cassiopeia.utils import _get_characters, _get_digraph, _get_parameter, _set_tree

    if not isinstance(tdata, TreeData):
        raise TypeError(
            "reroot() operates on TreeData. For a CassiopeiaTree, convert with "
            "CassiopeiaTree.to_treedata()."
        )
    if method not in _PROCEDURES:
        raise ValueError(f"Unknown rooting procedure {method!r}. Available: {sorted(_PROCEDURES)}")
    if method == "outgroup" and kwargs.get("outgroup") is None:
        raise ValueError("method='outgroup' requires the name of an outgroup leaf (outgroup=...).")

    tdata = tdata.copy() if copy else tdata
    g, tree_key = _get_digraph(tdata, tree_key, copy=True)

    # A unifurcating root (e.g. of simulated trees) would be a leaf of the
    # unrooted tree, so its branch is dropped.
    root = next(n for n in g if g.in_degree(n) == 0)
    while g.out_degree(root) == 1 and len(g) > 2:
        child = next(iter(g.successors(root)))
        g.remove_node(root)
        root = child

    has_time = all(time_key in d for _, d in g.nodes(data=True))
    if has_time:
        root_time = g.nodes[root][time_key]
        for u, v, d in g.edges(data=True):
            if length_key not in d:
                d[length_key] = g.nodes[v][time_key] - g.nodes[u][time_key]

    proc_kwargs = {"length_key": length_key, **kwargs}
    if method == "shared_mutation":
        if proc_kwargs.get("characters") is None:
            proc_kwargs["characters"] = _get_characters(tdata, characters_key)
        proc_kwargs.setdefault("missing_state", _get_parameter(tdata, "missing_state"))
        proc_kwargs.setdefault("unedited_state", _get_parameter(tdata, "unmodified_state"))

    rooted = _PROCEDURES[method](g, **proc_kwargs)

    if has_time:
        root = next(n for n in rooted if rooted.in_degree(n) == 0)
        rooted.nodes[root][time_key] = root_time
        for u, v in nx.bfs_edges(rooted, root):
            rooted.nodes[v][time_key] = rooted.nodes[u][time_key] + rooted[u][v].get(
                length_key, 1.0
            )

    _set_tree(tdata, rooted, key_added or tree_key)

    return tdata if copy else None
