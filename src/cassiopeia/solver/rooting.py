"""Rooting procedures for NJ-type unrooted trees.

Procedures are registered by name and looked up by the ``root`` parameter of
:func:`cassiopeia.solver.nj`.  Custom procedures can be added via
:func:`register`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import networkx as nx

from cassiopeia.mixins import DistanceSolverError

if TYPE_CHECKING:
    from treedata import TreeData

    from cassiopeia.data import CassiopeiaTree

# Registry: name → callable(graph: nx.Graph, **kwargs) → nx.DiGraph
_PROCEDURES: dict[str, Callable] = {}


def register(name: str) -> Callable:
    """Register a rooting procedure under *name*.

    The decorated function must have the signature::

        fn(graph: nx.Graph, **kwargs) -> nx.DiGraph

    where *graph* is the complete undirected NJ tree and the return value is a
    rooted directed tree.
    """

    def decorator(fn: Callable) -> Callable:
        _PROCEDURES[name] = fn
        return fn

    return decorator


def apply(
    graph: nx.Graph,
    procedure: str | None,
    data: CassiopeiaTree | TreeData,
    **kwargs,
) -> nx.DiGraph:
    """Root *graph* using the named procedure and return a directed tree.

    If *procedure* is ``None``, the existing ``root_sample_name`` is used for
    a :class:`~cassiopeia.data.CassiopeiaTree`, or the first obs name for a
    :class:`~treedata.TreeData`.

    Args:
        graph: Complete undirected NJ tree (all edges present).
        procedure: Name of a registered rooting procedure, or ``None`` for
            the default root.
        data: Source data object, used only when *procedure* is ``None`` to
            look up the default root.
        **kwargs: Forwarded to the rooting procedure.

    Returns:
        A rooted :class:`~networkx.DiGraph`.
    """
    if procedure is None:
        root_node = _default_root(data)
        rooted = nx.DiGraph()
        for e in nx.dfs_edges(graph, source=root_node):
            rooted.add_edge(e[0], e[1])
        return rooted
    if procedure not in _PROCEDURES:
        raise ValueError(
            f"Unknown rooting procedure {procedure!r}. Available: {sorted(_PROCEDURES)}"
        )
    return _PROCEDURES[procedure](graph, **kwargs)


def _default_root(data: CassiopeiaTree | TreeData) -> str:
    from treedata import TreeData

    if isinstance(data, TreeData):
        return list(data.obs_names)[0]
    if data.root_sample_name is None:
        raise DistanceSolverError(
            "No root is set on the CassiopeiaTree. Set root_sample_name or pass root='outgroup'."
        )
    return data.root_sample_name


@register("outgroup")
def outgroup(
    graph: nx.Graph,
    outgroup: str | None = None,
) -> nx.DiGraph:
    """Root the NJ tree using an outgroup.

    When *outgroup* is ``None`` (default), a synthetic all-zero leaf named
    ``"root"`` was already included in the NJ run; this function simply roots
    the tree at that node via DFS.

    When *outgroup* is a sample name present in *graph*, a new internal root
    node is inserted on the branch connecting *outgroup* to the rest of the
    tree.  The outgroup remains a leaf in the rooted tree.

    Args:
        graph: Complete undirected NJ tree (including the synthetic leaf if
            *outgroup* is ``None``).
        outgroup: Name of an existing leaf to use as outgroup, or ``None`` to
            use the synthetic ``"root"`` leaf.

    Returns:
        A rooted :class:`~networkx.DiGraph`.
    """
    if outgroup is None:
        if "root" not in graph.nodes:
            raise ValueError(
                "Expected synthetic 'root' node in the NJ tree. "
                "Ensure the 'root' leaf was included in the distance matrix."
            )
        rooted = nx.DiGraph()
        for e in nx.dfs_edges(graph, source="root"):
            rooted.add_edge(e[0], e[1])
        return rooted

    # Named outgroup: insert a new root on the branch between outgroup and
    # its single internal neighbour.
    if outgroup not in graph.nodes:
        raise ValueError(
            f"Outgroup {outgroup!r} not found in the tree. Available nodes: {sorted(graph.nodes)}"
        )
    neighbors = list(graph.neighbors(outgroup))
    if len(neighbors) != 1:
        raise ValueError(
            f"Outgroup {outgroup!r} must be a leaf node "
            f"(expected 1 neighbour, found {len(neighbors)})."
        )
    internal = neighbors[0]

    root_name = "root"
    while root_name in graph.nodes:
        root_name = root_name + "_internal"

    g = graph.copy()
    g.remove_edge(outgroup, internal)
    g.add_node(root_name)
    g.add_edge(root_name, outgroup)
    g.add_edge(root_name, internal)

    rooted = nx.DiGraph()
    for e in nx.dfs_edges(g, source=root_name):
        rooted.add_edge(e[0], e[1])
    return rooted
