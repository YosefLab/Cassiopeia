"""Utility functions for working with tree data structures."""

from __future__ import annotations

from typing import Any, Hashable

import networkx as nx
from treedata import TreeData

from cassiopeia.data import CassiopeiaTree


def _to_networkx(tree: Any, key: str | None = None) -> nx.DiGraph:
    """Convert a supported tree representation to a ``networkx.DiGraph``.

    Args:
        tree: A tree-like object. Supported inputs include
            :class:`networkx.DiGraph`, :class:`cassiopeia.data.CassiopeiaTree`,
            and :class:`treedata.TreeData`.
        key: Optional key identifying which tree to extract when ``tree`` is a
            :class:`treedata.TreeData` with multiple entries in ``obst``.

    Returns:
        nx.DiGraph: A copy of ``tree`` represented as a directed NetworkX graph.

    Raises:
        TypeError: If ``tree`` is not a supported type.
        ValueError: If ``tree`` is a :class:`treedata.TreeData` and no tree can
            be resolved from ``obst`` with the provided ``key``.
    """

    if isinstance(tree, nx.DiGraph):
        return tree.copy()

    if isinstance(tree, CassiopeiaTree):
        return tree.get_tree_topology()

    if isinstance(tree, TreeData):
        if not hasattr(tree, "obst"):
            raise ValueError("TreeData object is missing an 'obst' attribute.")

        keys = list(tree.obst_keys())
        if not keys:
            raise ValueError("TreeData object does not contain any trees in 'obst'.")

        if key is None:
            if len(keys) > 1:
                raise ValueError(
                    "TreeData contains multiple trees. Please specify the key to use."
                )
            key = keys[0]

        if key not in tree.obst:
            raise ValueError(f"Key '{key}' not found in TreeData obst.")

        return _to_networkx(tree.obst[key])

    raise TypeError(
        "Unsupported tree type. Expected networkx.DiGraph, CassiopeiaTree, or TreeData."
    )


def get_leaves(tree: Any, key: str | None = None) -> list[Hashable]:
    """Return the leaf labels of a tree.

    Args:
        tree: A tree-like object supported by :func:`_to_networkx`.
        key: Optional key used when ``tree`` is a :class:`treedata.TreeData`.

    Returns:
        list[Hashable]: Sorted leaf labels.
    """

    graph = tree if isinstance(tree, nx.DiGraph) else _to_networkx(tree, key)
    leaves = [node for node in graph.nodes if graph.out_degree(node) == 0]
    return sorted(leaves)


def get_root(tree: Any, key: str | None = None) -> Hashable:
    """Return the unique root of a tree.

    Args:
        tree: A tree-like object supported by :func:`_to_networkx`.
        key: Optional key used when ``tree`` is a :class:`treedata.TreeData`.

    Returns:
        Hashable: The node label of the root.

    Raises:
        ValueError: If the tree does not contain exactly one root.
    """

    graph = tree if isinstance(tree, nx.DiGraph) else _to_networkx(tree, key)
    roots = [node for node in graph.nodes if graph.in_degree(node) == 0]

    if not roots:
        raise ValueError("Tree does not have a root.")
    if len(roots) > 1:
        raise ValueError("Tree has multiple roots; expected a single rooted tree.")

    return roots[0]


def collapse_unifurcations(tree: Any, key: str | None = None) -> nx.DiGraph:
    """Return a copy of ``tree`` with all unifurcations collapsed.

    Internal nodes with exactly one child are removed and their parent and child
    are connected directly. When branch length metadata is present, lengths are
    summed so that the total distance between the parent and child is preserved.

    Args:
        tree: A tree-like object supported by :func:`_to_networkx`.
        key: Optional key used when ``tree`` is a :class:`treedata.TreeData`.

    Returns:
        nx.DiGraph: A directed graph with all unifurcations collapsed.

    Raises:
        ValueError: If a unifurcating node lacks a unique parent.
    """

    graph = _to_networkx(tree, key)
    if len(graph) <= 2:
        return graph

    root = get_root(graph)
    for node in list(nx.topological_sort(graph))[::-1]:
        if node not in graph:
            continue

        children = list(graph.successors(node))
        if node == root:
            if len(children) == 1:
                child = children[0]
                parent_edge = dict(graph.get_edge_data(node, child, default={}))
                grandchildren = list(graph.successors(child))
                for grandchild in grandchildren:
                    child_edge = dict(graph.get_edge_data(child, grandchild, default={}))
                    new_edge = _combine_edge_data(parent_edge, child_edge)
                    graph.add_edge(node, grandchild, **new_edge)
                graph.remove_node(child)
            continue

        if len(children) == 1:
            child = children[0]
            parents = list(graph.predecessors(node))
            if len(parents) != 1:
                raise ValueError(
                    "Unifurcating node does not have a unique parent; expected a rooted tree."
                )
            parent = parents[0]
            parent_edge = dict(graph.get_edge_data(parent, node, default={}))
            child_edge = dict(graph.get_edge_data(node, child, default={}))
            new_edge = _combine_edge_data(parent_edge, child_edge)
            graph.remove_node(node)
            graph.add_edge(parent, child, **new_edge)

    return graph


def _combine_edge_data(parent_edge: dict[str, Any], child_edge: dict[str, Any]) -> dict[str, Any]:
    """Merge edge metadata while preserving branch lengths."""

    new_edge = child_edge.copy()
    if "length" in parent_edge or "length" in child_edge:
        parent_length = parent_edge.get("length", 0)
        child_length = child_edge.get("length", 0)
        new_edge["length"] = parent_length + child_length
    return new_edge
