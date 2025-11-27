"""Utility functions for working with tree data structures."""

from __future__ import annotations

import random
import warnings
from collections.abc import Sequence
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from treedata import TreeData

from cassiopeia.data import CassiopeiaTree
from cassiopeia.mixins.errors import (
    CassiopeiaError,
)

from .typing import TreeLike


def _get_digraph(
    tree: TreeLike, tree_key: str | None = None, copy=False
) -> tuple[nx.DiGraph, str | None]:
    """Logic for getting `nx.DiGraph` from inputs.

    Args:
        tree: tree-like object. One of `nx.DiGraph`, `CassiopeiaTree`, or `TreeData`.
        tree_key: The `obst` key to use when ``tree`` is a :class:`treedata.TreeData`.
            Only required if multiple trees are present.
        copy: Whether to return a copy of the graph.

    Returns:
        nx.DiGraph: A directed graph representation of the input tree
        str: The tree key used if applicable.

    Raises:
        TypeError: If ``tree`` is not a supported type.
        ValueError: If ``tree`` is a :class:`treedata.TreeData` and no tree can
            be resolved from ``obst`` with the provided ``tree_key``.
    """
    if isinstance(tree, nx.DiGraph):
        t = tree

    elif isinstance(tree, CassiopeiaTree):
        warnings.warn(
            "CassiopeiaTree is deprecated and will be removed in v3.1.0."
            "Please convert to TreeData using CassiopeiaTree.to_treedata().",
            DeprecationWarning,
            stacklevel=2,
        )
        t = tree.get_tree_topology()

    elif isinstance(tree, TreeData):
        keys = list(tree.obst_keys())
        if not keys:
            raise ValueError("TreeData object does not contain any trees in 'obst'.")

        if tree_key is None:
            if len(keys) > 1:
                raise ValueError(
                    "TreeData contains multiple trees. Please specify the tree using `tree_key`."
                )
            tree_key = keys[0]

        if tree_key not in tree.obst:
            raise ValueError(f"Key '{tree_key}' not found in TreeData.obst.")

        t = tree.obst[tree_key]

    else:
        raise TypeError(
            f"Unsupported tree type {type(tree)}. Must be one of: TreeData, nx.DiGraph, CassiopeiaTree."
        )
    if copy:
        t = t.copy()

    return t, tree_key


def get_leaves(tree: TreeLike, tree_key: str | None = None) -> list[str]:
    """Return the leaf labels of a tree.

    Args:
        tree: The tree object.
        tree_key: The `obst` key to use when ``tree`` is a :class:`treedata.TreeData`.
            Only required if multiple trees are present.

    Returns:
        list[str]: Sorted leaf labels.
    """
    t, _ = _get_digraph(tree, tree_key=tree_key)
    leaves = [node for node in t.nodes if t.out_degree(node) == 0]
    return sorted(leaves)


def get_root(tree: TreeLike, tree_key: str | None = None) -> str:
    """Return the unique root of a tree.

    Args:
        tree: The tree object.
        tree_key: The `obst` key to use when ``tree`` is a :class:`treedata.TreeData`.
            Only required if multiple trees are present.

    Returns:
        str: The node label of the root.

    Raises:
        ValueError: If the tree does not contain exactly one root.
    """
    t, _ = _get_digraph(tree, tree_key=tree_key)
    roots = [node for node in t.nodes if t.in_degree(node) == 0]

    if not roots:
        raise ValueError("Tree does not have a root.")
    if len(roots) > 1:
        raise ValueError("Tree has multiple roots; expected a single rooted tree.")

    return roots[0]


def collapse_unifurcations(
    tree: TreeLike, tree_key: str | None = None, inplace: bool = False
) -> nx.DiGraph:
    """Return a copy of ``tree`` with all unifurcations collapsed.

    Internal nodes with exactly one child are removed and their parent and child
    are connected directly. When branch length metadata is present, lengths are
    summed so that the total distance between the parent and child is preserved.

    Args:
        tree: The tree object.
        tree_key: The `obst` key to use when ``tree`` is a :class:`treedata.TreeData`.
            Only required if multiple trees are present.
        inplace: Whether to modify the graph in place or return a new graph.

    Returns:
        nx.DiGraph: A directed graph with all unifurcations collapsed.

    Raises:
        ValueError: If a unifurcating node lacks a unique parent.
    """
    copy = True if not inplace or isinstance(tree, TreeData) else False
    t, tree_key = _get_digraph(tree, tree_key=tree_key, copy=copy)
    if len(t) <= 2:
        return t

    root = get_root(t)

    for node in reversed(list(nx.topological_sort(t))):
        children = list(t.successors(node))
        if len(children) != 1:
            continue
        child = children[0]
        # Root case: bypass a single child by wiring root -> grandchildren
        if node == root:
            parent_edge = dict(t.get_edge_data(node, child, default={}))
            for gc in list(t.successors(child)):
                child_edge = dict(t.get_edge_data(child, gc, default={}))
                t.add_edge(node, gc, **_combine_edge_data(parent_edge, child_edge))
            t.remove_node(child)
            continue
        # Non-root: splice node out between its unique parent and its child
        parents = list(t.predecessors(node))
        if len(parents) != 1:
            raise ValueError(
                "Unifurcating node does not have a unique parent; expected a rooted tree."
            )
        parent = parents[0]
        parent_edge = dict(t.get_edge_data(parent, node, default={}))
        child_edge = dict(t.get_edge_data(node, child, default={}))
        # Remove node, then connect parent -> child with combined edge data
        t.remove_node(node)
        t.add_edge(parent, child, **_combine_edge_data(parent_edge, child_edge))

    if inplace:
        if isinstance(tree, TreeData):
            tree.obst[tree_key] = t
    else:
        return t


def _combine_edge_data(parent_edge: dict[str, Any], child_edge: dict[str, Any]) -> dict[str, Any]:
    """Merge edge metadata while preserving branch lengths."""
    new_edge = child_edge.copy()
    if "length" in parent_edge or "length" in child_edge:
        parent_length = parent_edge.get("length", 0)
        child_length = child_edge.get("length", 0)
        new_edge["length"] = parent_length + child_length
    return new_edge


def _get_cell_meta(tree: CassiopeiaTree | TreeData) -> pd.DataFrame:
    """Return the cell metadata DataFrame from a CassiopeiaTree or TreeData.

    For CassiopeiaTree, this is `tree.cell_meta`.
    For TreeData, this is `tree.obs`.
    Raises a CassiopeiaError if neither attribute exists.
    """
    if isinstance(tree, CassiopeiaTree) and isinstance(tree.cell_meta, pd.DataFrame):
        return tree.cell_meta
    if isinstance(tree, TreeData) and isinstance(tree.obs, pd.DataFrame):
        return tree.obs
    raise CassiopeiaError(
        "Tree object does not have .cell_meta (CassiopeiaTree) or .obs (TreeData)."
    )


def _set_attribute_treelike(
    tree: TreeLike, node: str, attribute_name: str, value: Any | None = None
) -> None:
    """Sets an attribute in the tree.

    Args:
        tree: The tree object.
        node: Node name
        attribute_name: Name for the new attribute
        value: Value for the attribute.


    Raises:
        CassiopeiaTreeError if the tree has not been initialized.
        KeyError if the node is not found in the tree.
        TypeError if the tree type is unsupported.
    """
    if isinstance(tree, nx.DiGraph):
        if node not in tree.nodes:
            raise KeyError(f"Node {node} not found in DiGraph.")
        nx.set_node_attributes(tree, {node: {attribute_name: value}})

    elif isinstance(tree, CassiopeiaTree):
        tree._CassiopeiaTree__check_network_initialized()
        if node not in tree._CassiopeiaTree__network.nodes:
            raise KeyError(f"Node {node} not found in CassiopeiaTree.")
        tree._CassiopeiaTree__network.nodes[node][attribute_name] = value

    elif isinstance(tree, TreeData):
        if node not in tree.obs_names:
            raise KeyError(f"Node {node} not found in TreeData.")
        tree.obs.loc[node, attribute_name] = value

    else:
        raise TypeError("Unsupported tree type. Must be CassiopeiaTree or TreeData.")


def _get_attribute_treelike(tree: TreeLike, node: str, attribute_name: str) -> Any:
    """Retrieves the value of an attribute for a node.

    Args:
        tree: The tree object.
        node: Node name
        attribute_name: Name of the attribute.

    Returns:
        The value of the attribute for that node.

    Raises:
        CassiopeiaTreeError if the attribute has not been set for this node.
        KeyError if the node is not found in the tree.
        TypeError if the tree type is unsupported.
    """
    if isinstance(tree, nx.DiGraph):
        try:
            return tree.nodes[node][attribute_name]
        except KeyError as error:
            raise KeyError(f"Attribute {attribute_name} not detected for node {node}.") from error
    elif isinstance(tree, CassiopeiaTree):
        tree._CassiopeiaTree__check_network_initialized()
        try:
            return tree._CassiopeiaTree__network.nodes[node][attribute_name]
        except KeyError as error:
            raise KeyError(f"Attribute {attribute_name} not detected for node {node}.") from error

    elif isinstance(tree, TreeData):
        if attribute_name not in tree.obs.columns:
            raise KeyError(f"Attribute {attribute_name} not detected in TreeData.obs.")
        if node not in tree.obs_names:
            raise KeyError(f"Node {node} not found in TreeData.")
        return tree.obs.loc[node, attribute_name]

    else:
        raise TypeError("Unsupported tree type. Must be CassiopeiaTree or TreeData.")


def _get_children_treelike(tree: TreeLike, node: str, tree_key: str | None = None) -> list[str]:
    """Gets the children of a given node for any tree-like object.

    Args:
        tree: The tree object.
        node: A node in the tree.
        tree_key: Optional obst key if `tree` is a TreeData containing multiple trees.

    Returns:
        A list of nodes that are direct children of the input node.

    Raises:
        KeyError: If the node does not exist in the tree.
        TypeError: If the input tree type is unsupported.
        ValueError: If a TreeData has multiple trees and no tree_key is given.
    """
    G, _ = _get_digraph(tree, tree_key)

    if node not in G:
        raise KeyError(f"Node {node} not found in tree.")

    return list(G.successors(node))


def _get_character_matrix(
    tree: CassiopeiaTree | TreeData, characters_key: str = "characters", **kwargs
) -> np.ndarray:
    """Get character matrix from a tree object."""
    if "layer" in kwargs:
        warnings.warn(
            "'layer' is deprecated and will be removed in a future version. "
            "Use 'characters_key' instead.",
            DeprecationWarning,
            stacklevel=3,
        )
        characters_key = kwargs.pop("layer")

    if isinstance(tree, CassiopeiaTree):
        if characters_key == "characters":
            character_matrix = tree.character_matrix
        else:
            character_matrix = tree.layers[characters_key]
    elif isinstance(tree, TreeData):
        character_matrix = tree.obsm[characters_key]

    if isinstance(character_matrix, np.ndarray):
        character_matrix = pd.DataFrame(character_matrix)
    return character_matrix


def _get_missing_state_indicator(
    tree: CassiopeiaTree | TreeData,
    missing_state: str | int | Sequence[str | int] | None = (-1, "-1", "NA", "-"),
) -> str | int | Sequence[str | int] | None:
    user_provided = missing_state != (-1, "-1", "NA", "-")
    tree_value = None
    if isinstance(tree, CassiopeiaTree):
        tree_value = tree.missing_state_indicator
    elif isinstance(tree, TreeData):
        if "missing_state_indicator" in tree.uns:
            tree_value = tree.uns["missing_state_indicator"]
    if user_provided and tree_value is not None and missing_state != tree_value:
        warnings.warn(
            f"User-provided missing_state ({missing_state}) differs from tree's "
            f"missing_state_indicator ({tree_value}). Using user-provided value.",
            UserWarning,
            stacklevel=3,
        )
        return missing_state
    return tree_value if tree_value is not None else missing_state


def _get_tree_parameter(tree: CassiopeiaTree | TreeData, param_name: str, default=None):
    """Get a parameter from CassiopeiaTree or TreeData."""
    if isinstance(tree, CassiopeiaTree):
        return tree.parameters.get(param_name, default)
    elif isinstance(tree, TreeData):
        return tree.uns.get(param_name, default)
    return default


def get_mean_depth(tree: TreeLike, depth_key: str, tree_key: str | None = None) -> float:
    """Compute the mean depth of a tree's leaves.

    Calculates the average depth across all leaf nodes in the tree. Depth is
    retrieved from the node attribute specified by depth_key. This can represent
    either discrete generations (e.g., number of divisions) or continuous time
    (e.g., evolutionary time).

    Args:
        tree: Tree object (CassiopeiaTree, TreeData, or nx.DiGraph)
        depth_key: Node attribute key containing depth values (e.g., "depth", "time")
        tree_key: Tree key to use if tree is a TreeData object with multiple trees

    Returns:
        float: Mean depth of the tree's leaves
    """
    t, _ = _get_digraph(tree, tree_key=tree_key)
    _check_tree_has_key(t, depth_key)
    leaves = get_leaves(tree, tree_key=tree_key)
    depths = [t.nodes[leaf][depth_key] for leaf in leaves]
    return float(np.mean(depths))


def _check_tree_has_key(tree: nx.DiGraph, key: str):
    """Checks that tree nodes have a given key.

    Args:
        tree: NetworkX DiGraph
        key: Node attribute key to check for

    Raises:
        ValueError: If key is not present in one or more nodes
    """
    sampled_nodes = random.sample(list(tree.nodes), min(10, len(tree.nodes)))
    for node in sampled_nodes:
        if key not in tree.nodes[node]:
            message = f"One or more nodes do not have '{key}' attribute."
            raise ValueError(message)
