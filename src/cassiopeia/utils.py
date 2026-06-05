"""Utility functions for working with tree data structures."""

from __future__ import annotations

import random
import time
import warnings
from collections.abc import Generator
from hashlib import blake2b
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from treedata import TreeData

from cassiopeia.data.CassiopeiaTree import CassiopeiaTree
from cassiopeia.mixins.errors import (
    CassiopeiaError,
    PriorTransformationError,
)


def _get_characters(
    tree: CassiopeiaTree | TreeData,
    key: str | None = None,
    **kwargs,
) -> pd.DataFrame | None:
    """Return the character matrix from a tree-like object.

    Args:
        tree: A :class:`~cassiopeia.data.CassiopeiaTree`, :class:`~treedata.TreeData`,
            or ``nx.DiGraph``.
        key: For :class:`~treedata.TreeData`, the ``obsm`` key to look up
            (default ``"characters"``).  For :class:`~cassiopeia.data.CassiopeiaTree`,
            the layer name (default: the primary character matrix).
        kwargs: Deprecated argument ``layer`` is also accepted as an alias for ``key`` for backward compatibility.

    Returns:
        pd.DataFrame or None if no character matrix is available.
    """
    if "layer" in kwargs:
        warnings.warn(
            "'layer' is deprecated and will be removed in a future version. "
            "Use 'characters_key' instead.",
            DeprecationWarning,
            stacklevel=3,
        )
        key = kwargs.pop("layer")

    characters = None
    if isinstance(tree, TreeData):
        characters = tree.obsm.get(key or "characters", None)
    if isinstance(tree, CassiopeiaTree):
        if key == "characters":
            characters = tree.character_matrix
        else:
            characters = tree.layers[key]
    if isinstance(characters, np.ndarray):
        characters = pd.DataFrame(characters)

    if characters is None and isinstance(tree, TreeData):
        raise ValueError(f"No character matrix found; store characters in obsm[{key}]")

    return characters


def _set_characters(
    data: CassiopeiaTree | TreeData, characters: pd.DataFrame, characters_key: str = "characters"
):
    """Set the character matrix in a CassiopeiaTree or TreeData.

    Args:
        data: CassiopeiaTree or TreeData to modify in-place.
        characters: Character matrix to set.
        characters_key: Key under which to store characters.
    """
    if isinstance(data, CassiopeiaTree):
        if characters_key == "characters":
            data.character_matrix = characters
        else:
            data.layers[characters_key] = characters
    elif isinstance(data, TreeData):
        data.obsm[characters_key] = characters


def _get_digraph(
    tree: CassiopeiaTree | TreeData | nx.DiGraph, tree_key: str | None = None, copy=False
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
            "CassiopeiaTree is deprecated and will be removed in v3.1.0. "
            "Please convert to TreeData using CassiopeiaTree.to_treedata().",
            DeprecationWarning,
            stacklevel=2,
        )
        t = tree.get_tree_topology()

    elif isinstance(tree, TreeData):
        keys = list(tree.obst.keys())
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


def _set_tree(data: CassiopeiaTree | TreeData, g: nx.DiGraph, tree_key: str | None = None):
    """Store DiGraph, handling both CassiopeiaTree and TreeData.

    Args:
        data: CassiopeiaTree or TreeData to modify in-place.
        g: Directed tree DiGraph to store.
        tree_key: obst key for TreeData.
    """
    # Annotate the rooted tree with a per-node ``depth`` attribute (edges from root)
    # so every solver's output tree carries a depth key.
    _add_depth(g)

    if isinstance(data, TreeData):
        data.obst[tree_key] = g
    else:
        data.root_sample_name = _get_root(g)
        data.populate_tree(g)


def _get_root(g: nx.DiGraph) -> str:
    """Return the unique root of a directed tree graph.

    Args:
        g: A directed tree graph.

    Returns:
        str: The node label of the root.

    Raises:
        ValueError: If the graph does not contain exactly one root.
    """
    roots = [node for node in g.nodes if g.in_degree(node) == 0]

    if not roots:
        raise ValueError("Tree does not have a root.")
    if len(roots) > 1:
        raise ValueError("Tree has multiple roots; expected a single rooted tree.")

    return roots[0]


def _add_depth(tree: nx.DiGraph, depth_key: str = "depth") -> None:
    """Annotate each node of a rooted ``nx.DiGraph`` with its depth, in place.

    Depth is the number of edges from the root (root depth ``0``).  Called by the
    solvers after rooting so the output tree carries a ``depth`` node attribute.

    Args:
        tree: A rooted directed tree.
        depth_key: Node attribute key under which the depth is stored.
    """
    roots = [n for n in tree.nodes if tree.in_degree(n) == 0]
    if not roots:
        return
    for node, depth in nx.single_source_shortest_path_length(tree, roots[0]).items():
        tree.nodes[node][depth_key] = depth


def _collapse_unifurcations(
    tree: CassiopeiaTree | TreeData | nx.DiGraph,
    tree_key: str | None = None,
    inplace: bool = False,
    collapse_root: bool = True,
) -> nx.DiGraph:
    """Return a copy of ``tree`` with all unifurcations collapsed.

    Internal nodes with exactly one child are removed and their parent and child
    are connected directly. Numeric edge attributes (e.g. branch lengths) are
    summed so that additive quantities between the parent and child are preserved.

    Args:
        tree: The tree object.
        tree_key: The `obst` key to use when ``tree`` is a :class:`treedata.TreeData`.
            Only required if multiple trees are present.
        inplace: Whether to modify the graph in place or return a new graph.
        collapse_root: When ``True`` (default), collapse the root's single child
            into the root if the root is a unifurcation. When ``False``, the
            root's direct child is preserved even if it is a unifurcation.

    Returns:
        nx.DiGraph: A directed graph with all unifurcations collapsed.

    Raises:
        ValueError: If a unifurcating node lacks a unique parent.
    """
    copy = True if not inplace or isinstance(tree, TreeData) else False
    t, tree_key = _get_digraph(tree, tree_key=tree_key, copy=copy)
    if len(t) <= 2:
        return t

    root = _get_root(t)

    for node in reversed(list(nx.topological_sort(t))):
        children = list(t.successors(node))
        if len(children) != 1:
            continue
        child = children[0]
        # Root case: bypass a single child by wiring root -> grandchildren
        if node == root:
            if not collapse_root:
                continue
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


def _is_number(value: Any) -> bool:
    """Return ``True`` for real numeric scalars (ints/floats), excluding bools."""
    return isinstance(value, (int, float, np.number)) and not isinstance(value, bool)


def _combine_edge_data(parent_edge: dict[str, Any], child_edge: dict[str, Any]) -> dict[str, Any]:
    """Merge edge metadata when splicing out a node, summing numeric attributes.

    Numeric attributes (e.g. branch lengths) are summed across the two edges so
    additive quantities are preserved across the removed node; non-numeric
    attributes take the child edge's value.
    """
    new_edge = dict(child_edge)
    for key, parent_value in parent_edge.items():
        child_value = new_edge.get(key)
        if _is_number(parent_value) and (child_value is None or _is_number(child_value)):
            new_edge[key] = parent_value + (child_value or 0)
        elif key not in new_edge:
            new_edge[key] = parent_value
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


def _get_parameter(tree: CassiopeiaTree | TreeData, param_name: str, value=None):
    """Get a parameter from CassiopeiaTree or TreeData."""
    if value is not None:
        return value
    if isinstance(tree, CassiopeiaTree):
        if param_name == "missing_state":
            value = tree.missing_state_indicator
        elif param_name == "priors":
            value = tree.priors
        else:
            value = tree.parameters.get(param_name, None)
    elif isinstance(tree, TreeData):
        value = tree.uns.get(param_name, None)
    fallbacks = {
        "missing_state": (-1, "-1", "NA", "-"),
        "unmodified_state": (0, "0", "*"),
    }
    if value is None and param_name in fallbacks:
        value = fallbacks[param_name]
        warnings.warn(
            f"Parameter '{param_name}' not specified; using default value {value}.",
            UserWarning,
            stacklevel=3,
        )
    return value


def _normalize_missing(missing_state) -> set:
    """Normalize a missing-state value (scalar or sequence) into a set of values."""
    if isinstance(missing_state, (list, tuple, set)):
        return set(missing_state)
    return {missing_state}


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


def _get_leaf_data(g: nx.DiGraph, key: str) -> dict[str, Any]:
    """Get a dictionary mapping leaf node labels to a specified node attribute."""
    leaf_data = {}
    for node in g.nodes:
        if g.out_degree(node) == 0:  # Check if node is a leaf
            leaf_data[node] = g.nodes[node].get(key)
    return pd.Series(leaf_data)


def _node_name_generator() -> Generator[str, None, None]:
    """Yield unique internal node names for building reconstructed trees.

    Produces unique names by hashing timestamps.
    """
    while True:
        k = str(time.time()).encode("utf-8")
        h = blake2b(key=k, digest_size=12)
        yield "cassiopeia_internal_node" + h.hexdigest()


def _transform_priors(
    priors: dict[int, dict[int, float]] | None,
    prior_transformation: str = "negative_log",
) -> dict[int, dict[int, float]]:
    """Generate a dictionary of weights from priors.

    Generates a dictionary of weights from given priors for each character/state
    pair. Supported transformations include negative log, inverse, and square
    root inverse.

    Args:
        priors: A dictionary of prior probabilities for each character/state pair.
        prior_transformation: A function defining a transformation on the priors
            in forming weights. Supports the following transformations:
                "negative_log": Transforms each probability by the negative log
                "inverse": Transforms each probability p by taking 1/p
                "square_root_inverse": Transforms each probability by the
                    square root of 1/p

    Returns:
        A dictionary of weights for each character/state pair.

    Raises:
        PriorTransformationError: If *prior_transformation* is unsupported or a
            prior is not in ``(0, 1]``.
    """
    if prior_transformation not in [
        "negative_log",
        "inverse",
        "square_root_inverse",
    ]:
        raise PriorTransformationError("Please select one of the supported prior transformations.")

    prior_function = lambda x: -np.log(x)

    if prior_transformation == "square_root_inverse":
        prior_function = lambda x: (np.sqrt(1 / x))
    if prior_transformation == "inverse":
        prior_function = lambda x: 1 / x

    weights = {}
    for character in priors:
        state_weights = {}
        for state in priors[character]:
            p = priors[character][state]
            if p <= 0.0 or p > 1.0:
                raise PriorTransformationError(
                    "Please make sure all priors have a value between 0 and 1"
                )
            state_weights[state] = prior_function(p)
        weights[character] = state_weights
    return weights


def _save_dissimilarity(
    data: CassiopeiaTree | TreeData,
    dist_df: pd.DataFrame,
    dissim_key: str | None = None,
) -> None:
    """Store a pairwise dissimilarity map on a tree-like object.

    For :class:`~treedata.TreeData` the dense matrix is written to
    ``obsp[dissim_key or 'distances']``; for
    :class:`~cassiopeia.data.CassiopeiaTree` it is set via
    ``set_dissimilarity_map``.

    Args:
        data: CassiopeiaTree or TreeData to modify in-place.
        dist_df: Symmetric distance ``pd.DataFrame`` indexed by sample name.
        dissim_key: ``obsp`` key for the stored matrix (TreeData only).
    """
    if isinstance(data, TreeData):
        names = list(data.obs_names)
        data.obsp[dissim_key or "distances"] = dist_df.loc[names, names].to_numpy()
    else:
        data.set_dissimilarity_map(dist_df)
