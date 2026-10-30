"""Utility functions for working with tree data structures."""

from __future__ import annotations

import itertools
import random
import secrets
import warnings
from collections.abc import Generator, Mapping, Sequence
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
from cassiopeia.mixins.logging import logger


def _get_characters(
    tree: CassiopeiaTree | TreeData,
    key: str | None = None,
    missing_state=None,
    **kwargs,
) -> pd.DataFrame | None:
    """Return the character matrix from a tree-like object.

    Empty (NaN/None) entries of a TreeData's matrix are treated as missing data
    (see :func:`_fill_missing`).

    Args:
        tree: A :class:`~cassiopeia.data.CassiopeiaTree`, :class:`~treedata.TreeData`,
            or ``nx.DiGraph``.
        key: For :class:`~treedata.TreeData`, the ``obsm`` key to look up
            (default ``"characters"``).  For :class:`~cassiopeia.data.CassiopeiaTree`,
            the layer name (default: the primary character matrix).
        missing_state: Missing state used for empty entries; ``None`` uses
            ``tree.uns["missing_state"]``.
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
        index = tree.obs_names if isinstance(tree, TreeData) else None
        characters = pd.DataFrame(characters, index=index)

    if characters is None and isinstance(tree, TreeData):
        raise ValueError(f"No character matrix found; store characters in obsm[{key}]")
    if isinstance(tree, TreeData):
        if missing_state is None:
            missing_state = tree.uns.get("missing_state", None)
        characters = _fill_missing(characters, missing_state)

    return characters


def _fill_missing(characters: pd.DataFrame, missing_state=None) -> pd.DataFrame:
    """Replace empty (NaN/None) entries of a character matrix by the missing state.

    The fill value is *missing_state* (its first integer value for a numeric
    matrix if it is a sequence), or ``-1`` for numeric and ``"-"`` for other
    matrices. Numeric matrices of whole numbers become integer. Returns
    *characters* itself when it has no empty entries.
    """
    mask = characters.isna().to_numpy()
    if not mask.any():
        return characters
    values = characters.to_numpy(dtype=object)
    observed = values[~mask]
    numeric = all(
        isinstance(v, (int, float, np.number)) and not isinstance(v, (bool, np.bool_))
        for v in observed
    )
    if missing_state is None:
        candidates = []
    elif isinstance(missing_state, (list, tuple, set)):
        candidates = list(missing_state)
    else:
        candidates = [missing_state]
    if numeric:
        ints = [int(c) for c in candidates if str(c).lstrip("-").isdigit()]
        fill = ints[0] if ints else -1
    else:
        fill = str(candidates[0]) if candidates else "-"
    values[mask] = fill
    out = pd.DataFrame(values, index=characters.index, columns=characters.columns)
    if numeric:
        out = out.astype(float)
        if np.all(np.mod(out.to_numpy(), 1) == 0):
            out = out.astype(np.int64)
    return out


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
                    f"TreeData contains multiple trees ({', '.join(map(repr, keys))}); "
                    "pass tree_key to choose one."
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


def _check_leaf_alignment(tdata) -> None:
    """Raise a clear error if a solver is given a TreeData aligned to all nodes.

    Solvers treat every observation as a cell to place at a leaf, so the
    observations of internal nodes (``alignment="nodes"``) must be removed first.
    """
    if isinstance(tdata, TreeData) and tdata.alignment == "nodes":
        raise ValueError(
            "Tree reconstruction needs a TreeData with one observation per cell, but "
            "tdata.alignment is 'nodes'. Subset it to the leaves first, e.g. "
            "leaves = cas.tl.get_leaves(tdata); tdata.alignment = 'leaves'; "
            "tdata = tdata[leaves].copy()."
        )


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


_WARNED_DEFAULT_PARAMETERS: set[str] = set()


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
        # Warn once per parameter per process; a pipeline resolves these many times.
        if param_name not in _WARNED_DEFAULT_PARAMETERS:
            _WARNED_DEFAULT_PARAMETERS.add(param_name)
            warnings.warn(
                f"Parameter '{param_name}' not specified; using default value {value}. "
                f"Set tdata.uns[{param_name!r}] or pass {param_name}= to silence this.",
                UserWarning,
                stacklevel=3,
            )
    return value


# Priors convention (every module follows it):
#
# * Stored in ``tdata.uns["priors"]`` as ``{"<character name>": {"<state>": p}}``,
#   keyed by the column names of the character matrix, so they stay matched to
#   their characters after columns are subset or reordered. Keys and states are
#   strings because h5ad/h5td cannot store integer keys or lists of dicts. Write
#   them with :func:`_format_priors`.
# * Functions that take ``priors`` accept ``None`` (stored priors if present),
#   ``True`` (stored priors, required), ``False`` (none) or priors in any form
#   (dict keyed by character name or integer position, or a list of
#   per-character dicts), and read them with :func:`_resolve_priors`.
# * Resolved priors are keyed by integer character position, and their states
#   are the character matrix's own values, so ``priors[i][value]`` works for
#   string, integer, float and ambiguous (tuple) matrices alike.


def _priors_by_character(priors) -> dict:
    """Per-character priors from a dict (any key type) or a list of dicts."""
    if isinstance(priors, (list, tuple)):
        priors = dict(enumerate(priors))
    if not isinstance(priors, Mapping):
        raise ValueError(
            "priors must map each character index to a dict of state probabilities, "
            f"or be a list of such dicts; got {type(priors).__name__}."
        )
    out = {}
    for k, states in priors.items():
        if not isinstance(states, Mapping):
            raise ValueError(
                f"priors for character {k!r} must be a dict of state probabilities, "
                f"got {type(states).__name__}."
            )
        out[_int_key(k)] = states
    return out


def _format_priors(priors, columns: Sequence | None = None) -> dict[str, dict[str, float]]:
    """Priors in the stored form: ``{"<character name>": {"<state>": p}}``.

    Accepts the same forms as :func:`_resolve_priors`. Integer keys are
    positions in *columns* (the character matrix's column names) and are
    replaced by the column name; other keys are kept. String keys let the
    TreeData be written to h5ad/h5td.
    """
    columns = list(columns) if columns is not None else []
    out = {}
    for k, states in _priors_by_character(priors).items():
        if isinstance(k, int) and 0 <= k < len(columns):
            k = columns[k]
        out[str(k)] = {str(s): float(p) for s, p in states.items()}
    return out


def _match_prior_states(states: Mapping, values) -> dict:
    """Key one character's priors by the matrix's own state values.

    States are matched to *values* by their string form (integral floats also
    match their integer form). Unmatched states become integers when the
    character's values are numeric (or unknown) and stay strings otherwise.
    Ambiguous (tuple) values contribute their component states.
    """
    known: dict[str, Any] = {}
    for v in values:
        for s in v if isinstance(v, tuple) else (v,):
            if isinstance(s, np.generic):
                s = s.item()
            known.setdefault(str(s), s)
            if isinstance(s, float) and s.is_integer():
                known.setdefault(str(int(s)), s)
    numeric = all(isinstance(s, (int, float)) and not isinstance(s, bool) for s in known.values())
    out = {}
    for s, p in states.items():
        key = known.get(str(s))
        if key is None:
            key = _int_key(s) if numeric else str(s)
        out[key] = float(p)
    return out


def _resolve_priors(
    tree: CassiopeiaTree | TreeData,
    priors: dict | list | bool | None = None,
    characters: pd.DataFrame | None = None,
) -> dict[int, dict] | None:
    """Resolve a ``priors`` argument to a priors dict or ``None``.

    Centralizes the shared ``priors`` semantics so the behavior is identical
    everywhere (see the priors convention above).

    Args:
        tree: CassiopeiaTree or TreeData that may store priors.
        priors: ``None`` uses ``tree.uns["priors"]`` (``tree.priors`` for a
            CassiopeiaTree) if present, logging a message, and otherwise no
            priors. ``True`` requires stored priors and raises if there are none.
            ``False`` disables priors. Priors given directly may be a dict
            keyed by character (column) name or integer position, or a list of
            per-character dicts.
        characters: The character matrix the priors refer to. Its column names
            match name-keyed priors and its values become the state keys;
            without it, digit-string keys and states become integers.

    Returns:
        Priors keyed by integer character position, then by state, or ``None`` when
        disabled or unavailable.

    Raises:
        ValueError: If ``priors=True`` but no priors are stored on *tree*, or if
            the priors are malformed.
    """
    if priors is False:
        return None
    if priors is None or priors is True:
        stored = tree.priors if isinstance(tree, CassiopeiaTree) else tree.uns.get("priors", None)
        if not stored:
            if priors is True:
                raise ValueError(
                    "priors=True but no priors were found in tdata.uns['priors']. "
                    "Specify priors explicitly (priors=<dict>) or set priors=False "
                    "to run without priors."
                )
            return None
        if priors is None:
            logger.info("Using priors from tdata.uns['priors']; pass priors=False to ignore them.")
        priors = stored
    if isinstance(priors, Mapping) and characters is not None:
        # Keyed by character name when any key names a column: characters of
        # other matrices (or removed by subsetting) are skipped. Otherwise keys
        # are positions (lists, and integer or digit-string keys).
        position = {str(c): i for i, c in enumerate(characters.columns)}
        if any(str(k) in position for k in priors):
            priors = {position[str(k)]: v for k, v in priors.items() if str(k) in position}
    by_char = _priors_by_character(priors)
    columns = list(characters.columns) if characters is not None else []
    out = {}
    for k, states in by_char.items():
        if isinstance(k, int) and 0 <= k < len(columns):
            col = characters[columns[k]]
            values = list(pd.unique(col.astype(object).to_numpy()))
            values += list(getattr(col.dtype, "categories", []))
            out[k] = _match_prior_states(states, values)
        else:
            out[k] = _match_prior_states(states, [])
    return out


def _int_key(k):
    """Convert digit-string keys (from serialized priors) back to ints."""
    if isinstance(k, str) and k.lstrip("-").isdigit():
        return int(k)
    if isinstance(k, (np.integer,)):
        return int(k)
    return k


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


def _get_node_data(g: nx.DiGraph, key: str) -> pd.Series:
    """Get a Series mapping every node label to a specified node attribute."""
    return pd.Series({node: g.nodes[node].get(key) for node in g.nodes})


def _node_name_generator() -> Generator[str, None, None]:
    """Yield unique temporary internal node names for building reconstructed trees.

    Solvers rename the nodes with :func:`_name_internal_nodes` when done. Names
    combine a random prefix, drawn once per generator, with a counter: they
    are unique within a generator and, with high probability, across generators.
    (Hashing ``time.time()`` repeated names on platforms with a coarse clock, such
    as macOS, which turned reconstructed trees into graphs with merged nodes.)
    """
    prefix = secrets.token_hex(8)
    for i in itertools.count():
        yield f"internal_{prefix}_{i}"


def _name_internal_nodes(g: nx.DiGraph, samples) -> nx.DiGraph:
    """Name a reconstructed tree's nodes ``root``, ``node1``, ``node2``, ...

    Nodes in *samples* keep their names. The others are numbered in depth-first
    (preorder) order from the root, so the names are reproducible. If a sample
    already has one of the names, ``_`` is appended to the prefixes until none
    clash.
    """
    samples = set(samples)
    root = _get_root(g)
    internal = [n for n in nx.dfs_preorder_nodes(g, root) if n not in samples]
    root_name, prefix = "root", "node"
    while True:
        names = [root_name if n == root else f"{prefix}{i}" for i, n in enumerate(internal)]
        if samples.isdisjoint(names):
            break
        root_name, prefix = root_name + "_", prefix + "_"
    return nx.relabel_nodes(g, dict(zip(internal, names, strict=True)))


# Smallest "negative_log" weight: -log(1) = 0 would make the solvers ignore a state
# whose prior is 1 (a site with a single possible edit).
_MIN_LOG_WEIGHT = 1e-3


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
                "negative_log": Transforms each probability by the negative log,
                    floored at ``1e-3`` so that a state with prior 1 (e.g. the
                    only outcome of a base editor) still carries weight
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

    prior_function = lambda x: max(-np.log(x), _MIN_LOG_WEIGHT)

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
