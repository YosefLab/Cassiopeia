"""Utilities to assess topological properties of a phylogeny, such as balance and expansion."""

from collections.abc import Callable

import networkx as nx
import numpy as np
from treedata import TreeData

from cassiopeia.mixins import CassiopeiaError
from cassiopeia.utils import (
    _check_tree_has_key,
    _collapse_unifurcations,
    _combine_edge_data,
    _get_digraph,
    _get_parameter,
    _get_root,
    _normalize_missing,
)


def get_root(tree: TreeData | nx.DiGraph, tree_key: str | None = None) -> str:
    """Return the unique root of a tree.

    Args:
        tree: The tree object.
        tree_key: The ``obst`` key to use when ``tree`` is a
            :class:`treedata.TreeData`. Only required if multiple trees are present.

    Returns:
        The node label of the root.

    Raises:
        ValueError: If the tree does not contain exactly one root.
    """
    g, _ = _get_digraph(tree, tree_key=tree_key)
    return _get_root(g)


def get_leaves(tree: TreeData | nx.DiGraph, tree_key: str | None = None) -> list[str]:
    """Return the leaf labels of a tree, ordered by a depth-first traversal.

    Leaves are returned in the order they are first encountered in a depth-first
    traversal from the root, so the ordering reflects the tree topology rather
    than the node labels.

    Args:
        tree: The tree object.
        tree_key: The ``obst`` key to use when ``tree`` is a
            :class:`treedata.TreeData`. Only required if multiple trees are present.

    Returns:
        Leaf labels in depth-first traversal order.
    """
    g, _ = _get_digraph(tree, tree_key=tree_key)
    root = _get_root(g)
    return [node for node in nx.dfs_preorder_nodes(g, source=root) if g.out_degree(node) == 0]


def mean_depth(
    tree: TreeData | nx.DiGraph,
    depth_key: str | None = None,
    tree_key: str | None = None,
) -> float:
    """Compute the mean depth of a tree's leaves.

    Calculates the average depth across all leaf nodes in the tree. When
    ``depth_key`` is provided, depth is read from that node attribute; this can
    represent either discrete generations (e.g., number of divisions) or
    continuous time (e.g., evolutionary time). When ``depth_key`` is ``None``
    (default), the topological depth of each leaf (the number of edges from the
    root) is computed from the tree structure.

    Args:
        tree: Tree object.
        depth_key: Node attribute key containing depth values (e.g., ``"depth"``,
            ``"time"``). If ``None`` (default), topological depth is computed
            from the tree structure.
        tree_key: Tree key to use if ``tree`` is a TreeData object with multiple
            trees.

    Returns:
        Mean depth of the tree's leaves.
    """
    t, _ = _get_digraph(tree, tree_key=tree_key)
    leaves = get_leaves(tree, tree_key=tree_key)
    if depth_key is None:
        depths_from_root = nx.single_source_shortest_path_length(t, _get_root(t))
        depths = [depths_from_root[leaf] for leaf in leaves]
    else:
        _check_tree_has_key(t, depth_key)
        depths = [t.nodes[leaf][depth_key] for leaf in leaves]
    return float(np.mean(depths))


def rescale_node_times(
    tree: TreeData | nx.DiGraph,
    time_key: str = "time",
    min: float = 0,
    max: float = 1,
    key_added: str | None = None,
    tree_key: str | None = None,
    copy: bool = False,
) -> TreeData | nx.DiGraph | None:
    """Linearly rescale node times to a target range.

    Node times stored under ``time_key`` are linearly rescaled so the smallest
    node time maps to ``min`` and the largest maps to ``max``, preserving the
    relative spacing between nodes. When all node times are equal, every node is
    assigned ``min``.

    Args:
        tree: Tree object. Either a :class:`~treedata.TreeData` or an
            :class:`networkx.DiGraph`.
        time_key: Node attribute key containing the times to rescale.
        min: Target value for the smallest node time.
        max: Target value for the largest node time.
        key_added: Node attribute under which to store the rescaled times. If
            ``None`` (default), the rescaled times overwrite ``time_key``.
        tree_key: The ``obst`` key of the tree to use when ``tree`` is a TreeData
            object.
        copy: If ``True``, operate on and return a copy of *tree*; otherwise
            modify in place and return ``None``.

    Returns:
        A modified copy of *tree* (TreeData or DiGraph, matching the input) if
        ``copy=True``, else ``None``.
    """
    if not isinstance(tree, (TreeData, nx.DiGraph)):
        raise TypeError(
            f"rescale_node_times() operates on TreeData or nx.DiGraph, got {type(tree)}."
        )
    if max <= min:
        raise ValueError(f"max ({max}) must be greater than min ({min}).")

    if copy:
        tree = tree.copy()
    g, _ = _get_digraph(tree, tree_key)
    _check_tree_has_key(g, time_key)

    times = np.array([g.nodes[node][time_key] for node in g.nodes], dtype=float)
    t_min, t_max = times.min(), times.max()
    span = t_max - t_min
    output_key = key_added if key_added is not None else time_key

    for node in g.nodes:
        if span == 0:
            scaled = float(min)
        else:
            scaled = min + (g.nodes[node][time_key] - t_min) / span * (max - min)
        g.nodes[node][output_key] = scaled

    if copy:
        return tree if isinstance(tree, TreeData) else g
    return None


def _mutations_along_edge(
    parent_states: list,
    child_states: list,
    missing_states: set,
    treat_missing_as_mutation: bool = False,
) -> list[tuple[int, int]]:
    """Get the mutations along an edge from parent to child character states.

    Returns a list of ``(character, state)`` tuples of mutations that occur
    along an edge. Characters are 0-indexed. By default, transitions to a
    missing state (and from a missing state) are not counted as mutations.

    Note that parent states can be ambiguous if all child states have the
    same ambiguous state; in that case no mutation is detected.

    Args:
        parent_states: Character state list at the parent node.
        child_states: Character state list at the child node.
        missing_states: Set of values representing missing data.
        treat_missing_as_mutation: Whether to count transitions to a missing
            state as mutations.

    Returns:
        A list of ``(character, state)`` tuples indicating which character
        mutated and to which state.
    """
    mutations = []
    for i in range(len(parent_states)):
        parent_state = (
            list(parent_states[i]) if isinstance(parent_states[i], tuple) else [parent_states[i]]
        )
        child_state = (
            list(child_states[i]) if isinstance(child_states[i], tuple) else [child_states[i]]
        )
        if len(np.intersect1d(parent_state, child_state)) < 1:
            if treat_missing_as_mutation:
                mutations.append((i, child_states[i]))
            elif parent_states[i] not in missing_states and child_states[i] not in missing_states:
                mutations.append((i, child_states[i]))
    return mutations


def _mutationless_criteria(parent_states: list, child_states: list, missing_states: set) -> bool:
    """Return ``True`` when no real mutation occurs from parent to child.

    The default edge-collapse criterion: an edge carries no mutation when the
    parent and child have identical inferred character states up to missing
    data. Transitions to/from a missing state are not counted as mutations, so
    such edges are still collapsed.
    """
    return len(_mutations_along_edge(parent_states, child_states, missing_states)) == 0


# Registry of edge-collapse criteria. Maps a criterion name to a predicate
# ``(parent_states, child_states, missing_states) -> bool`` that is ``True``
# when the edge between them should be collapsed. The structural
# ``"unifurcation"`` criterion is handled separately (it needs no character
# states). Extend this to add future state-based criteria.
_COLLAPSE_CRITERIA: dict[str, Callable[[list, list, set], bool]] = {
    "mutationless": _mutationless_criteria,
}


def collapse_edges(
    tdata: TreeData,
    tree_key: str | None = None,
    characters_key: str = "characters",
    criteria: str = "mutationless",
    collapse_root: bool = True,
    copy: bool = False,
) -> TreeData | None:
    """Collapse edges of a tree according to a collapse criterion.

    For each internal node, any non-leaf child satisfying the collapse
    *criteria* is spliced out and its children are reattached to the node.
    Leaves are never removed. Numeric edge attributes (e.g. branch lengths) are
    summed across the removed edges so additive quantities are preserved;
    non-numeric attributes take the child edge's value.

    Two criteria are supported:

    * ``'mutationless'`` (default): collapse an edge when the parent and child
      have identical inferred character states. Ancestral character states must
      already be present on every node under the ``characters_key`` node
      attribute; call :func:`cassiopeia.tl.ancestral_characters` first if they
      are not.
    * ``'unifurcation'``: collapse every internal node with exactly one child
      (a structural criterion needing no character states).

    Only :class:`~treedata.TreeData` is supported.

    Args:
        tdata: TreeData object to operate on.
        tree_key: The ``obst`` key of the tree to use.
        characters_key: Node attribute holding character states (the same name
            as the obsm character matrix and the output of
            :func:`cassiopeia.tl.ancestral_characters`). Only used by the
            ``'mutationless'`` criterion.
        criteria: Name of the edge-collapse criterion to apply. Supports
            ``'mutationless'`` and ``'unifurcation'``.
        collapse_root: For ``criteria='unifurcation'``, whether to also collapse
            the root's single child into the root. Ignored otherwise.
        copy: If ``True``, operate on and return a copy of *tdata*; otherwise
            modify in place and return ``None``.

    Returns:
        A modified copy of *tdata* if ``copy=True``, else ``None``.

    Raises:
        TypeError: If *tdata* is not a TreeData object.
        ValueError: If *criteria* is not a recognized criterion.
        CassiopeiaError: If a node is missing character states (mutationless).
    """
    if not isinstance(tdata, TreeData):
        raise TypeError(
            "collapse_edges() operates on TreeData. For a CassiopeiaTree, convert "
            "with CassiopeiaTree.to_treedata()."
        )
    if criteria != "unifurcation" and criteria not in _COLLAPSE_CRITERIA:
        raise ValueError(
            f"Unknown collapse criteria {criteria!r}. "
            f"Available: {sorted([*_COLLAPSE_CRITERIA, 'unifurcation'])}"
        )

    tdata = tdata.copy() if copy else tdata
    # TreeData stores frozen graphs; operate on a copy and write back.
    g, tree_key = _get_digraph(tdata, tree_key, copy=True)

    if criteria == "unifurcation":
        g = _collapse_unifurcations(g, collapse_root=collapse_root)
        tdata.obst[tree_key] = g
        return tdata if copy else None

    predicate = _COLLAPSE_CRITERIA[criteria]
    missing_states = _normalize_missing(_get_parameter(tdata, "missing_state"))

    for node in g.nodes:
        if characters_key not in g.nodes[node]:
            raise CassiopeiaError(
                f"Node {node!r} has no character states under {characters_key!r}. "
                "Call cassiopeia.tl.ancestral_characters first."
            )

    for node in list(nx.dfs_postorder_nodes(g)):
        if g.out_degree(node) == 0:
            continue
        for child in list(g.successors(node)):
            if g.out_degree(child) == 0:
                continue
            if predicate(
                g.nodes[node][characters_key], g.nodes[child][characters_key], missing_states
            ):
                parent_edge = dict(g.get_edge_data(node, child, default={}))
                for grandchild in list(g.successors(child)):
                    child_edge = dict(g.get_edge_data(child, grandchild, default={}))
                    g.add_edge(node, grandchild, **_combine_edge_data(parent_edge, child_edge))
                g.remove_node(child)

    tdata.obst[tree_key] = g

    return tdata if copy else None


def count_edge_mutations(
    tdata: TreeData,
    tree_key: str | None = None,
    characters_key: str = "characters",
    treat_missing_as_mutation: bool = False,
    missing_state: str | int | None = None,
    key_added: str = "n_mutations",
    copy: bool = False,
) -> TreeData | None:
    """Count the mutations along each edge of a tree.

    For every edge, counts the number of character/state mutations between the
    parent and child node states and stores the count under the ``key_added``
    edge attribute. A mutation occurs at a character when the child has a state
    not present in the parent. By default, transitions to/from a missing state
    are not counted as mutations.

    Ancestral character states must already be present on every node under the
    ``characters_key`` node attribute; call
    :func:`cassiopeia.tl.ancestral_characters` first if they are not.

    Only :class:`~treedata.TreeData` is supported.

    Args:
        tdata: TreeData object to operate on.
        tree_key: The ``obst`` key of the tree to use.
        characters_key: Node attribute holding character states (the same name
            as the obsm character matrix and the output of
            :func:`cassiopeia.tl.ancestral_characters`).
        treat_missing_as_mutation: Whether to count transitions to a missing
            state as mutations.
        missing_state: Missing-data value. Resolved from the tree when ``None``.
        key_added: Edge attribute under which to store the per-edge mutation
            count.
        copy: If ``True``, operate on and return a copy of *tdata*; otherwise
            modify in place and return ``None``.

    Returns:
        A modified copy of *tdata* if ``copy=True``, else ``None``.

    Raises:
        TypeError: If *tdata* is not a TreeData object.
        CassiopeiaError: If a node is missing character states.
    """
    if not isinstance(tdata, TreeData):
        raise TypeError(
            "count_edge_mutations() operates on TreeData. For a CassiopeiaTree, "
            "convert with CassiopeiaTree.to_treedata()."
        )

    tdata = tdata.copy() if copy else tdata
    # TreeData stores frozen graphs; operate on a copy and write back.
    g, tree_key = _get_digraph(tdata, tree_key, copy=True)

    missing_states = _normalize_missing(_get_parameter(tdata, "missing_state", value=missing_state))

    for node in g.nodes:
        if characters_key not in g.nodes[node]:
            raise CassiopeiaError(
                f"Node {node!r} has no character states under {characters_key!r}. "
                "Call cassiopeia.tl.ancestral_characters first."
            )

    for parent, child in g.edges:
        mutations = _mutations_along_edge(
            g.nodes[parent][characters_key],
            g.nodes[child][characters_key],
            missing_states,
            treat_missing_as_mutation,
        )
        g.edges[parent, child][key_added] = len(mutations)

    tdata.obst[tree_key] = g

    return tdata if copy else None
