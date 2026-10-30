from __future__ import annotations

from collections.abc import Hashable, Iterable
from typing import Literal

import networkx as nx
import numpy as np
import pandas as pd
from treedata import TreeData

from cassiopeia.mixins import is_ambiguous_state
from cassiopeia.mixins.errors import CassiopeiaError
from cassiopeia.utils import (
    _get_characters,
    _get_digraph,
    _get_parameter,
    _normalize_missing,
    _set_tree,
)

_METHODS = ("camin_sokal", "conservative", "sankoff")

# Upper bound (bytes) on the Sankoff score tables held at once; characters are
# processed in chunks that respect it.
_SANKOFF_MEMORY = 2**28


def _get_lca_characters(
    vecs: list[list[int] | list[tuple[int, ...]]],
    missing_states: set,
    unmodified_state: int,
    missing_value: Hashable | None = None,
) -> list[int]:
    """Builds the character vector of the LCA of a list of character vectors, obeying Camin-Sokal Parsimony.

    For each index in the reconstructed vector, imputes the non-missing
    character if only one of the constituent vectors has a missing value at that
    index, and imputes missing value if all have a missing value at that index.

    Importantly, this method will infer ancestral characters for an ambiguous
    state. If the intersection between two states (even ambiguous) is non-zero
    and not the missing state, and has length exactly 1, we assign the ancestral
    state this value. Else, if the intersection length is greater than 1, the
    ``unmodified_state`` value is assigned.

    Args:
        vecs: A list of character vectors to generate an LCA for
        missing_states: The set of values representing missing data
        unmodified_state: The value representing the unmodified (uncut) state
        missing_value: The value written where all vectors are missing. Defaults
            to an element of ``missing_states``.

    Returns:
            A list representing the character vector of the LCA

    """
    k = len(vecs[0])
    for i in vecs:
        assert len(i) == k
    if missing_value is None:
        missing_value = next(iter(missing_states))
    lca_vec = [unmodified_state] * len(vecs[0])
    for i in range(k):
        if all(vec[i] in missing_states for vec in vecs):
            lca_vec[i] = missing_value
        else:
            all_states = [vec[i] for vec in vecs if vec[i] not in missing_states]

            # this check is specifically if all_states consists of a single
            # ambiguous state.
            if len(list(set(all_states))) == 1:
                state = all_states[0]
                if is_ambiguous_state(state) and len(state) == 1:
                    lca_vec[i] = state[0]
                else:
                    lca_vec[i] = all_states[0]
            else:
                all_ambiguous = np.all([is_ambiguous_state(s) for s in all_states])
                chars = set.intersection(
                    *map(
                        set,
                        [state if is_ambiguous_state(state) else [state] for state in all_states],
                    )
                )
                if len(chars) == 1:
                    lca_vec[i] = list(chars)[0]
                if all_ambiguous:
                    # if we only have ambiguous states, we set the LCA state
                    # to be the intersection.
                    lca_vec[i] = tuple(chars)
    return lca_vec


def _seed_leaf_states(g: nx.DiGraph, character_matrix, key_added: str) -> None:
    """Initialize leaf ``key_added`` node attributes from a character matrix.

    Args:
        g: Directed tree graph.
        character_matrix: ``pd.DataFrame`` indexed by leaf name with one column
            per character.
        key_added: Node attribute under which the per-leaf character state list
            is stored.

    Raises:
        CassiopeiaError: If a leaf is missing from *character_matrix*.
    """
    for node in g.nodes:
        if g.out_degree(node) != 0:
            continue
        if node not in character_matrix.index:
            raise CassiopeiaError(
                f"Leaf {node!r} is not present in the character matrix; cannot "
                "seed leaf character states."
            )
        g.nodes[node][key_added] = list(character_matrix.loc[node])


def _reconstruct(
    g: nx.DiGraph,
    missing_states: set,
    unmodified_state: int,
    key: str,
    missing_value: Hashable | None = None,
) -> None:
    """Reconstruct ancestral states in place using Camin-Sokal parsimony.

    Internal-node states under the ``key`` node attribute are inferred bottom-up
    from the children's states. Leaves are assumed to already carry ``key``.

    Raises:
        AttributeError: If a leaf has no character states under ``key``.
    """
    for n in nx.dfs_postorder_nodes(g):
        if g.out_degree(n) == 0:
            if g.nodes[n].get(key, None) is None or g.nodes[n][key] == []:
                raise AttributeError(
                    "Character states have not been initialized at leaves."
                    " Seed leaf character states from a character matrix"
                    " before reconstructing ancestral characters."
                )
            continue
        character_states = [g.nodes[c][key][:] for c in g.successors(n)]
        g.nodes[n][key] = _get_lca_characters(
            character_states, missing_states, unmodified_state, missing_value
        )


def _state_value(state, observed: Iterable) -> Hashable:
    """Pick the value written for a resolved missing or unmodified state.

    ``state`` is a scalar or a sequence of equivalent values (e.g. the fallback
    ``(-1, "-1", "NA", "-")``). For a sequence, prefer a value that occurs in
    ``observed``, then one of the same kind (string or not) as the observed values.
    """
    if not isinstance(state, (list, tuple, set, frozenset)):
        return state
    candidates = sorted(state, key=str) if isinstance(state, (set, frozenset)) else list(state)
    observed = set(observed)
    for value in candidates:
        if value in observed:
            return value
    strings = any(isinstance(value, str) for value in observed)
    for value in candidates:
        if isinstance(value, str) == strings:
            return value
    return candidates[0]


def _encode_states(
    values: np.ndarray, missing_states: set, unmodified_states: set
) -> tuple[np.ndarray, np.ndarray]:
    """Integer-encode character states.

    Args:
        values: Array of character states (any shape).
        missing_states: Values representing missing data (``NaN`` is also missing).
        unmodified_states: Values representing the unmodified state.

    Returns:
        ``codes`` with the shape of ``values`` (-1 missing, 0 unmodified, ``1..``
        edits; codes are shared across characters) and ``edits``, the edited state
        of each positive code (``edits[code - 1]``).

    Raises:
        CassiopeiaError: If a state is ambiguous (a tuple).
    """
    values = np.asarray(values, dtype=object)
    flat, uniques = pd.factorize(values.ravel(), use_na_sentinel=True)
    uniques = np.asarray(uniques, dtype=object)
    if any(isinstance(u, tuple) for u in uniques):
        raise CassiopeiaError("Ambiguous (tuple) character states are not supported here.")
    missing = np.array([u in missing_states for u in uniques], dtype=bool)
    unmodified = np.array([u in unmodified_states for u in uniques], dtype=bool) & ~missing
    edited = ~(missing | unmodified)
    # The extra last slot maps factorize's NaN sentinel (-1) to missing.
    lookup = np.full(len(uniques) + 1, -1, dtype=np.int64)
    lookup[:-1][unmodified] = 0
    lookup[:-1][edited] = np.arange(1, edited.sum() + 1)
    return lookup[flat].reshape(values.shape), uniques[edited]


def _decode_states(
    codes: np.ndarray, edits: np.ndarray, missing_value: Hashable, unmodified_value: Hashable
) -> list:
    """Map integer codes from :func:`_encode_states` back to state values."""
    lookup = np.empty(len(edits) + 2, dtype=object)
    lookup[0] = missing_value
    lookup[1] = unmodified_value
    lookup[2:] = edits
    return lookup[np.asarray(codes) + 1].tolist()


def _lca_codes(child_codes: np.ndarray) -> np.ndarray:
    """Camin-Sokal state of a node from its children's codes (children x characters).

    A character keeps an edit when every child with data carries it, is missing
    when no child has data, and is unmodified otherwise.
    """
    observed = child_codes >= 0
    low = np.where(observed, child_codes, np.iinfo(np.int64).max).min(axis=0)
    high = np.where(observed, child_codes, -1).max(axis=0)
    return np.where(observed.any(axis=0), np.where(low == high, low, 0), -1)


def _conservative(children: dict, order: list, parent: dict, codes: dict) -> dict:
    """Conservative maximum parsimony on integer codes.

    Camin-Sokal states, except that an edit is set to missing where parsimony
    cannot place it: a node whose edit is carried by fewer than two of its
    children (the others missing data), and by no ancestor that is itself
    supported this way, may equally have acquired the edit on the edge below it.
    The root sits below an implicit unedited founder, which supports no edit.

    Args:
        children: Children of every node.
        order: Nodes in postorder.
        parent: Parent of every non-root node.
        codes: State codes of the leaves.

    Returns:
        State codes of the internal nodes.
    """
    lca = {}
    for node in order:
        if children[node]:
            lca[node] = _lca_codes(np.stack([lca[c] for c in children[node]]))
        else:
            lca[node] = codes[node]
    supported = {}
    states = {}
    for node in reversed(order):
        if not children[node]:
            continue
        state = lca[node]
        edited = state > 0
        shared = sum((lca[c] == state).astype(np.int64) for c in children[node]) >= 2
        inherited = supported[parent[node]] if node in parent else False
        supported[node] = edited & (shared | inherited)
        states[node] = np.where(edited & ~supported[node], -1, state)
    return states


def _sankoff(
    children: dict,
    order: list,
    codes: dict,
    reedit_cost: float,
    weighted: bool,
    costs: np.ndarray | None,
) -> dict:
    """Weighted Sankoff parsimony on integer codes.

    An implicit unedited founder sits above the root. Leaves with missing data
    are compatible with every state, and nodes with no observed leaf below them
    are missing.

    Args:
        children: Children of every node.
        order: Nodes in postorder (the root last).
        codes: State codes of the leaves.
        reedit_cost: Cost of changing an edited state; editing the unmodified
            state costs 1. Ignored when ``costs`` is given.
        weighted: Scale each edge's costs by log2(1 + number of leaves below it).
        costs: Cost matrix over codes, rows from and columns to (index 0 is the
            unmodified state), or ``None`` for the default costs.

    Returns:
        State codes of the internal nodes.
    """
    internal = [n for n in order if children[n]]
    if not internal:
        return {}
    n_below = {}
    observed_below = {}
    for node in order:
        if children[node]:
            n_below[node] = sum(n_below[c] for c in children[node])
            observed_below[node] = sum(observed_below[c] for c in children[node])
        else:
            n_below[node] = 1
            observed_below[node] = (codes[node] >= 0).astype(np.int64)
    weight = {n: np.log2(1 + n_below[n]) if weighted else 1.0 for n in order}

    # Each character gets its own alphabet: local code 0 is unmodified and
    # 1..n are the edits seen at its leaves.
    leaves = [n for n in order if not children[n]]
    leaf_codes = np.stack([codes[n] for n in leaves])
    alphabets = [np.unique(column[column > 0]) for column in leaf_codes.T]
    n_states = np.array([len(alphabet) + 1 for alphabet in alphabets])
    states = {n: np.empty(leaf_codes.shape[1], dtype=np.int64) for n in internal}
    columns = np.argsort(n_states, kind="stable")
    start = 0
    while start < len(columns):
        stop = start + 1
        while stop < len(columns) and (
            (len(internal) + n_states[columns[stop]])
            * n_states[columns[stop]]
            * (stop - start + 1)
            * 8
            <= _SANKOFF_MEMORY
        ):
            stop += 1
        cols = columns[start:stop]
        size = n_states[cols].max()
        # Global code of each local code, per character (padding maps to 0).
        to_global = np.zeros((size, len(cols)), dtype=np.int64)
        for j, col in enumerate(cols):
            to_global[1 : n_states[col], j] = alphabets[col]
        block = leaf_codes[:, cols]
        for j, col in enumerate(cols):
            edited = block[:, j] > 0
            block[edited, j] = np.searchsorted(alphabets[col], block[edited, j]) + 1
        local = dict(zip(leaves, block, strict=True))
        tensor = None
        if costs is not None:
            tensor = costs[to_global[:, None, :], to_global[None, :, :]]
            valid = np.arange(size)[:, None] < n_states[cols][None, :]
            tensor = np.where(valid[:, None, :] & valid[None, :, :], tensor, np.inf)
        chosen = _sankoff_columns(
            children, order, local, n_states[cols], weight, reedit_cost, tensor
        )
        for node in internal:
            states[node][cols] = to_global[chosen[node], np.arange(len(cols))]
        start = stop
    for node in internal:
        states[node][observed_below[node] == 0] = -1
    return states


def _sankoff_columns(
    children: dict,
    order: list,
    local: dict,
    n_states: np.ndarray,
    weight: dict,
    reedit_cost: float,
    tensor: np.ndarray | None,
) -> dict:
    """Sankoff on a block of characters with local codes; see :func:`_sankoff`."""
    size = n_states.max()
    n_cols = len(n_states)
    cols = np.arange(n_cols)
    valid = np.arange(size)[:, None] < n_states[None, :]
    # Cost of leaving each parent state (default costs).
    leave = np.where(np.arange(size) == 0, 1.0, reedit_cost)[:, None]

    def leaf_scores(codes: np.ndarray) -> np.ndarray:
        scores = np.where(valid & (codes < 0)[None, :], 0.0, np.inf)
        observed = codes >= 0
        scores[codes[observed], cols[observed]] = 0.0
        return scores

    def best_two(scores: np.ndarray):
        first = scores.argmin(axis=0)
        rest = scores.copy()
        rest[first, cols] = np.inf
        second = rest.argmin(axis=0)
        return first, scores[first, cols], second, rest[second, cols]

    def subtree_cost(scores: np.ndarray, w: float) -> np.ndarray:
        """Cost of a child's subtree for each state of its parent."""
        if tensor is not None:
            return (scores[None, :, :] + w * tensor).min(axis=1)
        first, low, _, second_low = best_two(scores)
        other = np.where(np.arange(size)[:, None] == first[None, :], second_low, low)
        return np.minimum(scores, other + w * leave)

    def child_state(scores: np.ndarray, w: float, parent_state: np.ndarray) -> np.ndarray:
        """Best state of a child given its parent's; ties keep the parent's state."""
        if tensor is not None:
            options = scores + w * tensor[parent_state, :, cols].T
            keep = options[parent_state, cols] <= options.min(axis=0)
            return np.where(keep, parent_state, options.argmin(axis=0))
        stay = scores[parent_state, cols]
        first, low, second, second_low = best_two(scores)
        same = first == parent_state
        target = np.where(same, second, first)
        change = np.where(same, second_low, low) + w * leave[parent_state, 0]
        return np.where(stay <= change, parent_state, target)

    scores = {}
    for node in order:
        if not children[node]:
            continue
        total = np.zeros((size, n_cols))
        for child in children[node]:
            child_scores = scores[child] if children[child] else leaf_scores(local[child])
            total += subtree_cost(child_scores, weight[child])
        total[~valid] = np.inf
        scores[node] = total

    root = order[-1]
    chosen = {root: child_state(scores[root], weight[root], np.zeros(n_cols, dtype=np.int64))}
    for node in reversed(order):
        for child in children[node]:
            if children[child]:
                chosen[child] = child_state(scores[child], weight[child], chosen[node])
    return chosen


def _cost_matrix(costs: pd.DataFrame, labels: list) -> np.ndarray:
    """Cost matrix over ``labels`` (unmodified state then edits) from a DataFrame.

    States are matched to the DataFrame's labels exactly, or else by string.
    """
    if not isinstance(costs, pd.DataFrame):
        raise TypeError(f"costs must be a pandas DataFrame, got {type(costs)}.")

    def match(index: pd.Index, axis: str) -> list:
        by_str = {str(label): label for label in index}
        found = []
        for value in labels:
            if value in index:
                found.append(value)
            elif str(value) in by_str:
                found.append(by_str[str(value)])
            else:
                raise CassiopeiaError(f"State {value!r} is missing from the {axis} of costs.")
        return found

    matrix = costs.loc[match(costs.index, "index"), match(costs.columns, "columns")]
    matrix = matrix.to_numpy(dtype=float)
    if np.isnan(matrix).any() or (matrix < 0).any():
        raise ValueError("costs must be non-negative numbers.")
    return matrix


def ancestral_characters(
    tdata: TreeData,
    method: Literal["camin_sokal", "conservative", "sankoff"] = "camin_sokal",
    characters_key: str = "characters",
    tree_key: str | None = None,
    missing_state: int | str | None = None,
    unmodified_state: int | str | None = None,
    reedit_cost: float = 2.5,
    weighted: bool = True,
    costs: pd.DataFrame | None = None,
    key_added: str | None = None,
    copy: bool = False,
) -> TreeData | None:
    """Reconstruct ancestral character states.

    Leaf character states are seeded from the tree's character matrix and the
    internal-node states are inferred with one of the following methods:

    * ``"camin_sokal"``: Camin-Sokal parsimony, which assumes edits are
      irreversible. A node carries an edit when every descendant leaf with data
      carries it; otherwise it is unmodified (or missing if no descendant has
      data). Supports ambiguous states.
    * ``"conservative"``: Camin-Sokal states, but an edit that parsimony cannot
      place is set to missing. This happens when an edit is carried by only one
      child of a node (the others are missing data), so it may have arisen on
      the edge above the node or below it. This is the conservative maximum
      parsimony reconstruction used by ConvexML
      (https://github.com/songlab-cal/ConvexML) for branch length estimation.
    * ``"sankoff"``: weighted Sankoff parsimony, which tolerates noise in the
      character matrix. Editing an unmodified state costs 1 and changing an
      edited state (to another edit or back to unmodified) costs
      ``reedit_cost``, so a miscalled leaf costs one change at that leaf rather
      than removing a shared edit from all of its ancestors. With
      ``weighted=True`` each edge's costs are scaled by
      log2(1 + number of leaves below the edge), so changes are cheapest on the
      edges to leaves, where sequencing errors act. The root's state is chosen
      as if an unedited founder sat above it. Nodes with no descendant data at
      a character are missing.

    By default, states are written to the ``characters_key`` node attribute —
    the same name as the obsm character matrix — so leaves and internal nodes
    share a single, consistent attribute. Pass ``key_added`` to store the
    inferred states (and seeded leaf states) under a different node attribute.

    Args:
        tdata: The TreeData object to operate on.
        method: Reconstruction method: ``"camin_sokal"``, ``"conservative"`` or
            ``"sankoff"``. ``"conservative"`` and ``"sankoff"`` do not support
            ambiguous states.
        characters_key: The ``obsm`` key for the character matrix used to seed
            leaf states.
        tree_key: The ``obst`` key of the tree to use.
        missing_state: Missing-data value. Resolved from the tdata when ``None``.
        unmodified_state: Unmodified (uncut) state value. Resolved from the tdata
            when ``None``.
        reedit_cost: Cost of changing an edited state, relative to a cost of 1
            for editing the unmodified state. Only used by ``method="sankoff"``.
        weighted: Scale each edge's costs by log2(1 + number of leaves below the
            edge). Only used by ``method="sankoff"``.
        costs: Custom cost matrix for ``method="sankoff"`` as a DataFrame with
            the state changed from in the rows and the state changed to in the
            columns. It must cover the unmodified state and every edited state
            in the character matrix. Overrides ``reedit_cost``.
        key_added: Node attribute under which to store states. Defaults to
            ``characters_key`` when ``None``.
        copy: If ``True``, operate on and return a copy of *tdata*; otherwise
            modify in place and return ``None``.

    Returns:
        A modified copy of *tdata* if ``copy=True``, else ``None``.

    Raises:
        ValueError: If ``method`` is unknown or ``reedit_cost`` is not positive.
        CassiopeiaError: If ``method`` is ``"conservative"`` or ``"sankoff"`` and
            the character matrix has ambiguous states, or ``costs`` misses a state.
    """
    if method not in _METHODS:
        raise ValueError(f"Unknown method {method!r}; choose one of {', '.join(_METHODS)}.")
    if method == "sankoff" and costs is None and not reedit_cost > 0:
        raise ValueError(f"reedit_cost must be positive, got {reedit_cost}.")

    tdata = tdata.copy() if copy else tdata
    # TreeData stores frozen graphs; operate on a copy and write back.
    g, tree_key = _get_digraph(tdata, tree_key, copy=True)

    missing = _get_parameter(tdata, "missing_state", value=missing_state)
    unmodified = _get_parameter(tdata, "unmodified_state", value=unmodified_state)
    missing_states = _normalize_missing(missing)
    unmodified_states = _normalize_missing(unmodified)

    write_key = key_added if key_added is not None else characters_key
    character_matrix = _get_characters(tdata, characters_key, missing_state)
    observed = set(character_matrix.to_numpy(dtype=object).ravel().tolist())
    missing_value = _state_value(missing, observed)
    unmodified_value = _state_value(unmodified, observed)
    _seed_leaf_states(g, character_matrix, write_key)

    if method == "camin_sokal":
        _reconstruct(g, missing_states, unmodified_value, write_key, missing_value)
    else:
        root = next(n for n in g.nodes if g.in_degree(n) == 0)
        order = list(nx.dfs_postorder_nodes(g, root))
        children = {n: list(g.successors(n)) for n in order}
        leaves = [n for n in order if not children[n]]
        leaf_values = [g.nodes[n][write_key] for n in leaves]
        leaf_codes, edits = _encode_states(
            pd.DataFrame(leaf_values, dtype=object).to_numpy(), missing_states, unmodified_states
        )
        codes = dict(zip(leaves, leaf_codes, strict=True))
        if method == "conservative":
            parent = {c: p for p, c in g.edges}
            states = _conservative(children, order, parent, codes)
        else:
            matrix = None
            if costs is not None:
                matrix = _cost_matrix(costs, [unmodified_value, *edits])
            states = _sankoff(children, order, codes, reedit_cost, weighted, matrix)
        for node, state in states.items():
            g.nodes[node][write_key] = _decode_states(state, edits, missing_value, unmodified_value)

    _set_tree(tdata, g, tree_key)

    return tdata if copy else None
