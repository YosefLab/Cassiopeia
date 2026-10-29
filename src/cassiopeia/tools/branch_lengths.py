"""Branch length estimation.

``method="convexml"`` is adapted from ConvexML
(https://github.com/songlab-cal/ConvexML), Copyright (c) 2025, Song Lab @ Cal,
released under the BSD 3-Clause License.
"""

from __future__ import annotations

import heapq
from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd
from treedata import TreeData

from cassiopeia.mixins.errors import CassiopeiaError
from cassiopeia.tools.ancestral_characters import _encode_states, _lca_codes
from cassiopeia.utils import (
    _get_characters,
    _get_digraph,
    _get_parameter,
    _get_root,
    _normalize_missing,
)

_METHODS = ("convexml",)
_SOLVERS = ("CLARABEL", "SCS", "ECOS", "MOSEK")
# Keeps log(1 - exp(-x)) finite on zero-length edges, as in ConvexML.
_EPS = 1e-5


def _import_cvxpy():
    try:
        import cvxpy
    except ImportError as err:
        raise ImportError(
            "estimate_branch_lengths(method='convexml') requires cvxpy. Install it with "
            "`pip install cvxpy` or `pip install 'cassiopeia-lineage[convexml]'`."
        ) from err
    return cvxpy


def _preorder(children: dict, root) -> list:
    order, stack = [], [root]
    while stack:
        node = stack.pop()
        order.append(node)
        stack.extend(reversed(children[node]))
    return order


def _implicit_state(parent: np.ndarray, first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """State codes of a node added between ``parent`` and two of its children.

    ``first`` and ``second`` are the children's states, with missing entries
    replaced by the Camin-Sokal state of their leaves (an edit removed by a
    conservative reconstruction still counts). The node inherits the parent's
    edits. Elsewhere it carries an edit shared by both children, is missing
    where only one child has an edit and the other has no data, and is
    unmodified otherwise.
    """
    state = _lca_codes(np.stack([first, second]))
    shared = (first == state) & (second == state)
    state = np.where((state > 0) & ~shared, -1, state)
    return np.where(parent > 0, parent, state)


def _resolve_multifurcations(children: dict, states: dict, root) -> dict:
    """Binarize a tree for estimation, as ConvexML does.

    A root with several children gets one implicit child that takes over the
    root's states, so the root stands for the unedited founder. Each node with
    more than two children is resolved by repeatedly joining its two smallest
    subtrees (by number of nodes, ties broken by name). ``states`` gains the
    states of the implicit nodes (see :func:`_implicit_state`).

    Returns:
        The children of every node of the binary tree.
    """
    children = {node: list(kids) for node, kids in children.items()}
    names = {str(node) for node in children}

    def name(base: str) -> str:
        while base in names:
            base += "_"
        names.add(base)
        return base

    size, leaf_states = {}, {}
    for node in reversed(_preorder(children, root)):
        size[node] = 1 + sum(size[c] for c in children[node])
        if children[node]:
            leaf_states[node] = _lca_codes(np.stack([leaf_states[c] for c in children[node]]))
        else:
            leaf_states[node] = states[node]

    def effective(node) -> np.ndarray:
        return np.where(states[node] >= 0, states[node], leaf_states[node])

    if len(children[root]) >= 2:
        child = name(f"{root}-child")
        children[child], children[root] = children[root], [child]
        states[child], leaf_states[child] = states[root], leaf_states[root]
    for node in list(children):
        if len(children[node]) < 3:
            continue
        heap = [(size[c], str(c), c) for c in children[node]]
        heapq.heapify(heap)
        i = 0
        while len(heap) > 2:
            first_size, _, first = heapq.heappop(heap)
            second_size, _, second = heapq.heappop(heap)
            joined = name(f"{node}-coalesce-{i}")
            children[joined] = [first, second]
            states[joined] = _implicit_state(states[node], effective(first), effective(second))
            leaf_states[joined] = _lca_codes(np.stack([leaf_states[first], leaf_states[second]]))
            size[joined] = first_size + second_size + 1
            heapq.heappush(heap, (size[joined], joined, joined))
            i += 1
        children[node] = [heapq.heappop(heap)[2], heapq.heappop(heap)[2]]
    return children


def _leaf_depths(
    tdata: TreeData, leaves: list, leaf_time_key: str | None
) -> tuple[np.ndarray, float]:
    """Leaf depths relative to the deepest leaf, and the deepest leaf's time."""
    if leaf_time_key is None:
        return np.ones(len(leaves)), 1.0
    if leaf_time_key not in tdata.obs.columns:
        raise KeyError(f"{leaf_time_key!r} is not a column of tdata.obs.")
    times = pd.to_numeric(tdata.obs[leaf_time_key].reindex(leaves), errors="coerce")
    if times.isna().any():
        missing = times.index[times.isna()][0]
        raise ValueError(f"tdata.obs[{leaf_time_key!r}] has no numeric time for leaf {missing!r}.")
    times = times.to_numpy(dtype=float)
    if (times <= 0).any():
        raise ValueError(f"Leaf times in tdata.obs[{leaf_time_key!r}] must be positive.")
    return times / times.max(), float(times.max())


def _relative_rates(tdata: TreeData, characters_key: str, rates, n_characters: int) -> np.ndarray:
    """Per-character relative mutation rates, scaled to mean 1."""
    if rates is None:
        return np.ones(n_characters)
    if isinstance(rates, pd.Series) and characters_key in tdata.obsm:
        rates = rates.reindex(_get_characters(tdata, characters_key).columns)
    rates = np.asarray(rates, dtype=float)
    if rates.shape != (n_characters,):
        raise ValueError(
            f"relative_rates must have one rate per character ({n_characters}), "
            f"got shape {rates.shape}."
        )
    if not (np.isfinite(rates) & (rates > 0)).all():
        raise ValueError("relative_rates must be positive.")
    return rates / rates.mean()


def _long_edge_mutations(states: np.ndarray, parents: np.ndarray, children: np.ndarray):
    """Edits seen below one or more nodes with missing data.

    An edit at a node whose parent is missing data is placed on the path from
    the nearest ancestor with data when that ancestor's state differs.

    Args:
        states: State codes of the nodes in preorder (the root has data).
        parents: Parent index of each edge.
        children: Child index of each edge.

    Returns:
        Ancestor index, node index and character of each such edit.
    """
    nearest = np.empty_like(states)
    nearest[0] = 0
    parent_of = np.empty(len(states), dtype=np.int64)
    parent_of[children] = parents
    for node in range(1, len(states)):
        nearest[node] = np.where(states[node] >= 0, node, nearest[parent_of[node]])
    child_states = states[children]
    above = nearest[parents]
    candidate = (child_states > 0) & (states[parents] < 0)
    edge, character = np.nonzero(candidate)
    ancestor = above[edge, character]
    differs = states[ancestor, character] != child_states[edge, character]
    return ancestor[differs], children[edge[differs]], character[differs]


def _convexml(
    cp,
    children: dict,
    states: dict,
    root,
    leaves: list,
    depth: np.ndarray,
    rates: np.ndarray,
    min_branch_length: float,
    min_leaf_branch_length: float,
    pseudo_mutations: float,
    pseudo_non_mutations: float,
    solvers: list[str],
) -> tuple[dict, np.ndarray, float, float, str]:
    """Fit node times by maximum likelihood (ConvexML).

    Returns:
        Node times (the root at 0, the deepest leaf at 1), the mutation rate of
        each character on that time scale, the log-likelihood, the penalized
        log-likelihood and the solver used.
    """
    order = _preorder(children, root)
    index = {node: i for i, node in enumerate(order)}
    parents = np.array([index[p] for p in order for _ in children[p]], dtype=np.int64)
    kids = np.array([index[c] for p in order for c in children[p]], dtype=np.int64)
    codes = np.stack([states[node] for node in order])
    n_characters = codes.shape[1]

    leaf_index = np.array([index[leaf] for leaf in leaves])
    leaf_codes = codes[leaf_index]
    if pseudo_mutations == 0 and not (leaf_codes > 0).any():
        raise ValueError(
            "The character matrix has no edits; use pseudo_mutations > 0 to estimate branch lengths."
        )
    if pseudo_non_mutations == 0 and not (leaf_codes == 0).any():
        raise ValueError(
            "The character matrix has no unmodified states; use pseudo_non_mutations > 0 to "
            "estimate branch lengths."
        )

    # Every leaf must be reachable under the minimum branch lengths.
    is_leaf = np.zeros(len(order), dtype=bool)
    is_leaf[leaf_index] = True
    minimum = np.where(is_leaf[kids], min_leaf_branch_length, min_branch_length)
    required = np.zeros(len(order))
    for edge in range(len(kids)):
        required[kids[edge]] = required[parents[edge]] + minimum[edge]
    if (required[leaf_index] >= depth).any():
        raise ValueError(
            "min_branch_length is too large for this tree: the minimum branch lengths "
            "along a root-to-leaf path exceed the leaf's depth. Please reduce it."
        )

    parent_codes, child_codes = codes[parents], codes[kids]
    unmutated = (parent_codes == 0) & (child_codes == 0)
    mutated = (parent_codes >= 0) & (child_codes >= 0) & (parent_codes != child_codes)
    long_from, long_to, long_character = _long_edge_mutations(codes, parents, kids)

    t = cp.Variable(len(order))
    deepest = leaf_index[np.argmax(depth)]
    scale = t[deepest]
    lengths = t[kids] - t[parents]
    constraints = [t[0] == 0, lengths >= minimum * scale]
    others = np.flatnonzero(leaf_index != deepest)
    if len(others):
        constraints.append(t[leaf_index[others]] == depth[others] * scale)

    group_rates, group = np.unique(rates, return_inverse=True)
    objective = 0
    for g, rate in enumerate(group_rates):
        sites = group == g
        share = sites.sum() / n_characters
        n_unmutated = unmutated[:, sites].sum(axis=1) + pseudo_non_mutations * share
        n_mutated = mutated[:, sites].sum(axis=1) + pseudo_mutations * share
        objective += cp.sum(cp.multiply(-rate * n_unmutated, lengths))
        edges = np.flatnonzero(n_mutated > 0)
        if len(edges):
            objective += cp.sum(
                cp.multiply(n_mutated[edges], cp.log(1 - cp.exp(-rate * lengths[edges] - _EPS)))
            )
        in_group = group[long_character] == g
        if in_group.any():
            pairs, counts = np.unique(
                np.stack([long_from[in_group], long_to[in_group]], axis=1),
                axis=0,
                return_counts=True,
            )
            spans = t[pairs[:, 1]] - t[pairs[:, 0]]
            objective += cp.sum(cp.multiply(counts, cp.log(1 - cp.exp(-rate * spans - _EPS))))
    objective = objective / n_characters

    problem = cp.Problem(cp.Maximize(objective), constraints)
    used = None
    for solver in solvers:
        try:
            problem.solve(solver=solver)
        except cp.SolverError:
            continue
        if problem.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) and t.value is not None:
            used = solver
            break
    if used is None:
        raise CassiopeiaError(
            f"The convex solver(s) {', '.join(solvers)} failed to estimate branch lengths."
        )

    scale = float(t.value[deepest])
    if not 1e-8 < scale < 15:
        raise CassiopeiaError(
            f"The solver returned a degenerate solution (mutation rate {scale:.3g} per tree depth)."
        )
    times = t.value / scale
    times[0] = 0.0
    # Smooth out solver epsilons that put a child before its parent.
    for edge in range(len(kids)):
        times[kids[edge]] = max(times[kids[edge]], times[parents[edge]])
    penalized = float(problem.value) * n_characters
    if np.isnan(penalized):
        penalized = -np.inf

    site_rates = rates * scale
    edge_rates = (times[kids] - times[parents])[:, None] * site_rates[None, :]
    with np.errstate(divide="ignore"):
        log_mutated = np.log(1 - np.exp(-edge_rates - _EPS))
        long_rates = (times[long_to] - times[long_from]) * site_rates[long_character]
        log_likelihood = (
            -(edge_rates * unmutated).sum()
            + np.where(mutated, log_mutated, 0.0).sum()
            + np.log(1 - np.exp(-long_rates - _EPS)).sum()
        )
    return dict(zip(order, times, strict=True)), site_rates, float(log_likelihood), penalized, used


def estimate_branch_lengths(
    tdata: TreeData,
    method: Literal["convexml"] = "convexml",
    tree_key: str | None = None,
    characters_key: str = "characters",
    missing_state: int | str | None = None,
    unmodified_state: int | str | None = None,
    leaf_time_key: str | None = None,
    relative_rates: Sequence[float] | pd.Series | None = None,
    min_branch_length: float = 0.01,
    min_leaf_branch_length: float | None = None,
    pseudo_mutations: float = 0.1,
    pseudo_non_mutations: float = 0.1,
    resolve_multifurcations: bool = True,
    solver: str | Sequence[str] = ("CLARABEL", "SCS"),
    time_key: str = "time",
    length_key: str = "length",
    key_added: str = "branch_lengths",
    copy: bool = False,
) -> TreeData | None:
    """Estimate branch lengths from the character states on a tree.

    Fits node times by maximum likelihood under a model of irreversible editing
    in which every site is edited independently at a constant rate (ConvexML,
    https://github.com/songlab-cal/ConvexML). Given the topology and the
    character states of every node, the likelihood is concave in the node
    times, so the estimate is the exact optimum of a convex program. All leaves
    are assumed to be sampled at the same time unless their sampling times are
    given with ``leaf_time_key``.

    Ancestral character states must be stored on every node under
    ``characters_key``: run :func:`cassiopeia.tl.ancestral_characters` first.
    Its ``method="conservative"`` gives ConvexML's estimates (exactly so on
    trees without multifurcations). If the character matrix is noisy (e.g.
    sequencing errors or miscalled edits), reconstruct with
    ``method="sankoff"`` instead: under the other methods a single miscalled
    leaf removes a shared edit from all of its ancestors, so the edit appears
    as many independent edits on the edges below them, which distorts the
    branch lengths.

    The root is taken to be the unedited founder cell at time 0, so states
    stored at the root are ignored. Missing data are treated as missing at
    random; an edit seen below nodes with missing data is placed on the path
    from the nearest ancestor with data. Pseudocounts of mutations and
    non-mutations on every edge regularize the estimate.

    Results are written in place:

    * ``tdata.obst[tree_key].nodes[node][time_key]``: node times. The root is at
      0 and the deepest leaf at 1, or at its time in ``obs[leaf_time_key]``.
    * ``tdata.obst[tree_key].edges[edge][length_key]``: branch lengths.
    * ``tdata.uns[key_added]``: the fit, with ``method``, ``mutation_rate``
      (per site and unit time; a list with one rate per character when
      ``relative_rates`` is given), ``log_likelihood`` (without pseudocounts),
      ``penalized_log_likelihood`` and the ``solver`` used.

    Args:
        tdata: The TreeData object to operate on.
        method: Estimation method. Only ``"convexml"`` is available; it requires
            cvxpy (``pip install 'cassiopeia-lineage[convexml]'``).
        tree_key: The ``obst`` key of the tree to use.
        characters_key: Node attribute holding the character states of every
            node, as written by :func:`cassiopeia.tl.ancestral_characters`.
        missing_state: Missing-data value. Resolved from the tdata when ``None``.
        unmodified_state: Unmodified (uncut) state value. Resolved from the tdata
            when ``None``.
        leaf_time_key: Column of ``tdata.obs`` with the sampling time of each
            leaf, measured from the root (e.g. ``"timepoint"`` from
            :func:`cassiopeia.sim.sample_timepoints`). Relaxes the assumption
            that all leaves are sampled at the same time, and node times and the
            mutation rate are reported in the units of this column.
        relative_rates: Relative mutation rate of each character, in the order
            of the character matrix (a Series is aligned to its columns). By
            default all characters mutate at the same rate.
        min_branch_length: Minimum length of each branch as a fraction of the
            depth of the deepest leaf. Without it, the estimate puts branches
            without mutations at length 0.
        min_leaf_branch_length: Minimum length of branches to leaves, which do
            not end in a cell division. Defaults to half of
            ``min_branch_length``.
        pseudo_mutations: Number of fictitious mutations added to each edge,
            spread over the characters.
        pseudo_non_mutations: Number of fictitious non-mutations added to each
            edge, spread over the characters.
        resolve_multifurcations: Resolve nodes with more than two children into
            binary splits for the estimate, as ConvexML does. A multifurcation
            hides cell divisions, so its children are placed as many minimum
            branch lengths below it as the binary splits need. A root with
            several children also gets an implicit branch from the founder.
            The implicit nodes are not added to the tree.
        solver: Convex solver, or solvers to try in order: ``"CLARABEL"``,
            ``"SCS"``, ``"ECOS"`` or ``"MOSEK"`` (installed separately).
        time_key: Node attribute under which to store node times.
        length_key: Edge attribute under which to store branch lengths.
        key_added: ``tdata.uns`` key under which to store the fit.
        copy: If ``True``, operate on and return a copy of *tdata*; otherwise
            modify in place and return ``None``.

    Returns:
        A modified copy of *tdata* if ``copy=True``, else ``None``.

    Raises:
        TypeError: If *tdata* is not a TreeData object.
        CassiopeiaError: If a node has no character states, a state is ambiguous,
            or the solver fails.
        ValueError: If the parameters are invalid, ``min_branch_length`` is too
            large for the tree, or the data have no edits (or no unmodified
            states) and the matching pseudocount is 0.
    """
    if method not in _METHODS:
        raise ValueError(f"Unknown method {method!r}; choose one of {', '.join(_METHODS)}.")
    if not isinstance(tdata, TreeData):
        raise TypeError(
            "estimate_branch_lengths() operates on TreeData. For a CassiopeiaTree, "
            "convert with CassiopeiaTree.to_treedata()."
        )
    solvers = [solver] if isinstance(solver, str) else list(solver)
    unknown = [name for name in solvers if name not in _SOLVERS]
    if unknown or not solvers:
        raise ValueError(f"Unknown solver {unknown}; choose from {', '.join(_SOLVERS)}.")
    if min_leaf_branch_length is None:
        min_leaf_branch_length = min_branch_length / 2
    for name, value in [
        ("min_branch_length", min_branch_length),
        ("min_leaf_branch_length", min_leaf_branch_length),
        ("pseudo_mutations", pseudo_mutations),
        ("pseudo_non_mutations", pseudo_non_mutations),
    ]:
        if not value >= 0:
            raise ValueError(f"{name} must be non-negative, got {value}.")
    cp = _import_cvxpy()

    tdata = tdata.copy() if copy else tdata
    # TreeData stores frozen graphs; operate on a copy and write back.
    g, tree_key = _get_digraph(tdata, tree_key, copy=True)
    missing_states = _normalize_missing(_get_parameter(tdata, "missing_state", value=missing_state))
    unmodified_states = _normalize_missing(
        _get_parameter(tdata, "unmodified_state", value=unmodified_state)
    )

    root = _get_root(g)
    children = {node: list(g.successors(node)) for node in g.nodes}
    nodes = _preorder(children, root)
    if len(nodes) < 2:
        raise ValueError("The tree must have at least one edge.")
    for node in nodes:
        if characters_key not in g.nodes[node]:
            raise CassiopeiaError(
                f"Node {node!r} has no character states under {characters_key!r}. "
                "Call cassiopeia.tl.ancestral_characters first."
            )
    if len({len(g.nodes[node][characters_key]) for node in nodes}) > 1:
        raise CassiopeiaError(
            f"Nodes have different numbers of characters under {characters_key!r}."
        )
    values = pd.DataFrame([list(g.nodes[node][characters_key]) for node in nodes], dtype=object)
    codes, _ = _encode_states(values.to_numpy(), missing_states, unmodified_states)
    states = dict(zip(nodes, codes, strict=True))

    leaves = [node for node in nodes if not children[node]]
    depth, max_time = _leaf_depths(tdata, leaves, leaf_time_key)
    rates = _relative_rates(tdata, characters_key, relative_rates, codes.shape[1])

    if resolve_multifurcations:
        children = _resolve_multifurcations(children, states, root)
    states[root] = np.zeros(codes.shape[1], dtype=np.int64)

    times, site_rates, log_likelihood, penalized, used = _convexml(
        cp,
        children,
        states,
        root,
        leaves,
        depth,
        rates,
        min_branch_length,
        min_leaf_branch_length,
        pseudo_mutations,
        pseudo_non_mutations,
        solvers,
    )

    for node in nodes:
        g.nodes[node][time_key] = float(times[node] * max_time)
    for parent, child in g.edges:
        g.edges[parent, child][length_key] = g.nodes[child][time_key] - g.nodes[parent][time_key]
    tdata.obst[tree_key] = g

    site_rates = site_rates / max_time
    tdata.uns[key_added] = {
        "method": method,
        "mutation_rate": float(site_rates[0]) if relative_rates is None else site_rates.tolist(),
        "log_likelihood": log_likelihood,
        "penalized_log_likelihood": penalized,
        "solver": used,
    }

    return tdata if copy else None
