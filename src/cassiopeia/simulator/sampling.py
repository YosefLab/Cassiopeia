"""Functional leaf sampling functions for Cassiopeia."""

import warnings
from collections import defaultdict
from collections.abc import Mapping
from typing import Literal, overload

import networkx as nx
import numpy as np
import pandas as pd
import treedata as td

from cassiopeia.mixins import LeafSubsamplerError, LeafSubsamplerWarning
from cassiopeia.simulator.tracing import _check_node_attr, _rebuild_matrix
from cassiopeia.utils import _collapse_unifurcations, _get_digraph, _normalize_missing


@overload
def sample_uniform(
    tdata: td.TreeData,
    ratio: float | None = None,
    number_of_leaves: int | None = None,
    keep_root_edge: bool = True,
    random_seed: int | None = None,
    tree_key: str | None = None,
    key_added: str = "sampled",
    copy: Literal[True] = True,
) -> td.TreeData: ...
@overload
def sample_uniform(
    tdata: td.TreeData,
    ratio: float | None = None,
    number_of_leaves: int | None = None,
    keep_root_edge: bool = True,
    random_seed: int | None = None,
    tree_key: str | None = None,
    key_added: str = "sampled",
    *,
    copy: Literal[False],
) -> None: ...
def sample_uniform(
    tdata: td.TreeData,
    ratio: float | None = None,
    number_of_leaves: int | None = None,
    keep_root_edge: bool = True,
    random_seed: int | None = None,
    tree_key: str | None = None,
    key_added: str = "sampled",
    copy: bool = True,
) -> td.TreeData | None:
    """Uniformly subsample leaves from a TreeData.

    Selects a random subset of the leaves of ``tdata.obst[tree_key]`` and returns
    the induced subtree on those leaves. Unifurcations created by leaf removal
    are collapsed in every tree of the result.

    Args:
        tdata: TreeData with a tree in ``obst[tree_key]``.
        ratio: Fraction of leaves to keep (rounded down). Mutually exclusive
            with ``number_of_leaves``.
        number_of_leaves: Exact number of leaves to keep. Mutually exclusive
            with ``ratio``.
        keep_root_edge: If ``True`` (default), preserve the root's single
            child edge when the root is a unifurcation after pruning. If
            ``False``, collapse that edge into the root.
        random_seed: NumPy random seed for reproducibility.
        tree_key: Key in ``tdata.obst`` for the tree. Only required if
            ``tdata`` contains multiple trees.
        key_added: Key under which to store the sampled tree when
            ``copy=False``.
        copy: If ``True`` (default), return a new TreeData with the sampled
            subtree. If ``False``, add the sampled tree to
            ``tdata.obst[key_added]`` in-place and return ``None``.

    Returns:
        If ``copy=True`` (default), a new TreeData with the induced subtree on
        the sampled leaves; otherwise ``None`` (the sampled tree is added to
        ``tdata.obst[key_added]``).

    Raises:
        LeafSubsamplerError: On invalid parameters or sample size.
    """
    tree, tree_key = _get_digraph(tdata, tree_key)
    if (ratio is None) == (number_of_leaves is None):
        raise LeafSubsamplerError("Specify exactly one of `ratio` or `number_of_leaves`.")
    if random_seed is not None:
        np.random.seed(random_seed)

    leaves = _tree_leaves(tdata, tree)
    n_keep = number_of_leaves if number_of_leaves is not None else int(len(leaves) * ratio)

    if n_keep <= 0:
        raise LeafSubsamplerError("Number of leaves to keep is <= 0.")
    if n_keep > len(leaves):
        raise LeafSubsamplerError(
            f"Number of leaves to keep ({n_keep}) exceeds tree size ({len(leaves)})."
        )

    keep = [str(x) for x in np.random.choice(leaves, n_keep, replace=False)]
    result = _prune_tdata(tdata, keep, keep_root_edge, tree_key)
    if copy:
        return result
    tdata.obst[key_added] = result.obst[tree_key]


@overload
def sample_spatial(
    tdata: td.TreeData,
    bounding_box: list[tuple] | None = None,
    space: np.ndarray | None = None,
    spatial_key: str = "spatial",
    keep_root_edge: bool = True,
    tree_key: str | None = None,
    key_added: str = "sampled",
    copy: Literal[True] = True,
) -> td.TreeData: ...
@overload
def sample_spatial(
    tdata: td.TreeData,
    bounding_box: list[tuple] | None = None,
    space: np.ndarray | None = None,
    spatial_key: str = "spatial",
    keep_root_edge: bool = True,
    tree_key: str | None = None,
    key_added: str = "sampled",
    *,
    copy: Literal[False],
) -> None: ...
def sample_spatial(
    tdata: td.TreeData,
    bounding_box: list[tuple] | None = None,
    space: np.ndarray | None = None,
    spatial_key: str = "spatial",
    keep_root_edge: bool = True,
    tree_key: str | None = None,
    key_added: str = "sampled",
    copy: bool = True,
) -> td.TreeData | None:
    """Subsample leaves within a spatial region of interest.

    Subsets leaves to those within a bounding box or binary mask. Spatial
    coordinates are read from ``tdata.obsm[spatial_key]``. To select an
    exact leaf count, compose with :func:`sample_uniform`; to merge cells
    at the same pixel, compose with :func:`sample_supercellular`::

        tdata = sample_spatial(tdata, space=mask)
        tdata = sample_uniform(tdata, number_of_leaves=100)
        tdata = sample_supercellular(tdata, spatial_key="spatial")

    Args:
        tdata: TreeData with a tree in ``obst[tree_key]`` and spatial
            coordinates in ``obsm[spatial_key]``.
        bounding_box: List of ``(min, max)`` tuples, one per spatial
            dimension. Leaves within these bounds are kept. Mutually
            exclusive with ``space``.
        space: Boolean numpy array defining the region of interest.
            Coordinates are cast to integers; a leaf is kept when
            ``space[tuple(int_coords)]`` is ``True``. Mutually exclusive
            with ``bounding_box``.
        spatial_key: Key in ``tdata.obsm`` holding spatial coordinates.
        keep_root_edge: Preserve root's single child edge after pruning.
        tree_key: Key in ``tdata.obst`` for the tree. Only required if
            ``tdata`` contains multiple trees.
        key_added: Key under which to store the sampled tree when
            ``copy=False``.
        copy: If ``True`` (default), return a new TreeData with the sampled
            subtree. If ``False``, add the sampled tree to
            ``tdata.obst[key_added]`` in-place and return ``None``.

    Returns:
        If ``copy=True`` (default), a new TreeData with the spatially filtered
        leaves; otherwise ``None`` (the sampled tree is added to
        ``tdata.obst[key_added]``).

    Raises:
        LeafSubsamplerError: On invalid parameters or empty region.
    """
    tree, tree_key = _get_digraph(tdata, tree_key)
    if (bounding_box is None) == (space is None):
        raise LeafSubsamplerError("Specify exactly one of `bounding_box` or `space`.")
    if spatial_key not in tdata.obsm:
        raise LeafSubsamplerError(f"Spatial key `{spatial_key}` not present in tdata.obsm.")

    leaves = _tree_leaves(tdata, tree)
    coords_raw = tdata.obsm[spatial_key]
    if isinstance(coords_raw, pd.DataFrame):
        coords_arr = coords_raw.values
    else:
        coords_arr = np.asarray(coords_raw)
    position = {name: i for i, name in enumerate(tdata.obs_names)}
    leaf_coords = {leaf: coords_arr[position[leaf]] for leaf in leaves}

    # Filter by region
    if bounding_box is not None:
        ndim = len(leaf_coords[leaves[0]])
        if len(bounding_box) != ndim:
            raise LeafSubsamplerError(
                f"Coordinate dimensions ({ndim}) and bounding_box length "
                f"({len(bounding_box)}) do not match."
            )
        leaf_keep = [
            leaf
            for leaf in leaves
            if all(lo <= leaf_coords[leaf][d] <= hi for d, (lo, hi) in enumerate(bounding_box))
        ]
    else:
        ndim = len(leaf_coords[leaves[0]])
        if len(space.shape) != ndim:
            raise LeafSubsamplerError(
                f"Coordinate dimensions ({ndim}) and space rank ({len(space.shape)}) do not match."
            )
        max_coord = max(np.max(np.abs(c)) for c in leaf_coords.values()) if leaf_coords else 0
        if max_coord > 0 and max_coord * 10 < np.max(space.shape):
            warnings.warn(
                f"Maximum coordinate {max_coord} is much smaller than maximum "
                f"space dimension {np.max(space.shape)}. Consider rescaling "
                f"since coordinates are converted to integers for spatial filtering.",
                LeafSubsamplerWarning,
                stacklevel=2,
            )
        leaf_keep = []
        for leaf in leaves:
            c = tuple(int(x) for x in leaf_coords[leaf])
            if any(x < 0 or x >= s for x, s in zip(c, space.shape, strict=False)):
                raise LeafSubsamplerError(
                    f"Coordinates {c} for leaf '{leaf}' are outside the space."
                )
            if space[c]:
                leaf_keep.append(leaf)

    if len(leaf_keep) == 0:
        raise LeafSubsamplerError("No leaves within the specified region.")

    result = _prune_tdata(tdata, leaf_keep, keep_root_edge, tree_key)
    if copy:
        return result
    tdata.obst[key_added] = result.obst[tree_key]


def sample_supercellular(
    tdata: td.TreeData,
    ratio: float | None = None,
    number_of_merges: int | None = None,
    spatial_key: str | None = None,
    keep_root_edge: bool = True,
    collapse_duplicates: bool = True,
    random_seed: int | None = None,
    tree_key: str | None = None,
    time_key: str = "time",
) -> td.TreeData:
    """Merge pairs of leaves to simulate supercellular observations.

    Two merging modes are available, selected by ``spatial_key``:

    **Iterative mode** (``spatial_key=None``, default): pairs of leaves are
    selected and merged iteratively until a stopping condition is met. The
    first leaf is chosen uniformly at random; the second is chosen with
    probability inversely proportional to branch distance from the first.
    Merged leaves can be selected for further merging in subsequent rounds.
    Requires exactly one of ``ratio`` or ``number_of_merges``.

    **Pixel mode** (``spatial_key`` provided): leaves that share the same
    integer pixel in ``tdata.obsm[spatial_key]`` are merged in one pass.
    Can be composed after :func:`sample_spatial`::

        tdata = sample_spatial(tdata, space=mask)
        tdata = sample_supercellular(tdata, spatial_key="spatial")

    In both modes, the states of merged leaves are combined into ambiguous
    states: tuples of the states observed in the merged cells, sorted, such as
    ``("1", "4")``. This is the format accepted by the solvers (see
    :func:`cassiopeia.pp.alleles_to_characters` with ``allow_ambiguous=True``).
    Missing states (``tdata.uns["missing_state"]``) are dropped unless all
    merged cells are missing, and a state shared by all merged cells is kept
    as a single state rather than a tuple. Other ``obsm`` DataFrames are
    combined the same way.

    Args:
        tdata: TreeData with a tree in ``obst[tree_key]``.
        ratio: Number of merges as a fraction of the total leaf count.
            Iterative mode only; mutually exclusive with ``number_of_merges``.
        number_of_merges: Exact number of merge operations. Iterative mode
            only; mutually exclusive with ``ratio``.
        spatial_key: Key in ``tdata.obsm`` holding spatial coordinates.
            When provided, activates pixel mode and ``ratio`` /
            ``number_of_merges`` must not be set.
        keep_root_edge: Preserve root's single child edge after pruning.
        collapse_duplicates: If ``True``, deduplicate repeated states within a
            merged state (e.g., ``("1", "1", "2")`` → ``("1", "2")`` and
            ``("1", "1")`` → ``"1"``).
        random_seed: NumPy random seed for reproducibility (iterative mode).
        tree_key: Key in ``tdata.obst`` for the tree. Only required if
            ``tdata`` contains multiple trees.
        time_key: Node attribute key for time values (iterative mode).
             Required for distance-based merging; ignored in pixel mode.
             Must be present on all nodes.

    Returns:
        A new TreeData with merged leaves and combined character states, or
        ``None`` if ``copy=False``.

    Raises:
        LeafSubsamplerError: On invalid parameters.
    """
    _, tree_key = _get_digraph(tdata, tree_key)
    pixel_mode = spatial_key is not None

    if pixel_mode:
        if ratio is not None or number_of_merges is not None:
            raise LeafSubsamplerError(
                "`ratio` and `number_of_merges` must not be set in pixel mode "
                "(when `spatial_key` is provided)."
            )
        if spatial_key not in tdata.obsm:
            raise LeafSubsamplerError(f"Spatial key `{spatial_key}` not present in tdata.obsm.")
        return _pixel_merge(
            tdata, spatial_key, keep_root_edge, collapse_duplicates, tree_key, time_key
        )

    # Iterative mode
    if (ratio is None) == (number_of_merges is None):
        raise LeafSubsamplerError("Specify exactly one of `ratio` or `number_of_merges`.")

    if random_seed is not None:
        np.random.seed(random_seed)

    leaves = list(tdata.obs_names)
    n_merges = number_of_merges if number_of_merges is not None else int(len(leaves) * ratio)

    if n_merges >= len(leaves):
        raise LeafSubsamplerError("Number of merges must be less than the number of leaves.")
    if n_merges <= 0:
        raise LeafSubsamplerError("Number of merges must be > 0.")

    working_tree = tdata.obst[tree_key].copy()
    current_leaves = list(leaves)
    missing = _missing_states(tdata)

    # Snapshot obsm data per leaf as lists of states
    obsm_data: dict[str, dict[str, list]] = {}
    obsm_cols: dict[str, list] = {}
    for key, val in tdata.obsm.items():
        if isinstance(val, pd.DataFrame):
            obsm_cols[key] = list(val.columns)
            obsm_data[key] = {leaf: list(val.loc[leaf]) for leaf in leaves}

    for _ in range(n_merges):
        leaf1 = str(np.random.choice(current_leaves))
        other = [l for l in current_leaves if l != leaf1]

        distances = np.array([_leaf_distance(working_tree, leaf1, l, time_key) for l in other])
        weights = 1.0 / np.maximum(distances, 1e-10)
        probs = weights / weights.sum()
        leaf2 = str(np.random.choice(other, p=probs))

        lca = nx.lowest_common_ancestor(working_tree, leaf1, leaf2)
        new_time = (working_tree.nodes[leaf1][time_key] + working_tree.nodes[leaf2][time_key]) / 2
        new_leaf = f"{leaf1}-{leaf2}"

        working_tree.add_node(new_leaf, **{time_key: new_time})
        working_tree.add_edge(lca, new_leaf)

        for key in obsm_data:
            s1 = obsm_data[key][leaf1]
            s2 = obsm_data[key][leaf2]
            obsm_data[key][new_leaf] = [
                _merge_states([a, b], collapse_duplicates, missing)
                for a, b in zip(s1, s2, strict=False)
            ]

        current_leaves = [l for l in current_leaves if l != leaf1 and l != leaf2]
        current_leaves.append(new_leaf)

    final_leaves = current_leaves
    new_tree = _induce_and_collapse(working_tree, final_leaves, keep_root_edge)

    new_obs = pd.DataFrame(index=final_leaves)
    new_obsm = {}
    for key, cols in obsm_cols.items():
        new_obsm[key] = pd.DataFrame(
            {leaf: obsm_data[key][leaf] for leaf in final_leaves},
            index=cols,
        ).T.infer_objects()

    return td.TreeData(
        obs=new_obs,
        obsm=new_obsm,
        obst={tree_key: new_tree},
        uns=dict(tdata.uns),
    )


@overload
def sample_timepoints(
    tdata: td.TreeData,
    timepoints: dict[float, int],
    keep_root_edge: bool = True,
    random_seed: int | None = None,
    tree_key: str | None = None,
    time_key: str = "time",
    key_added: str = "sampled",
    copy: Literal[True] = True,
) -> td.TreeData: ...
@overload
def sample_timepoints(
    tdata: td.TreeData,
    timepoints: dict[float, int],
    keep_root_edge: bool = True,
    random_seed: int | None = None,
    tree_key: str | None = None,
    time_key: str = "time",
    key_added: str = "sampled",
    *,
    copy: Literal[False],
) -> None: ...
def sample_timepoints(
    tdata: td.TreeData,
    timepoints: dict[float, int],
    keep_root_edge: bool = True,
    random_seed: int | None = None,
    tree_key: str | None = None,
    time_key: str = "time",
    key_added: str = "sampled",
    copy: bool = True,
) -> td.TreeData | None:
    """Sample cells at multiple timepoints from a growing tree.

    Simulates sequential destructive sampling: at each timepoint (processed
    in ascending order) a set of currently-alive cells is drawn without
    replacement. Once a cell is sampled its entire lineage is terminated —
    none of its progeny are eligible at any later timepoint. Cells from all
    timepoints are pooled as leaves in the returned tree.

    A cell is alive at time ``t`` when its parent has already divided
    (``parent_time <= t``) and it has not yet divided (``t < node_time``
    for internal nodes) or has survived to its leaf time (``t <= node_time``
    for leaves).

    Results are written to a new TreeData:

    * ``tdata.obs["timepoint"]``: sampling timepoint for each cell.
    * ``tdata.obs``: other columns carried over for cells that were leaves in
      the input tree; ``NaN`` for cells sampled at intermediate timepoints.
    * ``tdata.obsm``: rows of ``tdata.obsm`` (DataFrames and arrays) for
      original leaves, and node attributes of the same name for intermediate
      cells. Entries without values for every sampled cell are dropped with a
      warning.

    Args:
        tdata: TreeData with a tree in ``obst[tree_key]``.
        timepoints: Mapping of timepoint → number of cells to sample at that
            time. Timepoints are processed in ascending order.
        keep_root_edge: Preserve root's single child edge after pruning.
        random_seed: NumPy random seed for reproducibility.
        tree_key: Key in ``tdata.obst`` for the tree. Only required if
            ``tdata`` contains multiple trees.
        time_key: Node attribute key for time values. Must be present on all nodes.
        key_added: Key under which to store the sampled tree when
            ``copy=False``.
        copy: If ``True`` (default), return a new TreeData with one leaf per
            sampled cell. If ``False``, add the sampled tree to
            ``tdata.obst[key_added]`` in-place and return ``None``.

    Returns:
        If ``copy=True`` (default), a new TreeData with one leaf per sampled
        cell and a ``"timepoint"`` column in ``obs``; otherwise ``None`` (the
        sampled tree is added to ``tdata.obst[key_added]``).

    Raises:
        LeafSubsamplerError: If the timepoints dict is empty, a node has no
            time, a requested count exceeds the number of available cells, or
            no cells are sampled.
    """
    tree, tree_key = _get_digraph(tdata, tree_key)
    if not timepoints:
        raise LeafSubsamplerError("timepoints dict is empty.")
    _check_node_attr(tree, time_key, LeafSubsamplerError)
    if random_seed is not None:
        np.random.seed(random_seed)

    original_leaves = set(tdata.obs_names)

    parent_times: dict[str, float] = {
        node: (
            tree.nodes[next(iter(tree.predecessors(node)))][time_key]
            if tree.in_degree(node) > 0
            else float("-inf")
        )
        for node in tree.nodes
    }

    def _alive_at(node: str, t: float) -> bool:
        pt = parent_times[node]
        nt = tree.nodes[node][time_key]
        # Internal cells are alive strictly before their division time;
        # leaves are alive up to and including their leaf time.
        if tree.out_degree(node) == 0:
            return pt <= t <= nt
        return pt <= t < nt

    available: set[str] = set(tree.nodes)
    sampled: dict[str, float] = {}  # node → sampling timepoint

    for t in sorted(timepoints):
        n_sample = timepoints[t]
        if n_sample == 0:
            continue
        # Iterate in tree order, not set order (which varies between processes).
        alive = [n for n in tree.nodes if n in available and _alive_at(n, t)]
        if len(alive) < n_sample:
            raise LeafSubsamplerError(
                f"Only {len(alive)} cells alive at timepoint {t}, but {n_sample} were requested."
            )
        chosen = [str(n) for n in np.random.choice(alive, n_sample, replace=False)]
        sampled.update(dict.fromkeys(chosen, t))
        terminated: set[str] = set(chosen)
        for node in chosen:
            terminated |= nx.descendants(tree, node)
        available -= terminated

    if not sampled:
        raise LeafSubsamplerError("No cells were sampled.")

    # Sampled nodes form an antichain (no sampled node is an ancestor of
    # another), so they appear naturally as leaves in the induced subtree.
    ancestors_needed: set[str] = set()
    for node in sampled:
        ancestors_needed |= nx.ancestors(tree, node)
    induced = _induced_subgraph(tree, ancestors_needed | set(sampled))

    # For cells sampled before their natural division time, override time
    # to reflect when they were actually sampled.
    for node, t in sampled.items():
        induced.nodes[node][time_key] = t

    new_tree = _collapse_unifurcations(induced, collapse_root=not keep_root_edge)

    # obs: "timepoint" column for all sampled cells; carry over existing
    # columns for cells that were already leaves in the input tree.
    sampled_list = list(sampled)
    new_obs = pd.DataFrame({"timepoint": sampled}, index=sampled_list)
    for col in tdata.obs.columns:
        new_obs[col] = pd.Series(
            {node: tdata.obs.loc[node, col] for node in sampled if node in original_leaves}
        )

    # obsm: use tdata.obsm for original leaves; fall back to node attributes
    # for cells sampled at intermediate timepoints.
    position = {name: i for i, name in enumerate(tdata.obs_names)}
    new_obsm: dict = {}
    for key, val in tdata.obsm.items():
        if not isinstance(val, (pd.DataFrame, np.ndarray)):
            continue
        rows = _obsm_rows(val, key, sampled_list, position, tree)
        if rows is None:
            warnings.warn(
                f"obsm[{key!r}] is dropped: some cells sampled at intermediate timepoints "
                f"have no {key!r} node attribute.",
                LeafSubsamplerWarning,
                stacklevel=2,
            )
        elif isinstance(val, pd.DataFrame):
            new_obsm[key] = _rebuild_matrix(rows, val, sampled_list)[0]
        else:
            new_obsm[key] = np.asarray(rows)

    result = td.TreeData(
        obs=new_obs,
        obsm=new_obsm,
        obst={tree_key: new_tree},
        uns=dict(tdata.uns),
    )
    if copy:
        return result
    tdata.obst[key_added] = result.obst[tree_key]


# --- Private helpers ---


def _tree_leaves(tdata: td.TreeData, tree: nx.DiGraph) -> list[str]:
    """Observations of *tdata* that are leaves of *tree*, in ``obs_names`` order.

    Differs from ``obs_names`` when observations include internal nodes
    (``alignment="nodes"``) or cells of other trees.
    """
    return [name for name in tdata.obs_names if name in tree and tree.out_degree(name) == 0]


def _obsm_rows(
    val: pd.DataFrame | np.ndarray,
    key: str,
    nodes: list[str],
    position: dict[str, int],
    tree: nx.DiGraph,
) -> list | None:
    """Rows of the obsm entry *val* for *nodes*, or ``None`` if some are unavailable.

    Observed nodes take their row of *val*; others take their node attribute
    *key* (a sequence aligned with the columns, or a mapping keyed by column).
    """
    frame = val if isinstance(val, pd.DataFrame) else None
    n_cols = val.shape[1] if val.ndim > 1 else None
    rows = []
    for node in nodes:
        if node in position:
            i = position[node]
            rows.append(frame.iloc[i].tolist() if frame is not None else val[i])
            continue
        attr = tree.nodes[node].get(key) if node in tree else None
        if isinstance(attr, Mapping) and frame is not None:
            attr = [attr.get(col) for col in frame.columns]
        if not hasattr(attr, "__len__") or isinstance(attr, str):  # missing or scalar
            return None
        if n_cols is not None and len(attr) != n_cols:
            return None
        rows.append(list(attr))
    return rows


def _prune_tdata(
    tdata: td.TreeData,
    keep_leaves: list[str],
    keep_root_edge: bool,
    tree_key: str,
) -> td.TreeData:
    """Return new TreeData pruned to *keep_leaves*, collapsing unifurcations in every tree.

    With ``alignment="nodes"`` the observations are tree nodes: the sampled
    leaves, their ancestors kept after collapsing, and the nodes of other trees.
    """
    if tdata.alignment == "leaves":
        sub = tdata[list(keep_leaves)].copy()
    else:
        tree = tdata.obst[tree_key]
        keep = set(keep_leaves)
        for leaf in keep_leaves:
            keep |= nx.ancestors(tree, leaf)
        sub = tdata[[name for name in tdata.obs_names if name in keep]].copy()
    for key in list(sub.obst.keys()):
        sub.obst[key] = _collapse_unifurcations(sub.obst[key], collapse_root=not keep_root_edge)
    if tdata.alignment != "leaves":
        # Drop the observations of internal nodes removed by collapsing.
        nodes = set().union(*(sub.obst[key].nodes for key in sub.obst.keys()))
        sub = sub[[name for name in sub.obs_names if name in nodes]].copy()
    return sub


def _pixel_merge(
    tdata: td.TreeData,
    spatial_key: str,
    keep_root_edge: bool,
    collapse_duplicates: bool,
    tree_key: str,
    time_key: str = "time",
) -> td.TreeData:
    """Merge leaves that share the same integer pixel coordinate."""
    leaves = list(tdata.obs_names)
    coords_raw = tdata.obsm[spatial_key]
    if isinstance(coords_raw, pd.DataFrame):
        coords_arr = coords_raw.values
    else:
        coords_arr = np.asarray(coords_raw)
    leaf_coords = {leaf: coords_arr[i] for i, leaf in enumerate(leaves)}

    pixel_groups: dict[tuple, list[str]] = defaultdict(list)
    for leaf in leaves:
        pixel = tuple(int(x) for x in leaf_coords[leaf])
        pixel_groups[pixel].append(leaf)

    merge_map: dict[str, list[str]] = {}
    single_leaves: list[str] = []
    pixel_of: dict[str, tuple] = {}

    for pixel, group in pixel_groups.items():
        if len(group) == 1:
            single_leaves.append(group[0])
        else:
            new_name = "-".join(sorted(group))
            merge_map[new_name] = group
            pixel_of[new_name] = pixel

    if not merge_map:
        return _prune_tdata(tdata, leaves, keep_root_edge, tree_key)

    final_leaves = single_leaves + list(merge_map.keys())
    new_tree = _build_merged_tree(
        tdata.obst[tree_key], merge_map, final_leaves, keep_root_edge, time_key
    )
    new_obs = pd.DataFrame(index=final_leaves)
    new_obsm = _merge_obsm(
        tdata.obsm,
        final_leaves,
        merge_map,
        spatial_key=spatial_key,
        pixel_of=pixel_of,
        collapse_duplicates=collapse_duplicates,
        missing=_missing_states(tdata),
    )
    return td.TreeData(
        obs=new_obs,
        obsm=new_obsm,
        obst={tree_key: new_tree},
        uns=dict(tdata.uns),
    )


def _leaf_distance(tree: nx.DiGraph, leaf1: str, leaf2: str, time_key: str = "time") -> float:
    """Branch distance between two leaves via their LCA."""
    lca = nx.lowest_common_ancestor(tree, leaf1, leaf2)
    return tree.nodes[leaf1][time_key] + tree.nodes[leaf2][time_key] - 2 * tree.nodes[lca][time_key]


def _missing_states(tdata: td.TreeData) -> set:
    """Missing-state values of *tdata* (``uns["missing_state"]`` or the defaults)."""
    return _normalize_missing(tdata.uns.get("missing_state", (-1, "-1", "NA", "-")))


def _merge_states(values: list, collapse_duplicates: bool, missing: set):
    """Combine the states of merged cells into one (possibly ambiguous) state.

    Tuples (states of previously merged cells) are flattened and missing states
    are dropped unless every state is missing. Several remaining states are
    returned as a sorted tuple; a single one is returned as is.
    """
    states = []
    for value in values:
        states.extend(value if isinstance(value, tuple) else (value,))
    observed = [s for s in states if s not in missing]
    if not observed:
        return states[0]
    if collapse_duplicates:
        observed = list(dict.fromkeys(observed))
    if len(observed) == 1:
        return observed[0]
    return tuple(sorted(observed, key=str))


def _induced_subgraph(tree: nx.DiGraph, nodes: set) -> nx.DiGraph:
    """Copy of the subgraph induced by *nodes*, in the node order of *tree*.

    ``tree.subgraph(nodes)`` iterates over the set of nodes, whose order depends on
    ``PYTHONHASHSEED``; downstream random draws (e.g. ``missing_data``) follow the node
    order, so seeded simulations would differ between processes.
    """
    induced = nx.DiGraph(**tree.graph)
    induced.add_nodes_from((n, data) for n, data in tree.nodes(data=True) if n in nodes)
    induced.add_edges_from(
        (u, v, data) for u, v, data in tree.edges(data=True) if u in nodes and v in nodes
    )
    return induced


def _induce_and_collapse(
    tree: nx.DiGraph, final_leaves: list[str], keep_root_edge: bool
) -> nx.DiGraph:
    """Build induced subtree on final_leaves and collapse unifurcations."""
    ancestors: set[str] = set()
    for leaf in final_leaves:
        ancestors |= nx.ancestors(tree, leaf)
    induced = _induced_subgraph(tree, ancestors | set(final_leaves))
    return _collapse_unifurcations(induced, collapse_root=not keep_root_edge)


def _build_merged_tree(
    orig_tree: nx.DiGraph,
    merge_map: dict[str, list[str]],
    final_leaves: list[str],
    keep_root_edge: bool,
    time_key: str = "time",
) -> nx.DiGraph:
    """Add merged leaf nodes to tree, then induce subtree on final_leaves."""
    working = orig_tree.copy()
    for new_name, old_leaves in merge_map.items():
        lca = old_leaves[0]
        for other in old_leaves[1:]:
            lca = nx.lowest_common_ancestor(working, lca, other)
        new_time = float(np.mean([working.nodes[l][time_key] for l in old_leaves]))
        working.add_node(new_name, **{time_key: new_time})
        working.add_edge(lca, new_name)
    return _induce_and_collapse(working, final_leaves, keep_root_edge)


def _merge_obsm(
    obsm: dict,
    final_leaves: list[str],
    merge_map: dict[str, list[str]],
    spatial_key: str,
    pixel_of: dict[str, tuple],
    collapse_duplicates: bool,
    missing: set,
) -> dict:
    """Build merged obsm for pixel merges."""
    new_obsm = {}
    for key, val in obsm.items():
        if not isinstance(val, pd.DataFrame):
            continue
        cols = list(val.columns)
        rows: dict[str, list] = {}

        for leaf in final_leaves:
            if leaf not in merge_map:
                if leaf in val.index:
                    rows[leaf] = list(val.loc[leaf])
            else:
                old_leaves = merge_map[leaf]
                if key == spatial_key:
                    rows[leaf] = list(pixel_of[leaf])
                else:
                    col_vals = [list(val.loc[l]) for l in old_leaves]
                    rows[leaf] = [
                        _merge_states(list(group), collapse_duplicates, missing)
                        for group in zip(*col_vals, strict=False)
                    ]

        if rows:
            new_obsm[key] = pd.DataFrame(rows, index=cols).T[cols].infer_objects()

    return new_obsm
