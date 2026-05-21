"""Functional leaf sampling functions for Cassiopeia."""

import warnings
from collections import defaultdict

import networkx as nx
import numpy as np
import pandas as pd
import treedata as td

from cassiopeia.mixins import LeafSubsamplerError, LeafSubsamplerWarning
from cassiopeia.utils import collapse_unifurcations


def sample_uniform(
    tdata: td.TreeData,
    ratio: float | None = None,
    number_of_leaves: int | None = None,
    keep_root_edge: bool = True,
    random_seed: int | None = None,
    tree_key: str = "tree",
) -> td.TreeData:
    """Uniformly subsample leaves from a TreeData.

    Selects a random subset of leaves and returns the induced subtree on
    those leaves. Unifurcations created by leaf removal are collapsed.

    Args:
        tdata: TreeData with a tree in ``obst[tree_key]``. Nodes must have a
            ``"time"`` attribute.
        ratio: Fraction of leaves to keep (rounded down). Mutually exclusive
            with ``number_of_leaves``.
        number_of_leaves: Exact number of leaves to keep. Mutually exclusive
            with ``ratio``.
        keep_root_edge: If ``True`` (default), preserve the root's single
            child edge when the root is a unifurcation after pruning. If
            ``False``, collapse that edge into the root.
        random_seed: NumPy random seed for reproducibility.
        tree_key: Key in ``tdata.obst`` for the tree.

    Returns:
        A new TreeData with the induced subtree on the sampled leaves.

    Raises:
        LeafSubsamplerError: On invalid parameters or sample size.
    """
    if (ratio is None) == (number_of_leaves is None):
        raise LeafSubsamplerError(
            "Specify exactly one of `ratio` or `number_of_leaves`."
        )
    if random_seed is not None:
        np.random.seed(random_seed)

    leaves = list(tdata.obs_names)
    n_keep = number_of_leaves if number_of_leaves is not None else int(len(leaves) * ratio)

    if n_keep <= 0:
        raise LeafSubsamplerError("Number of leaves to keep is <= 0.")
    if n_keep > len(leaves):
        raise LeafSubsamplerError(
            f"Number of leaves to keep ({n_keep}) exceeds tree size ({len(leaves)})."
        )

    keep = [str(x) for x in np.random.choice(leaves, n_keep, replace=False)]
    return _prune_tdata(tdata, keep, keep_root_edge, tree_key)


def sample_spatial(
    tdata: td.TreeData,
    bounding_box: list[tuple] | None = None,
    space: np.ndarray | None = None,
    ratio: float | None = None,
    number_of_leaves: int | None = None,
    attribute_key: str = "spatial",
    merge_cells: bool = False,
    keep_root_edge: bool = True,
    collapse_duplicates: bool = True,
    random_seed: int | None = None,
    tree_key: str = "tree",
) -> td.TreeData:
    """Subsample leaves within a spatial region of interest.

    Subsets leaves to those within a bounding box or binary mask, then
    optionally downsamples uniformly. When ``merge_cells=True`` and a
    ``space`` mask is provided, leaves that map to the same integer pixel
    are merged into a single observation with combined character states.

    Spatial coordinates are read from ``tdata.obsm[attribute_key]``.

    Args:
        tdata: TreeData with a tree in ``obst[tree_key]`` and spatial
            coordinates in ``obsm[attribute_key]``.
        bounding_box: List of ``(min, max)`` tuples, one per spatial
            dimension. Leaves within these bounds are kept. Mutually
            exclusive with ``space``.
        space: Boolean numpy array defining the region of interest.
            Coordinates are cast to integers; a leaf is kept when
            ``space[tuple(int_coords)]`` is ``True``. Mutually exclusive
            with ``bounding_box``.
        ratio: Fraction of region-of-interest leaves to keep after spatial
            filtering.
        number_of_leaves: Exact number of leaves to keep after spatial
            filtering.
        attribute_key: Key in ``tdata.obsm`` holding spatial coordinates.
        merge_cells: If ``True``, merge leaves that share a pixel. Requires
            ``space``. Merged character state strings are joined with ``"|"``.
        keep_root_edge: Preserve root's single child edge after pruning.
        collapse_duplicates: Remove duplicate state tokens in merged cells
            (e.g., ``"1|1|2"`` → ``"1|2"``).
        random_seed: NumPy random seed for reproducibility.
        tree_key: Key in ``tdata.obst`` for the tree.

    Returns:
        A new TreeData with spatially subsampled (and optionally merged) leaves.

    Raises:
        LeafSubsamplerError: On invalid parameters or empty region.
    """
    if (bounding_box is None) == (space is None):
        raise LeafSubsamplerError(
            "Specify exactly one of `bounding_box` or `space`."
        )
    if ratio is not None and number_of_leaves is not None:
        raise LeafSubsamplerError(
            "Specify at most one of `ratio` or `number_of_leaves`."
        )
    if ratio is not None and (ratio <= 0 or ratio > 1):
        raise LeafSubsamplerError("`ratio` must be in (0, 1].")
    if number_of_leaves is not None and number_of_leaves <= 0:
        raise LeafSubsamplerError("`number_of_leaves` must be > 0.")
    if merge_cells and space is None:
        raise LeafSubsamplerError("`merge_cells` requires `space`.")
    if attribute_key not in tdata.obsm:
        raise LeafSubsamplerError(
            f"Attribute `{attribute_key}` not present in tdata.obsm."
        )

    if random_seed is not None:
        np.random.seed(random_seed)

    leaves = list(tdata.obs_names)
    coords_raw = tdata.obsm[attribute_key]
    if isinstance(coords_raw, pd.DataFrame):
        coords_arr = coords_raw.values
    else:
        coords_arr = np.asarray(coords_raw)
    leaf_coords = {leaf: coords_arr[i] for i, leaf in enumerate(leaves)}

    # Filter by region
    if bounding_box is not None:
        ndim = len(leaf_coords[leaves[0]])
        if len(bounding_box) != ndim:
            raise LeafSubsamplerError(
                f"Coordinate dimensions ({ndim}) and bounding_box length "
                f"({len(bounding_box)}) do not match."
            )
        leaf_keep = [
            leaf for leaf in leaves
            if all(lo <= leaf_coords[leaf][d] <= hi for d, (lo, hi) in enumerate(bounding_box))
        ]
    else:
        ndim = len(leaf_coords[leaves[0]])
        if len(space.shape) != ndim:
            raise LeafSubsamplerError(
                f"Coordinate dimensions ({ndim}) and space rank "
                f"({len(space.shape)}) do not match."
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
            if any(x < 0 or x >= s for x, s in zip(c, space.shape)):
                raise LeafSubsamplerError(
                    f"Coordinates {c} for leaf '{leaf}' are outside the space."
                )
            if space[c]:
                leaf_keep.append(leaf)

    if len(leaf_keep) == 0:
        raise LeafSubsamplerError("No leaves within the specified region.")

    # Downsample within region
    if ratio is not None:
        n_keep = int(len(leaf_keep) * ratio)
    elif number_of_leaves is not None:
        n_keep = number_of_leaves
    else:
        n_keep = len(leaf_keep)

    if n_keep <= 0:
        raise LeafSubsamplerError("Number of leaves to keep is <= 0.")
    if n_keep > len(leaf_keep):
        raise LeafSubsamplerError(
            f"Number of leaves to keep ({n_keep}) exceeds region size ({len(leaf_keep)})."
        )

    leaf_keep = [str(x) for x in np.random.choice(leaf_keep, n_keep, replace=False)]

    if not merge_cells:
        return _prune_tdata(tdata, leaf_keep, keep_root_edge, tree_key)

    # Group leaves by pixel
    pixel_groups: dict[tuple, list[str]] = defaultdict(list)
    for leaf in leaf_keep:
        pixel = tuple(int(x) for x in leaf_coords[leaf])
        pixel_groups[pixel].append(leaf)

    # Build merge map
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

    final_leaves = single_leaves + list(merge_map.keys())
    new_tree = _build_merged_tree(
        tdata.obst[tree_key], merge_map, final_leaves, keep_root_edge
    )
    new_obs = pd.DataFrame(index=final_leaves)
    new_obsm = _merge_obsm(
        tdata.obsm, final_leaves, merge_map,
        spatial_key=attribute_key, pixel_of=pixel_of,
        collapse_duplicates=collapse_duplicates,
    )
    return td.TreeData(
        obs=new_obs,
        obsm=new_obsm,
        obst={tree_key: new_tree},
        uns=dict(tdata.uns),
    )


def sample_supercellular(
    tdata: td.TreeData,
    ratio: float | None = None,
    number_of_merges: int | None = None,
    keep_root_edge: bool = True,
    collapse_duplicates: bool = True,
    random_seed: int | None = None,
    tree_key: str = "tree",
) -> td.TreeData:
    """Iteratively merge pairs of leaves to simulate supercellular observations.

    Pairs of leaves are selected and merged iteratively until a stopping
    condition is met. The first leaf is chosen uniformly at random; the second
    is chosen with probability inversely proportional to branch distance from
    the first. Merged leaves can be selected for further merging in subsequent
    rounds. Character states are combined with ``"|"`` separating tokens from
    each source leaf.

    Args:
        tdata: TreeData with a tree in ``obst[tree_key]``. Nodes must have a
            ``"time"`` attribute.
        ratio: Number of merges as a fraction of the total leaf count.
            Mutually exclusive with ``number_of_merges``.
        number_of_merges: Exact number of merge operations. Mutually exclusive
            with ``ratio``.
        keep_root_edge: Preserve root's single child edge after pruning.
        collapse_duplicates: If ``True``, deduplicate repeated state tokens
            within a merged state string (e.g., ``"1|1|2"`` → ``"1|2"``).
        random_seed: NumPy random seed for reproducibility.
        tree_key: Key in ``tdata.obst`` for the tree.

    Returns:
        A new TreeData with merged leaves and combined character states.

    Raises:
        LeafSubsamplerError: On invalid parameters.
    """
    if (ratio is None) == (number_of_merges is None):
        raise LeafSubsamplerError(
            "Specify exactly one of `ratio` or `number_of_merges`."
        )

    if random_seed is not None:
        np.random.seed(random_seed)

    leaves = list(tdata.obs_names)
    n_merges = (
        number_of_merges if number_of_merges is not None else int(len(leaves) * ratio)
    )

    if n_merges >= len(leaves):
        raise LeafSubsamplerError(
            "Number of merges must be less than the number of leaves."
        )
    if n_merges <= 0:
        raise LeafSubsamplerError("Number of merges must be > 0.")

    working_tree = tdata.obst[tree_key].copy()
    current_leaves = list(leaves)

    # Snapshot obsm character data per leaf as string lists
    obsm_data: dict[str, dict[str, list[str]]] = {}
    obsm_cols: dict[str, list] = {}
    for key, val in tdata.obsm.items():
        if isinstance(val, pd.DataFrame):
            obsm_cols[key] = list(val.columns)
            obsm_data[key] = {leaf: list(val.loc[leaf].astype(str)) for leaf in leaves}

    for _ in range(n_merges):
        leaf1 = str(np.random.choice(current_leaves))
        other = [l for l in current_leaves if l != leaf1]

        distances = np.array(
            [_leaf_distance(working_tree, leaf1, l) for l in other]
        )
        weights = 1.0 / np.maximum(distances, 1e-10)
        probs = weights / weights.sum()
        leaf2 = str(np.random.choice(other, p=probs))

        lca = nx.lowest_common_ancestor(working_tree, leaf1, leaf2)
        new_time = (
            working_tree.nodes[leaf1]["time"] + working_tree.nodes[leaf2]["time"]
        ) / 2
        new_leaf = f"{leaf1}-{leaf2}"

        working_tree.add_node(new_leaf, time=new_time)
        working_tree.add_edge(lca, new_leaf)

        for key in obsm_data:
            s1 = obsm_data[key][leaf1]
            s2 = obsm_data[key][leaf2]
            obsm_data[key][new_leaf] = [
                _merge_state(a, b, collapse_duplicates) for a, b in zip(s1, s2)
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
        ).T

    return td.TreeData(
        obs=new_obs,
        obsm=new_obsm,
        obst={tree_key: new_tree},
        uns=dict(tdata.uns),
    )


# --- Private helpers ---


def _prune_tdata(
    tdata: td.TreeData,
    keep_leaves: list[str],
    keep_root_edge: bool,
    tree_key: str,
) -> td.TreeData:
    """Return new TreeData pruned to keep_leaves with unifurcations collapsed."""
    sub = tdata[list(keep_leaves)].copy()
    pruned = collapse_unifurcations(
        sub.obst[tree_key], collapse_root=not keep_root_edge
    )
    sub.obst[tree_key] = pruned
    return sub


def _leaf_distance(tree: nx.DiGraph, leaf1: str, leaf2: str) -> float:
    """Branch distance between two leaves via their LCA."""
    lca = nx.lowest_common_ancestor(tree, leaf1, leaf2)
    return (
        tree.nodes[leaf1]["time"]
        + tree.nodes[leaf2]["time"]
        - 2 * tree.nodes[lca]["time"]
    )


def _merge_state(s1: str, s2: str, collapse_duplicates: bool) -> str:
    """Merge two state strings with ``|`` separator."""
    parts = s1.split("|") + s2.split("|")
    if collapse_duplicates:
        seen: set[str] = set()
        unique: list[str] = []
        for x in parts:
            if x not in seen:
                seen.add(x)
                unique.append(x)
        parts = unique
    return "|".join(parts)


def _induce_and_collapse(
    tree: nx.DiGraph, final_leaves: list[str], keep_root_edge: bool
) -> nx.DiGraph:
    """Build induced subtree on final_leaves and collapse unifurcations."""
    ancestors: set[str] = set()
    for leaf in final_leaves:
        ancestors |= nx.ancestors(tree, leaf)
    induced = tree.subgraph(ancestors | set(final_leaves)).copy()
    return collapse_unifurcations(induced, collapse_root=not keep_root_edge)


def _build_merged_tree(
    orig_tree: nx.DiGraph,
    merge_map: dict[str, list[str]],
    final_leaves: list[str],
    keep_root_edge: bool,
) -> nx.DiGraph:
    """Add merged leaf nodes to tree, then induce subtree on final_leaves."""
    working = orig_tree.copy()
    for new_name, old_leaves in merge_map.items():
        lca = old_leaves[0]
        for other in old_leaves[1:]:
            lca = nx.lowest_common_ancestor(working, lca, other)
        new_time = float(np.mean([working.nodes[l]["time"] for l in old_leaves]))
        working.add_node(new_name, time=new_time)
        working.add_edge(lca, new_name)
    return _induce_and_collapse(working, final_leaves, keep_root_edge)


def _merge_obsm(
    obsm: dict,
    final_leaves: list[str],
    merge_map: dict[str, list[str]],
    spatial_key: str,
    pixel_of: dict[str, tuple],
    collapse_duplicates: bool,
) -> dict:
    """Build merged obsm for spatial merges."""
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
                    col_vals = [list(val.loc[l].astype(str)) for l in old_leaves]
                    merged_row = []
                    for group in zip(*col_vals):
                        merged_row.append(
                            _merge_state_multi(list(group), collapse_duplicates)
                        )
                    rows[leaf] = merged_row

        if rows:
            new_obsm[key] = pd.DataFrame(rows, index=cols).T[cols]

    return new_obsm


def _merge_state_multi(values: list[str], collapse_duplicates: bool) -> str:
    """Merge a list of state strings with ``|`` separator."""
    parts: list[str] = []
    for v in values:
        parts.extend(v.split("|"))
    if collapse_duplicates:
        seen: set[str] = set()
        unique: list[str] = []
        for x in parts:
            if x not in seen:
                seen.add(x)
                unique.append(x)
        parts = unique
    return "|".join(parts)
