"""Functionality for spatial imputation."""

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse
import tqdm
from treedata import TreeData

from cassiopeia.utils import _get_characters, _get_parameter, _normalize_missing


def _impute_single_state(
    cell: str,
    character: int,
    character_matrix: pd.DataFrame,
    neighborhood_graph: nx.DiGraph,
    missing_states: frozenset,
    number_of_hops: int = 1,
    max_neighbor_distance: float = np.inf,
    coordinates: pd.DataFrame | None = None,
) -> tuple[object, float, int]:
    """Imputes missing character state for a cell at a defined position.

    Args:
        cell: Cell barcode.
        character: Which character to impute.
        character_matrix: Character matrix of all character states.
        neighborhood_graph: Spatial graph connecting cells.
        missing_states: Set of values representing missing data (excluded from votes).
        number_of_hops: Number of hops to make during imputation.
        max_neighbor_distance: Maximum distance to neighbor to be used for
            imputation.
        coordinates: Coordinates of all cells.

    Returns:
        The state, the frequency of votes, and the absolute number of votes.
        Returns ``(None, 0, 0)`` when no eligible votes are found.
    """
    votes = []
    for _, node in nx.bfs_edges(neighborhood_graph, cell, depth_limit=number_of_hops):
        if node not in character_matrix.index:
            continue

        distance = 0
        if coordinates is not None:
            distance = np.sqrt(
                np.sum((coordinates.loc[cell].values - coordinates.loc[node].values) ** 2)
            )

        state = character_matrix.loc[node].iloc[character]
        if distance <= max_neighbor_distance and state not in missing_states:
            if isinstance(state, tuple):
                for _state in state:
                    votes.append(_state)
            else:
                votes.append(state)

    if len(votes) > 0:
        values, counts = np.unique(votes, return_counts=True)
        return (
            values[np.argmax(counts)],
            np.max(counts) / np.sum(counts),
            int(np.max(counts)),
        )

    return None, 0, 0


def _build_spatial_nx_graph(tdata: TreeData, connect_key: str) -> nx.Graph:
    """Converts a sparse spatial connectivity matrix to a labeled NetworkX graph.

    Args:
        tdata: TreeData with spatial connectivity in ``tdata.obsp[connect_key]``.
        connect_key: Key in ``tdata.obsp`` for the spatial connectivity matrix.

    Returns:
        A NetworkX graph with cell names as nodes.
    """
    if connect_key not in tdata.obsp:
        raise KeyError(
            f"connect_key '{connect_key}' not found in tdata.obsp. "
            f"Available keys: {list(tdata.obsp.keys())}."
        )
    adj = tdata.obsp[connect_key]
    if not scipy.sparse.issparse(adj):
        adj = scipy.sparse.csr_matrix(adj)
    graph = nx.from_scipy_sparse_array(adj)
    node_map = dict(zip(range(adj.shape[0]), tdata.obs_names, strict=False))
    return nx.relabel_nodes(graph, node_map)


def impute_alleles_spatial(
    tdata: TreeData,
    *,
    connect_key: str | None = None,
    characters_key: str = "characters",
    spatial_key: str | None = None,
    imputation_hops: int = 2,
    imputation_concordance: float = 0.8,
    num_imputation_iterations: int = 1,
    max_neighbor_distance: float = np.inf,
    missing_state=None,
    unmodified_state=None,
    key_added: str = "characters_imputed",
    copy: bool = False,
) -> TreeData | None:
    """Imputes missing alleles using spatial proximity.

    Iteratively imputes missing character states in ``tdata.obsm[characters_key]``
    using a spatial connectivity graph stored in ``tdata.obsp[connect_key]``. For
    each missing state a plurality vote is collected from spatial neighbors
    (within ``imputation_hops`` BFS hops); the imputation is accepted when the
    winning vote fraction meets ``imputation_concordance``. Unmodified states
    (e.g. 0) are not imputed even when they win the vote.

    Run ``squidpy.gr.spatial_neighbors`` on your AnnData, store the connectivity
    in ``tdata.obsp``, then call this function::

        import squidpy as sq

        sq.gr.spatial_neighbors(adata, key_added="spatial")
        tdata.obsp["spatial_connectivities"] = adata.obsp["spatial_connectivities"]
        cas.sp.impute_alleles_spatial(tdata, connect_key="spatial_connectivities")

    Args:
        tdata: TreeData object.
        connect_key: Key in ``tdata.obsp`` for the spatial connectivity graph.
            Required; raises ``ValueError`` if ``None``.
        characters_key: Key in ``tdata.obsm`` for the character matrix.
        spatial_key: Key in ``tdata.obsm`` for spatial coordinates. Required only
            when ``max_neighbor_distance < inf``.
        imputation_hops: Number of BFS hops used to gather neighbor votes.
        imputation_concordance: Minimum fraction of neighbor votes required to
            accept an imputation.
        num_imputation_iterations: Number of imputation rounds (enables
            multi-hop propagation of imputed values).
        max_neighbor_distance: Maximum Euclidean distance to a neighbor for it
            to contribute a vote. Requires ``spatial_key``.
        missing_state: Value(s) representing missing data. May be a scalar or a
            sequence. Defaults to ``tdata.uns["missing_state"]`` if present,
            otherwise ``(-1, "-1", "NA", "-")``.
        unmodified_state: Value(s) representing the unmodified (uncut) state.
            Imputation will not produce these values. Defaults to
            ``tdata.uns["unmodified_state"]`` if present, otherwise
            ``(0, "0", "*")``.
        key_added: Key in ``tdata.obsm`` under which the imputed character matrix
            is stored.
        copy: If ``True``, return a copy of ``tdata`` instead of modifying in-place.

    Returns:
        ``None`` when ``copy=False`` (modifies ``tdata`` in-place), or the modified
        copy when ``copy=True``.
    """
    if connect_key is None:
        raise ValueError(
            "connect_key must be provided. First compute a spatial connectivity graph "
            "with squidpy.gr.spatial_neighbors(adata, key_added=...) and store the "
            "result in tdata.obsp, then pass the key here."
        )

    if copy:
        tdata = tdata.copy()

    raw_missing = _get_parameter(tdata, "missing_state", missing_state)
    raw_unmodified = _get_parameter(tdata, "unmodified_state", unmodified_state)
    missing_states = frozenset(_normalize_missing(raw_missing))
    unmodified_states = frozenset(_normalize_missing(raw_unmodified))

    character_matrix = _get_characters(tdata, characters_key)
    spatial_graph = _build_spatial_nx_graph(tdata, connect_key)

    coordinates: pd.DataFrame | None = None
    if spatial_key is not None:
        coords_arr = tdata.obsm[spatial_key]
        if not isinstance(coords_arr, pd.DataFrame):
            coords_arr = pd.DataFrame(coords_arr, index=tdata.obs_names)
        coordinates = coords_arr

    prev = character_matrix.copy()
    for _round in range(num_imputation_iterations):
        print(f">> Imputation round {_round + 1}...")

        current = prev.copy()
        missing_mask = prev.isin(missing_states)
        missing_indices = np.where(missing_mask)

        for i, j in tqdm.tqdm(
            zip(missing_indices[0], missing_indices[1], strict=False),
            total=len(missing_indices[0]),
        ):
            imputed_value, proportion_of_votes, n_votes = _impute_single_state(
                prev.index.values[i],
                j,
                prev,
                neighborhood_graph=spatial_graph,
                missing_states=missing_states,
                number_of_hops=imputation_hops,
                max_neighbor_distance=max_neighbor_distance,
                coordinates=coordinates,
            )
            if (
                n_votes > 0
                and proportion_of_votes >= imputation_concordance
                and imputed_value not in missing_states
                and imputed_value not in unmodified_states
            ):
                current.iloc[i, j] = imputed_value

        prev = current

    tdata.obsm[key_added] = prev

    if copy:
        return tdata
