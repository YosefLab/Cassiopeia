"""Deprecated spatial API shims."""

import warnings

import networkx as nx
import numpy as np
import pandas as pd

from cassiopeia.mixins import try_import


def impute_alleles_from_spatial_data(
    character_matrix: pd.DataFrame,
    adata=None,
    spatial_graph: nx.Graph | None = None,
    neighborhood_size: int | None = None,
    neighborhood_radius: float = 30.0,
    imputation_hops: int = 2,
    imputation_concordance: float = 0.8,
    num_imputation_iterations: int = 1,
    max_neighbor_distance: float = np.inf,
    coordinates: pd.DataFrame | None = None,
    connect_key: str = "spatial_connectivities",
    missing_state=None,
    unmodified_state=None,
) -> pd.DataFrame:
    """Deprecated alias for :func:`cassiopeia.sp.impute_alleles_spatial`.

    .. deprecated::
        Use :func:`cassiopeia.sp.impute_alleles_spatial` instead. Store spatial
        connectivity in ``tdata.obsp`` (e.g. via ``squidpy.gr.spatial_neighbors``)
        and pass the character matrix in ``tdata.obsm``.
    """
    warnings.warn(
        "impute_alleles_from_spatial_data() is deprecated and will be removed in a "
        "future release; use impute_alleles_spatial() instead. See the docstring of "
        "impute_alleles_spatial() for migration instructions.",
        DeprecationWarning,
        stacklevel=2,
    )

    from treedata import TreeData

    from cassiopeia.spatial.spatial_imputation import impute_alleles_spatial

    tdata = TreeData(obs=pd.DataFrame(index=character_matrix.index))
    tdata.obsm["characters"] = character_matrix.copy()

    _connect_key = "spatial_connectivities"

    if spatial_graph is not None:
        adj = nx.to_scipy_sparse_array(spatial_graph, nodelist=character_matrix.index.tolist())
        tdata.obsp[_connect_key] = adj
    elif adata is not None:
        sq = try_import("squidpy")
        if sq is None:
            raise ImportError(
                "squidpy is required when adata is provided. Install it with: pip install squidpy"
            )
        if neighborhood_size:
            sq.gr.spatial_neighbors(
                adata,
                coord_type="generic",
                spatial_key="spatial",
                n_neighs=neighborhood_size,
                key_added=connect_key,
            )
        else:
            sq.gr.spatial_neighbors(
                adata,
                coord_type="generic",
                spatial_key="spatial",
                radius=neighborhood_radius,
                key_added=connect_key,
            )
        tdata.obsp[_connect_key] = adata.obsp[f"{connect_key}_connectivities"]
    else:
        raise ValueError("One of `spatial_graph` or `adata` must be provided.")

    _spatial_key = None
    if coordinates is not None:
        tdata.obsm["spatial"] = coordinates
        _spatial_key = "spatial"
    elif adata is not None and "spatial" in adata.obsm:
        tdata.obsm["spatial"] = adata.obsm["spatial"]
        _spatial_key = "spatial"

    impute_alleles_spatial(
        tdata,
        connect_key=_connect_key,
        characters_key="characters",
        spatial_key=_spatial_key,
        imputation_hops=imputation_hops,
        imputation_concordance=imputation_concordance,
        num_imputation_iterations=num_imputation_iterations,
        max_neighbor_distance=max_neighbor_distance,
        missing_state=missing_state,
        unmodified_state=unmodified_state,
        key_added="characters_imputed",
    )
    return tdata.obsm["characters_imputed"]
