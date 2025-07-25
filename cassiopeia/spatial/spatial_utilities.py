"""
Utilities for spatial lineage-tracing module.
"""

from typing import Tuple

import anndata
import networkx as nx
import numpy as np
import pandas as pd
import squidpy as sq


def get_spatial_graph_from_anndata(
    adata: anndata.AnnData,
    neighborhood_radius: int = 30.0,
    neighborhood_size: float | None = None,
) -> nx.DiGraph:
    """Get a spatial graph structure from an spatial anndata

    Construct a spatial graph connecting each node to its nearest neighbors in
    space. Assumes that the specified adata has spatial coordinates specified in
    the `.obsm` key.

    Args:
        adata: Anndata of spatially-resolved data. Only the spatial coordinates
            need to be stored, and this is used to construct a graph.
        neighborhood_size: If a connectivitity graph is being constructed,
            this is the number of nearest neighbors to connect to a node.
        neighborhood_radius: Intead of passing in `neighborhood_size`, this
            is the radius of the connectivity graph.

    Returns:
        A networkx object storing the spatial graph.
    """
    # create spatial graph if needed
    if neighborhood_size:
        sq.gr.spatial_neighbors(
            adata,
            coord_type="generic",
            spatial_key="spatial",
            n_neighs=neighborhood_size,
        )

    else:
        sq.gr.spatial_neighbors(
            adata,
            coord_type="generic",
            spatial_key="spatial",
            radius=neighborhood_radius,
        )

    spatial_graph = nx.from_numpy_array(adata.obsp["spatial_connectivities"])

    return spatial_graph


def impute_single_state(
    cell: str,
    character: int,
    character_matrix: pd.DataFrame,
    neighborhood_graph: nx.DiGraph = None,
    number_of_hops: int = 1,
    max_neighbor_distance: float = np.inf,
    coordinates: pd.DataFrame | None = None,
) -> Tuple[int, float, int]:
    """Imputes missing character state for a cell at a defined position.

    Args:
        cell: Cell barcode
        character: Which character to impute.
        character_matrix: Character matrix of all character states
        adata: Anndata object with spatial nearest neighbors
        number_of_hops: Number of hops to make during imputation.
        max_neighbor_distance: Maximum distance to neighbor to be used for
            imputation.
    Returns:
        The state, the frequency of votes, and the absolute number of votes
    """

    votes = []
    for _, node in nx.bfs_edges(
        neighborhood_graph, cell, depth_limit=number_of_hops
    ):
        if node not in character_matrix.index:
            continue

        distance = 0
        if not (coordinates is None):
            distance = np.sqrt(
                np.sum(
                    (
                        coordinates.loc[cell].values
                        - coordinates.loc[node].values
                    )
                    ** 2
                )
            )

        state = character_matrix.loc[node].iloc[character]
        if distance <= max_neighbor_distance and state != -1:
            if type(state) == tuple:
                for _state in state:
                    votes.append(_state)
            else:
                votes.append(state)

    if len(votes) > 0:
        values, counts = np.unique(votes, return_counts=True)
        return (
            values[np.argmax(counts)],
            np.max(counts) / np.sum(counts),
            np.max(counts),
        )

    return -1, 0, 0
