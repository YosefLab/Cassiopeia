"""Functionality for spatial imputation."""

import anndata
import networkx as nx
import numpy as np
import pandas as pd
import tqdm

from cassiopeia.spatial import spatial_utilities


def impute_alleles_from_spatial_data(
    character_matrix: pd.DataFrame,
    adata: anndata.AnnData | None = None,
    spatial_graph: nx.Graph | None = None,
    neighborhood_size: int | None = None,
    neighborhood_radius: float = 30.0,
    imputation_hops: int = 2,
    imputation_concordance: float = 0.8,
    num_imputation_iterations: int = 1,
    max_neighbor_distance: float = np.inf,
    coordinates: pd.DataFrame | None = None,
    connect_key: str = "spatial_connectivities"
) -> pd.DataFrame:
    """Imputes data based on spatial location.

    This procedure iterates over spots in a given anndata and imputes missing
    data based on spatial neigborhoods.

    The procedure is an interative algorithm where for each iteration a missing
    character is imputed based on the neighborhood of a node. Node neighborhoods
    are built off of a spatial graph that can passed in directly or inferred
    from a specified spatial anndata. In the simplest example, node neighobrhoods
    are just the immediate neighbors of a node, but users can also include
    neighbors-of-neighbors (and neighbors-of-neighbors-of-neighbors, so on)
    using the `imputation_hops` argument. An imputation will be accepted if the
    fraction of neighbors that agree with an observed non-zero allele exceeds
    the specified `imputation_concordance` threshold (by default 0.8). This
    procedure can be repeated for several rounds, controlled by the
    `num_imputation_iterations` argument and in this way approximates a
    message-passing process.

    Args:
        character_matrix: A character matrix of spots, constructed using a
            function like `convert_allele_table_to_character_matrix`.
        adata: Anndata of spatially-resolved data. Only the spatial coordinates
            need to be stored, and this is used to construct a graph.
        spatial_graph: Optionally, the user can provide a spatial connectivity
            graph instead of passing in an adata.
        neighborhood_size: If a connectivitity graph is being constructed,
            this is the number of nearest neighbors to connect to a node. If
            both neighborhood_size and neighborhood_radius are passed in,
            neighborhood_size is preferred.
        neighborhood_radius: Intead of passing in `neighborhood_size`, this
            is the radius of the connectivity graph.
        imputation_hops: Number of adjacent node's adjacencies to query. For
            example, if this is 2, this means that imputation is done not just
            on nearest neighbors of a given node, but also each nearest
            neighbor's nearest neighbors.
        imputation_concordance: Fraction of votes that must agree in order
            to accept an imputation.
        num_imputation_iterations: Number of iterations for imputation
            procedure.
        max_neighbor_distance: Maximum distance to neighbor to be used for
            imputation.
        coordinates: If an AnnData is not specified, and you wish to set an
            upper limit on the distance for spatial imputation, these
            coordinates can be passed to the imputation procedure.
        connect_key: Key used to store spatial connectivities in
            `adata.obsp`. This will be passed into the `key_added` argument
            of sq.gr.spatial_neighbors and an etnry in `adata.obsp` will be added
            of the form `{connect_key}_connectivities`.

    Returns
    -------
        An imputed character matrix.
    """
    if (not spatial_graph) and (not adata):
        raise Exception(
            "One of the following must be specified: "
            "`spatial_graph` or `adata`."
        )

    if not spatial_graph:

        # create spatial graph if needed
        spatial_graph = spatial_utilities.get_spatial_graph_from_anndata(
            adata,
            neighborhood_radius=neighborhood_radius,
            neighborhood_size=neighborhood_size,
            connect_key=connect_key,
        )

    prev_character_matrix_imputed = character_matrix.copy()
    missing_indices = np.where(character_matrix == -1)

    for _round in range(num_imputation_iterations):

        print(f">> Imputation round {_round+1}...")

        character_matrix_imputed = prev_character_matrix_imputed.copy()
        missing_indices = np.where(prev_character_matrix_imputed == -1)

        for i, j in tqdm.tqdm(
            zip(missing_indices[0], missing_indices[1], strict=False),
            total=len(missing_indices[0]),
        ):
            (imputed_value, proportion_of_votes, number_of_votes) = (
                spatial_utilities.impute_single_state(
                    prev_character_matrix_imputed.index.values[i],
                    j,
                    prev_character_matrix_imputed,
                    neighborhood_graph=spatial_graph,
                    number_of_hops=imputation_hops,
                    max_neighbor_distance=max_neighbor_distance,
                    coordinates=coordinates,
                )
            )
            if (
                proportion_of_votes >= imputation_concordance
                and imputed_value != -1
                and imputed_value != 0
            ):
                character_matrix_imputed.iloc[i, j] = int(imputed_value)

        prev_character_matrix_imputed = character_matrix_imputed.copy()

    # apply final missingness filter
    final_character_matrix = character_matrix_imputed

    return final_character_matrix
