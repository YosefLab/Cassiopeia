"""Shared fixtures for spatial tests."""

import networkx as nx
import numpy as np
import pandas as pd
import pytest
from treedata import TreeData


@pytest.fixture
def coordinates() -> np.ndarray:
    return np.array(
        [
            [0.0, 0.0],
            [0.0, 10],
            [0.0, 50],
            [10, 0.0],
            [10, 50],
            [15, 50],
            [90, 100],
            [100, 100],
            [100, 90],
        ]
    )


@pytest.fixture
def spatial_adata(coordinates):
    import anndata

    adata = anndata.AnnData(obs=pd.DataFrame(index=[f"cell_{x}" for x in range(len(coordinates))]))
    adata.obsm["spatial"] = coordinates
    return adata


@pytest.fixture
def spatial_graph_neigh3():
    g = nx.Graph()
    for edge in [
        ("cell_0", "cell_1"),
        ("cell_0", "cell_2"),
        ("cell_0", "cell_3"),
        ("cell_1", "cell_2"),
        ("cell_1", "cell_3"),
        ("cell_1", "cell_4"),
        ("cell_1", "cell_5"),
        ("cell_2", "cell_4"),
        ("cell_2", "cell_5"),
        ("cell_3", "cell_4"),
        ("cell_4", "cell_5"),
        ("cell_5", "cell_6"),
        ("cell_5", "cell_7"),
        ("cell_5", "cell_8"),
        ("cell_6", "cell_7"),
        ("cell_6", "cell_8"),
        ("cell_7", "cell_8"),
    ]:
        g.add_edge(*edge)
    return g


@pytest.fixture
def character_matrix():
    return pd.DataFrame.from_dict(
        {
            "cell_0": [1, 1, 0, 1],
            "cell_1": [2, 1, 0, 2],
            "cell_2": [1, 1, 2, 3],
            "cell_3": [1, 1, 2, 4],
            "cell_4": [1, 2, 3, 5],
            "cell_5": [1, 2, 3, 6],
            "cell_6": [2, 0, 5, 8],
            "cell_7": [2, 0, 5, 9],
            "cell_8": [2, 0, 5, 10],
        },
        orient="index",
    )


@pytest.fixture
def character_matrix_missing():
    return pd.DataFrame.from_dict(
        {
            "cell_0": [1, 1, 0, 1],
            "cell_1": [-1, 1, 0, 2],
            "cell_2": [1, 1, 2, 3],
            "cell_3": [1, -1, 2, 4],
            "cell_4": [1, 2, 3, 5],
            "cell_5": [1, 2, 3, 6],
            "cell_6": [2, 0, 5, 8],
            "cell_7": [2, 0, -1, 9],
            "cell_8": [2, 0, 5, 10],
        },
        orient="index",
    )


@pytest.fixture
def spatial_tdata(character_matrix_missing, spatial_graph_neigh3, coordinates):
    """TreeData with characters in obsm and spatial connectivity in obsp."""
    cell_names = character_matrix_missing.index.tolist()
    tdata = TreeData(obs=pd.DataFrame(index=cell_names))
    tdata.obsm["characters"] = character_matrix_missing.copy()
    adj = nx.to_scipy_sparse_array(spatial_graph_neigh3, nodelist=cell_names)
    tdata.obsp["spatial_connectivities"] = adj
    tdata.obsm["spatial"] = pd.DataFrame(coordinates, index=cell_names)
    tdata.uns["missing_state"] = -1
    tdata.uns["unmodified_state"] = 0
    return tdata
