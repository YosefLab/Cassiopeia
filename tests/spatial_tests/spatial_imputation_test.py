"""Tests for the spatial imputation module."""

import warnings

import networkx as nx
import numpy as np
import pandas as pd
import pytest
from treedata import TreeData

import cassiopeia as cas
from cassiopeia.spatial.spatial_imputation import (
    _build_spatial_nx_graph,
    _impute_single_state,
)

# ---------------------------------------------------------------------------
# Private helper tests
# ---------------------------------------------------------------------------


def test_impute_single_state_basic(character_matrix, spatial_graph_neigh3):
    imputed_state, frequency, count = _impute_single_state(
        "cell_4",
        0,
        character_matrix,
        spatial_graph_neigh3,
        missing_states=frozenset({-1}),
        number_of_hops=1,
        max_neighbor_distance=np.inf,
    )
    assert imputed_state == 1
    assert frequency == pytest.approx(3 / 4)
    assert count == 3


def test_impute_single_state_max_neighbor_distance(
    character_matrix, spatial_graph_neigh3, coordinates, spatial_adata
):
    cell_names = character_matrix.index.tolist()
    coords_df = pd.DataFrame(coordinates, index=cell_names)

    # Without distance limit: majority across all neighbors
    imputed_state, frequency, count = _impute_single_state(
        "cell_5",
        0,
        character_matrix,
        spatial_graph_neigh3,
        missing_states=frozenset({-1}),
        number_of_hops=1,
        max_neighbor_distance=np.inf,
    )
    assert imputed_state == 2
    assert frequency == pytest.approx(2 / 3)
    assert count == 4

    # With distance limit: only nearby cells vote
    imputed_state, frequency, count = _impute_single_state(
        "cell_5",
        0,
        character_matrix,
        spatial_graph_neigh3,
        missing_states=frozenset({-1}),
        number_of_hops=1,
        max_neighbor_distance=15,
        coordinates=coords_df,
    )
    assert imputed_state == 1
    assert frequency == pytest.approx(1.0)
    assert count == 2


def test_build_spatial_nx_graph(spatial_tdata, spatial_graph_neigh3):
    graph = _build_spatial_nx_graph(spatial_tdata, "spatial_connectivities")
    assert set(graph.nodes) == set(spatial_graph_neigh3.nodes)


def test_build_spatial_nx_graph_missing_key(spatial_tdata):
    with pytest.raises(KeyError, match="bad_key"):
        _build_spatial_nx_graph(spatial_tdata, "bad_key")


# ---------------------------------------------------------------------------
# impute_alleles_spatial — error handling
# ---------------------------------------------------------------------------


def test_impute_alleles_spatial_no_connect_key(spatial_tdata):
    with pytest.raises(ValueError, match="connect_key"):
        cas.sp.impute_alleles_spatial(spatial_tdata, connect_key=None)


# ---------------------------------------------------------------------------
# impute_alleles_spatial — imputation behavior
# ---------------------------------------------------------------------------


def test_impute_one_hop_low_concordance(spatial_tdata):
    cas.sp.impute_alleles_spatial(
        spatial_tdata,
        connect_key="spatial_connectivities",
        imputation_hops=1,
        imputation_concordance=0.0,
        num_imputation_iterations=1,
        key_added="out",
    )
    result = spatial_tdata.obsm["out"]
    assert result.loc["cell_1", 0] == 1
    assert result.loc["cell_3", 1] == 1
    assert result.loc["cell_7", 2] == 5


def test_impute_one_hop_high_concordance(spatial_tdata):
    cas.sp.impute_alleles_spatial(
        spatial_tdata,
        connect_key="spatial_connectivities",
        imputation_hops=1,
        imputation_concordance=0.8,
        num_imputation_iterations=1,
        key_added="out",
    )
    result = spatial_tdata.obsm["out"]
    assert result.loc["cell_1", 0] == 1
    assert result.loc["cell_3", 1] == -1  # ambiguous — below threshold
    assert result.loc["cell_7", 2] == -1  # ambiguous — below threshold


def test_impute_two_hops(character_matrix_missing, spatial_graph_neigh3, coordinates):
    cm = character_matrix_missing.copy()
    cm.loc["cell_2", 1] = 2
    cm.loc["cell_5", 2] = 5

    cell_names = cm.index.tolist()
    tdata = TreeData(obs=pd.DataFrame(index=cell_names))
    tdata.obsm["characters"] = cm
    tdata.obsp["spatial_connectivities"] = nx.to_scipy_sparse_array(
        spatial_graph_neigh3, nodelist=cell_names
    )
    tdata.uns["missing_state"] = -1
    tdata.uns["unmodified_state"] = 0

    cas.sp.impute_alleles_spatial(
        tdata,
        connect_key="spatial_connectivities",
        imputation_hops=2,
        imputation_concordance=0.0,
        num_imputation_iterations=1,
        key_added="out",
    )
    result = tdata.obsm["out"]
    assert result.loc["cell_1", 0] == 1
    assert result.loc["cell_3", 1] == 2
    assert result.loc["cell_7", 2] == 5


def test_impute_no_zero(character_matrix_missing, spatial_graph_neigh3):
    cm = character_matrix_missing.copy()
    cm.loc["cell_0", 1] = 0
    cm.loc["cell_4", 1] = 0

    cell_names = cm.index.tolist()
    tdata = TreeData(obs=pd.DataFrame(index=cell_names))
    tdata.obsm["characters"] = cm
    tdata.obsp["spatial_connectivities"] = nx.to_scipy_sparse_array(
        spatial_graph_neigh3, nodelist=cell_names
    )
    tdata.uns["missing_state"] = -1
    tdata.uns["unmodified_state"] = 0

    cas.sp.impute_alleles_spatial(
        tdata,
        connect_key="spatial_connectivities",
        imputation_hops=1,
        imputation_concordance=0.0,
        num_imputation_iterations=1,
        key_added="out",
    )
    # majority is 0 (unmodified) — should not be imputed
    assert tdata.obsm["out"].loc["cell_3", 1] == -1


def test_impute_two_iterations(character_matrix_missing, spatial_graph_neigh3):
    g = spatial_graph_neigh3.copy()
    g.add_edge("cell_7", "cell_9")

    cm = character_matrix_missing.copy()
    cm.loc["cell_9"] = [-1, -1, -1, -1]

    def _make_tdata(mat):
        td = TreeData(obs=pd.DataFrame(index=mat.index.tolist()))
        td.obsm["characters"] = mat.copy()
        td.obsp["spatial_connectivities"] = nx.to_scipy_sparse_array(g, nodelist=mat.index.tolist())
        td.uns["missing_state"] = -1
        td.uns["unmodified_state"] = 0
        return td

    # One iteration: cell_9 neighbors are still missing — nothing imputed
    tdata1 = _make_tdata(cm)
    cas.sp.impute_alleles_spatial(
        tdata1,
        connect_key="spatial_connectivities",
        imputation_hops=1,
        imputation_concordance=0.0,
        num_imputation_iterations=1,
        key_added="out",
    )
    assert tdata1.obsm["out"].loc["cell_9", 2] == -1

    # Two iterations: cell_7 gets imputed in round 1, cell_9 in round 2
    tdata2 = _make_tdata(cm)
    cas.sp.impute_alleles_spatial(
        tdata2,
        connect_key="spatial_connectivities",
        imputation_hops=1,
        imputation_concordance=0.0,
        num_imputation_iterations=2,
        key_added="out",
    )
    assert tdata2.obsm["out"].loc["cell_9", 2] == 5


def test_impute_max_neighbor_distance(character_matrix_missing, spatial_graph_neigh3, coordinates):
    cm = character_matrix_missing.copy()
    cm.loc["cell_5", 0] = -1
    cell_names = cm.index.tolist()

    coords_df = pd.DataFrame(coordinates, index=cell_names)

    def _make_tdata():
        td = TreeData(obs=pd.DataFrame(index=cell_names))
        td.obsm["characters"] = cm.copy()
        td.obsp["spatial_connectivities"] = nx.to_scipy_sparse_array(
            spatial_graph_neigh3, nodelist=cell_names
        )
        td.obsm["spatial"] = coords_df
        td.uns["missing_state"] = -1
        td.uns["unmodified_state"] = 0
        return td

    # Within 15 units: only cell_4 close enough → votes 1
    tdata1 = _make_tdata()
    cas.sp.impute_alleles_spatial(
        tdata1,
        connect_key="spatial_connectivities",
        spatial_key="spatial",
        imputation_hops=1,
        imputation_concordance=0.0,
        num_imputation_iterations=1,
        max_neighbor_distance=15,
        key_added="out",
    )
    assert tdata1.obsm["out"].loc["cell_5", 0] == 1

    # No distance limit: more distant cell_7/cell_8 (state=2) can vote
    tdata2 = _make_tdata()
    cas.sp.impute_alleles_spatial(
        tdata2,
        connect_key="spatial_connectivities",
        imputation_hops=1,
        imputation_concordance=0.0,
        num_imputation_iterations=1,
        max_neighbor_distance=np.inf,
        key_added="out",
    )
    assert tdata2.obsm["out"].loc["cell_5", 0] == 2


def test_impute_copy_does_not_modify_original(spatial_tdata):
    original_chars = spatial_tdata.obsm["characters"].copy()
    result = cas.sp.impute_alleles_spatial(
        spatial_tdata,
        connect_key="spatial_connectivities",
        imputation_hops=1,
        imputation_concordance=0.0,
        num_imputation_iterations=1,
        key_added="characters_imputed",
        copy=True,
        unmodified_state=0,
    )
    assert result is not spatial_tdata
    assert "characters_imputed" not in spatial_tdata.obsm
    pd.testing.assert_frame_equal(spatial_tdata.obsm["characters"], original_chars)


# ---------------------------------------------------------------------------
# squidpy-dependent tests
# ---------------------------------------------------------------------------


def test_impute_squidpy_radius(spatial_adata, character_matrix_missing):
    sq = pytest.importorskip("squidpy")

    sq.gr.spatial_neighbors(
        spatial_adata,
        coord_type="generic",
        spatial_key="spatial",
        radius=15.0,
        key_added="spatial",
    )
    cell_names = character_matrix_missing.index.tolist()
    tdata = TreeData(obs=pd.DataFrame(index=cell_names))
    tdata.obsm["characters"] = character_matrix_missing.copy()
    tdata.obsp["spatial_connectivities"] = spatial_adata.obsp["spatial_connectivities"]

    cm2 = character_matrix_missing.copy()
    cm2.loc["cell_8", 2] = 11
    tdata.obsm["characters"] = cm2

    cas.sp.impute_alleles_spatial(
        tdata,
        connect_key="spatial_connectivities",
        imputation_hops=1,
        imputation_concordance=0.6,
        num_imputation_iterations=1,
        key_added="out",
        unmodified_state=0,
        missing_state=-1,
    )
    result = tdata.obsm["out"]
    assert result.loc["cell_1", 0] == 1
    assert result.loc["cell_3", 1] == 1
    assert result.loc["cell_7", 2] == -1  # cell_8 disagreement lowers concordance


def test_impute_squidpy_size(spatial_adata, character_matrix_missing):
    sq = pytest.importorskip("squidpy")

    sq.gr.spatial_neighbors(
        spatial_adata,
        coord_type="generic",
        spatial_key="spatial",
        n_neighs=3,
        key_added="spatial",
    )
    cm = character_matrix_missing.copy()
    cm.loc["cell_5", 1] = -1
    cm.loc["cell_6", 1] = 1

    cell_names = cm.index.tolist()
    tdata = TreeData(obs=pd.DataFrame(index=cell_names))
    tdata.obsm["characters"] = cm
    tdata.obsp["spatial_connectivities"] = spatial_adata.obsp["spatial_connectivities"]

    cas.sp.impute_alleles_spatial(
        tdata,
        connect_key="spatial_connectivities",
        imputation_hops=1,
        imputation_concordance=0.0,
        num_imputation_iterations=1,
        key_added="out",
        unmodified_state=0,
        missing_state=-1,
    )
    assert tdata.obsm["out"].loc["cell_5", 1] == 1


# ---------------------------------------------------------------------------
# Deprecation shim
# ---------------------------------------------------------------------------


def test_deprecated_shim_warns_and_returns_dataframe(
    character_matrix_missing, spatial_graph_neigh3
):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = cas.sp.impute_alleles_from_spatial_data(
            character_matrix_missing,
            spatial_graph=spatial_graph_neigh3,
            imputation_hops=1,
            imputation_concordance=0.0,
            num_imputation_iterations=1,
        )
    assert any(issubclass(warning.category, DeprecationWarning) for warning in w)
    assert isinstance(result, pd.DataFrame)
    assert result.loc["cell_1", 0] == 1
    assert result.loc["cell_3", 1] == 1
    assert result.loc["cell_7", 2] == 5


def test_deprecated_shim_squidpy(spatial_adata, character_matrix_missing):
    pytest.importorskip("squidpy")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = cas.sp.impute_alleles_from_spatial_data(
            character_matrix_missing,
            adata=spatial_adata,
            imputation_hops=1,
            imputation_concordance=0.0,
            num_imputation_iterations=1,
            neighborhood_radius=30.0,
            unmodified_state=0,
        )
    assert any(issubclass(warning.category, DeprecationWarning) for warning in w)
    assert isinstance(result, pd.DataFrame)


def test_deprecated_shim_no_graph_raises(character_matrix_missing):
    with pytest.raises(ValueError, match="spatial_graph.*adata"):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            cas.sp.impute_alleles_from_spatial_data(character_matrix_missing)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
