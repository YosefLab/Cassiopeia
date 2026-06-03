"""Tests for brownian_spatial() and clonal_spatial()."""

import networkx as nx
import numpy as np
import pandas as pd
import pytest
import treedata as td

from cassiopeia.mixins import DataSimulatorError
from cassiopeia.simulator import (
    BrownianSpatialDataSimulator,
    ClonalSpatialDataSimulator,
    brownian_spatial,
    clonal_spatial,
)

# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------


def _make_tdata(n_leaves=8, seed=42) -> td.TreeData:
    """Build a deterministic TreeData with an ultrametric balanced binary tree.

    Balanced binary tree with depth=log2(n_leaves). Nodes labeled as strings.
    All times are integer depths (branch length = 1 per level).
    """
    np.random.seed(seed)
    depth = int(np.log2(n_leaves))
    tree = nx.balanced_tree(2, depth, create_using=nx.DiGraph)
    tree = nx.relabel_nodes(tree, {i: str(i) for i in tree.nodes})
    root = "0"
    for node in tree.nodes:
        d = nx.shortest_path_length(tree, root, node)
        tree.nodes[node]["time"] = float(d)

    leaves = [n for n in tree if tree.out_degree(n) == 0]
    obs = pd.DataFrame(index=leaves)
    return td.TreeData(obs=obs, obst={"simulated": tree})


# ---------------------------------------------------------------------------
# brownian_spatial — validation
# ---------------------------------------------------------------------------


def test_brownian_bad_dim():
    tdata = _make_tdata()
    with pytest.raises(DataSimulatorError):
        brownian_spatial(tdata, dim=0, diffusion_coefficient=1.0)


def test_brownian_bad_diffusion_coef():
    tdata = _make_tdata()
    with pytest.raises(DataSimulatorError):
        brownian_spatial(tdata, dim=2, diffusion_coefficient=-1.0)


# ---------------------------------------------------------------------------
# brownian_spatial — output shape and content
# ---------------------------------------------------------------------------


def test_brownian_obsm_shape():
    n_leaves = 8
    tdata = _make_tdata(n_leaves=n_leaves)
    brownian_spatial(tdata, dim=2, diffusion_coefficient=1.0, random_seed=0)
    assert tdata.obsm["spatial"].shape == (n_leaves, 2)


def test_brownian_obsm_columns():
    tdata = _make_tdata()
    brownian_spatial(tdata, dim=2, diffusion_coefficient=1.0, random_seed=0)
    assert tdata.obsm["spatial"].shape[1] == 2


def test_brownian_leaf_coords_valid():
    tdata = _make_tdata()
    brownian_spatial(tdata, dim=2, diffusion_coefficient=1.0, random_seed=0)
    assert tdata.obsm["spatial"].shape[0] == len(tdata.obs_names)


def test_brownian_scale_unit_area():
    tdata = _make_tdata()
    brownian_spatial(tdata, dim=2, diffusion_coefficient=1.0, scale_unit_area=True, random_seed=0)
    coords = tdata.obsm["spatial"]
    assert coords.min() >= 0.0
    assert coords.max() <= 1.0


def test_brownian_no_scale_root_origin():
    tdata = _make_tdata()
    brownian_spatial(tdata, dim=2, diffusion_coefficient=1.0, scale_unit_area=False, random_seed=0)
    tree = tdata.obst["simulated"]
    root = next(n for n in tree if tree.in_degree(n) == 0)
    np.testing.assert_array_equal(tree.nodes[root]["spatial"], np.zeros(2))


def test_brownian_reproducibility():
    tdata1 = _make_tdata()
    tdata2 = _make_tdata()
    brownian_spatial(tdata1, dim=2, diffusion_coefficient=1.0, random_seed=7)
    brownian_spatial(tdata2, dim=2, diffusion_coefficient=1.0, random_seed=7)
    np.testing.assert_array_equal(tdata1.obsm["spatial"], tdata2.obsm["spatial"])


def test_brownian_different_seeds():
    tdata1 = _make_tdata()
    tdata2 = _make_tdata()
    brownian_spatial(tdata1, dim=2, diffusion_coefficient=1.0, random_seed=1)
    brownian_spatial(tdata2, dim=2, diffusion_coefficient=1.0, random_seed=2)
    assert not np.array_equal(tdata1.obsm["spatial"], tdata2.obsm["spatial"])


def test_brownian_node_attrs():
    tdata = _make_tdata()
    brownian_spatial(tdata, dim=2, diffusion_coefficient=1.0, random_seed=0)
    tree = tdata.obst["simulated"]
    for node in tree.nodes:
        assert "spatial" in tree.nodes[node]
        assert len(tree.nodes[node]["spatial"]) == 2


def test_brownian_dim_3():
    n_leaves = 8
    tdata = _make_tdata(n_leaves=n_leaves)
    brownian_spatial(tdata, dim=3, diffusion_coefficient=1.0, random_seed=0)
    assert tdata.obsm["spatial"].shape == (n_leaves, 3)


def test_brownian_key_added():
    tdata = _make_tdata()
    brownian_spatial(tdata, dim=2, diffusion_coefficient=1.0, random_seed=0, key_added="coords")
    assert "coords" in tdata.obsm
    assert "spatial" not in tdata.obsm
    tree = tdata.obst["simulated"]
    root = next(n for n in tree if tree.in_degree(n) == 0)
    assert "coords" in tree.nodes[root]


def test_brownian_zero_diffusion():
    tdata = _make_tdata()
    brownian_spatial(tdata, dim=2, diffusion_coefficient=0.0, scale_unit_area=True, random_seed=0)
    # All nodes should have the same coordinate, so after scaling all should equal 0
    coords = tdata.obsm["spatial"]
    assert np.all(coords == coords[0])


# ---------------------------------------------------------------------------
# clonal_spatial — validation
# ---------------------------------------------------------------------------


def test_clonal_bad_params_both():
    tdata = _make_tdata()
    space = np.ones((100, 100), dtype=bool)
    with pytest.raises(DataSimulatorError):
        clonal_spatial(tdata, shape=(100, 100), space=space)


def test_clonal_bad_params_neither():
    tdata = _make_tdata()
    with pytest.raises(DataSimulatorError):
        clonal_spatial(tdata, shape=None, space=None)


def test_clonal_obsm_shape():
    n_leaves = 8
    tdata = _make_tdata(n_leaves=n_leaves)
    clonal_spatial(tdata, shape=(100, 100), random_seed=0)
    assert tdata.obsm["spatial"].shape == (n_leaves, 2)


def test_clonal_obsm_columns():
    tdata = _make_tdata()
    clonal_spatial(tdata, shape=(100, 100), random_seed=0)
    assert tdata.obsm["spatial"].shape[1] == 2


def test_clonal_coords_in_space():
    shape = (100, 100)
    center_x = shape[1] // 2
    center_y = shape[0] // 2
    y, x = np.ogrid[: shape[0], : shape[1]]
    space = ((x - center_x) / center_x) ** 2 + ((y - center_y) / center_y) ** 2 <= 1
    tdata = _make_tdata()
    clonal_spatial(tdata, shape=shape, random_seed=0)
    coords = tdata.obsm["spatial"]
    for row in coords:
        xi, yi = int(row[0]), int(row[1])
        assert space[xi, yi], f"Coordinate ({xi}, {yi}) outside space"


def test_clonal_reproducibility():
    tdata1 = _make_tdata()
    tdata2 = _make_tdata()
    clonal_spatial(tdata1, shape=(100, 100), random_seed=5)
    clonal_spatial(tdata2, shape=(100, 100), random_seed=5)
    np.testing.assert_array_equal(tdata1.obsm["spatial"], tdata2.obsm["spatial"])


def test_clonal_space_param():
    space = np.ones((100, 100), dtype=bool)
    tdata = _make_tdata()
    clonal_spatial(tdata, space=space, random_seed=0)
    assert "spatial" in tdata.obsm
    assert tdata.obsm["spatial"].shape == (8, 2)


def test_clonal_node_attrs():
    tdata = _make_tdata()
    clonal_spatial(tdata, shape=(100, 100), random_seed=0)
    tree = tdata.obst["simulated"]
    internal_nodes = [n for n in tree if tree.out_degree(n) > 0]
    for node in internal_nodes:
        assert "spatial" in tree.nodes[node], f"Node {node} missing spatial attribute"


def test_clonal_spatial_autocorrelation():
    n_leaves = 8
    tdata = _make_tdata(n_leaves=n_leaves)
    tree = tdata.obst["simulated"]

    # Collect sibling pairs: leaves that share the same immediate parent
    leaves = [n for n in tree if tree.out_degree(n) == 0]
    sibling_pairs = []
    non_sibling_pairs = []
    for leaf in leaves:
        parent = next(iter(tree.predecessors(leaf)))
        siblings = [n for n in tree.successors(parent) if n != leaf and tree.out_degree(n) == 0]
        for sib in siblings:
            pair = tuple(sorted([leaf, sib]))
            if pair not in sibling_pairs:
                sibling_pairs.append(pair)

    for i, l1 in enumerate(leaves):
        for l2 in leaves[i + 1 :]:
            pair = tuple(sorted([l1, l2]))
            if pair not in sibling_pairs:
                non_sibling_pairs.append(pair)

    sibling_dists_across_seeds = []
    non_sibling_dists_across_seeds = []

    for seed in range(10):
        td_test = _make_tdata(n_leaves=n_leaves)
        clonal_spatial(td_test, shape=(200, 200), random_seed=seed)
        coords = td_test.obsm["spatial"]
        obs_names = list(td_test.obs_names)

        sibling_d = [
            np.linalg.norm(coords[obs_names.index(a)] - coords[obs_names.index(b)])
            for a, b in sibling_pairs
        ]
        non_sibling_d = [
            np.linalg.norm(coords[obs_names.index(a)] - coords[obs_names.index(b)])
            for a, b in non_sibling_pairs
        ]
        sibling_dists_across_seeds.extend(sibling_d)
        non_sibling_dists_across_seeds.extend(non_sibling_d)

    assert np.mean(sibling_dists_across_seeds) < np.mean(non_sibling_dists_across_seeds), (
        f"Expected sibling distances ({np.mean(sibling_dists_across_seeds):.2f}) < "
        f"non-sibling distances ({np.mean(non_sibling_dists_across_seeds):.2f})"
    )


def test_clonal_key_added():
    tdata = _make_tdata()
    clonal_spatial(tdata, shape=(100, 100), random_seed=0, key_added="mycoords")
    assert "mycoords" in tdata.obsm
    assert "spatial" not in tdata.obsm
    tree = tdata.obst["simulated"]
    root = next(n for n in tree if tree.in_degree(n) == 0)
    assert "mycoords" in tree.nodes[root]


# ---------------------------------------------------------------------------
# Deprecation stubs
# ---------------------------------------------------------------------------


def test_brownian_spatial_data_simulator_deprecated():
    with pytest.warns(DeprecationWarning, match="brownian_spatial"):
        sim = BrownianSpatialDataSimulator()
    with pytest.raises(NotImplementedError):
        sim.overlay_data(None)


def test_clonal_spatial_data_simulator_deprecated():
    with pytest.warns(DeprecationWarning, match="clonal_spatial"):
        sim = ClonalSpatialDataSimulator()
    with pytest.raises(NotImplementedError):
        sim.overlay_data(None)


# ---------------------------------------------------------------------------
# copy parameter
# ---------------------------------------------------------------------------


def test_brownian_copy_false_modifies_inplace():
    tdata = _make_tdata()
    result = brownian_spatial(tdata, dim=2, diffusion_coefficient=1.0, random_seed=0, copy=False)
    assert result is None
    assert "spatial" in tdata.obsm


def test_brownian_copy_true_returns_new():
    tdata = _make_tdata()
    result = brownian_spatial(tdata, dim=2, diffusion_coefficient=1.0, random_seed=0, copy=True)
    assert result is not tdata
    assert "spatial" not in tdata.obsm
    assert "spatial" in result.obsm


def test_clonal_copy_false_modifies_inplace():
    tdata = _make_tdata()
    result = clonal_spatial(tdata, shape=(100, 100), random_seed=0, copy=False)
    assert result is None
    assert "spatial" in tdata.obsm


def test_clonal_copy_true_returns_new():
    tdata = _make_tdata()
    result = clonal_spatial(tdata, shape=(100, 100), random_seed=0, copy=True)
    assert result is not tdata
    assert "spatial" not in tdata.obsm
    assert "spatial" in result.obsm


if __name__ == "__main__":
    pytest.main(["-v", __file__])
