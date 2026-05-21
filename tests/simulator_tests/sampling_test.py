"""Tests for sample_uniform, sample_spatial, sample_supercellular."""

import inspect

import networkx as nx
import numpy as np
import pandas as pd
import pytest
import treedata as td

from cassiopeia.mixins import LeafSubsamplerError, LeafSubsamplerWarning
from cassiopeia.simulator import (
    LeafSubsampler,
    SpatialLeafSubsampler,
    SupercellularSampler,
    UniformLeafSubsampler,
    sample_spatial,
    sample_supercellular,
    sample_uniform,
)


# --- Shared fixture helpers ---


def _make_tdata(n_leaves: int = 8, with_chars: bool = True, seed: int = 42) -> td.TreeData:
    """Build a deterministic TreeData with an ultrametric binary tree."""
    np.random.seed(seed)
    tree = nx.balanced_tree(2, int(np.log2(n_leaves)), create_using=nx.DiGraph)
    tree = nx.relabel_nodes(tree, {i: str(i) for i in tree.nodes})
    root = "0"
    leaves = [n for n in tree if tree.out_degree(n) == 0]

    for node in tree.nodes:
        depth = nx.shortest_path_length(tree, root, node)
        tree.nodes[node]["time"] = float(depth)

    obs = pd.DataFrame(index=leaves)
    tdata = td.TreeData(obs=obs, obst={"tree": tree})

    if with_chars:
        char_data = {}
        for i, leaf in enumerate(leaves):
            char_data[leaf] = [str(i % 3 + 1), str((i + 1) % 3 + 1), "*"]
        tdata.obsm["characters"] = pd.DataFrame(char_data, index=["c1", "c2", "c3"]).T
        tdata.uns["cassette_size"] = 3

    return tdata


def _make_spatial_tdata(n_leaves: int = 8, seed: int = 42) -> td.TreeData:
    """Build TreeData with spatial coordinates."""
    tdata = _make_tdata(n_leaves=n_leaves, with_chars=True, seed=seed)
    leaves = list(tdata.obs_names)
    coords = np.array([[float(i), float(i % 4)] for i in range(n_leaves)])
    tdata.obsm["spatial"] = pd.DataFrame(coords, index=leaves, columns=["x", "y"])
    return tdata


# ============================================================
# sample_uniform
# ============================================================


def test_uniform_bad_params_both():
    tdata = _make_tdata()
    with pytest.raises(LeafSubsamplerError):
        sample_uniform(tdata, ratio=0.5, number_of_leaves=4)


def test_uniform_bad_params_neither():
    tdata = _make_tdata()
    with pytest.raises(LeafSubsamplerError):
        sample_uniform(tdata)


def test_uniform_bad_n_too_large():
    tdata = _make_tdata(n_leaves=4)
    with pytest.raises(LeafSubsamplerError):
        sample_uniform(tdata, number_of_leaves=100)


def test_uniform_bad_n_zero():
    tdata = _make_tdata(n_leaves=4)
    with pytest.raises(LeafSubsamplerError):
        sample_uniform(tdata, number_of_leaves=0)


def test_uniform_bad_ratio_zero():
    tdata = _make_tdata(n_leaves=4)
    with pytest.raises(LeafSubsamplerError):
        sample_uniform(tdata, ratio=0.001)


def test_uniform_number_of_leaves():
    tdata = _make_tdata(n_leaves=8)
    sub = sample_uniform(tdata, number_of_leaves=4, random_seed=1)
    assert len(sub.obs_names) == 4
    assert set(sub.obs_names) <= set(tdata.obs_names)


def test_uniform_ratio():
    tdata = _make_tdata(n_leaves=8)
    sub = sample_uniform(tdata, ratio=0.5, random_seed=1)
    assert len(sub.obs_names) == 4


def test_uniform_reproducibility():
    tdata = _make_tdata(n_leaves=8)
    sub1 = sample_uniform(tdata, number_of_leaves=4, random_seed=7)
    sub2 = sample_uniform(tdata, number_of_leaves=4, random_seed=7)
    assert list(sub1.obs_names) == list(sub2.obs_names)


def test_uniform_obsm_preserved():
    tdata = _make_tdata(n_leaves=8, with_chars=True)
    sub = sample_uniform(tdata, number_of_leaves=4, random_seed=1)
    assert "characters" in sub.obsm
    assert sub.obsm["characters"].shape == (4, 3)
    for leaf in sub.obs_names:
        assert tdata.obsm["characters"].loc[leaf].tolist() == sub.obsm["characters"].loc[leaf].tolist()


def test_uniform_uns_preserved():
    tdata = _make_tdata(n_leaves=8, with_chars=True)
    sub = sample_uniform(tdata, number_of_leaves=4, random_seed=1)
    assert sub.uns["cassette_size"] == 3


def test_uniform_tree_valid():
    tdata = _make_tdata(n_leaves=8)
    sub = sample_uniform(tdata, number_of_leaves=4, random_seed=5)
    tree = sub.obst["tree"]
    assert len([n for n in tree if tree.out_degree(n) == 0]) == 4
    root = next(n for n in tree if tree.in_degree(n) == 0)
    for node in tree:
        if tree.out_degree(node) != 0 and node != root:
            assert tree.out_degree(node) >= 2


def test_uniform_all_leaves():
    tdata = _make_tdata(n_leaves=4)
    sub = sample_uniform(tdata, number_of_leaves=4, random_seed=1)
    assert set(sub.obs_names) == set(tdata.obs_names)


# ============================================================
# sample_spatial
# ============================================================


def test_spatial_bad_params_both_region():
    tdata = _make_spatial_tdata()
    space = np.ones((10, 10), dtype=bool)
    with pytest.raises(LeafSubsamplerError):
        sample_spatial(tdata, bounding_box=[(0, 5), (0, 5)], space=space)


def test_spatial_bad_params_no_region():
    tdata = _make_spatial_tdata()
    with pytest.raises(LeafSubsamplerError):
        sample_spatial(tdata)


def test_spatial_missing_spatial_key():
    tdata = _make_tdata()
    with pytest.raises(LeafSubsamplerError):
        sample_spatial(tdata, bounding_box=[(0, 10), (0, 10)])


def test_spatial_bad_spatial_dims():
    tdata = _make_spatial_tdata()
    space_3d = np.ones((10, 10, 10), dtype=bool)
    with pytest.raises(LeafSubsamplerError):
        sample_spatial(tdata, space=space_3d)


def test_spatial_no_leaves_in_region():
    tdata = _make_spatial_tdata()
    with pytest.raises(LeafSubsamplerError):
        sample_spatial(tdata, bounding_box=[(100, 200), (100, 200)])


def test_spatial_bounding_box():
    tdata = _make_spatial_tdata(n_leaves=8)
    sub = sample_spatial(tdata, bounding_box=[(0, 3), (0, 10)])
    for leaf in sub.obs_names:
        assert 0 <= tdata.obsm["spatial"].loc[leaf, "x"] <= 3


def test_spatial_space_mask():
    tdata = _make_spatial_tdata(n_leaves=8)
    space = np.zeros((10, 10), dtype=bool)
    space[0:4, :] = True
    sub = sample_spatial(tdata, space=space)
    for leaf in sub.obs_names:
        x = int(tdata.obsm["spatial"].loc[leaf, "x"])
        y = int(tdata.obsm["spatial"].loc[leaf, "y"])
        assert space[x, y]


def test_spatial_obsm_preserved():
    tdata = _make_spatial_tdata(n_leaves=8)
    sub = sample_spatial(tdata, bounding_box=[(0, 3), (0, 10)])
    assert "characters" in sub.obsm
    assert "spatial" in sub.obsm
    assert sub.obsm["characters"].shape[1] == 3


def test_spatial_key_param():
    tdata = _make_tdata(n_leaves=4)
    leaves = list(tdata.obs_names)
    tdata.obsm["coords"] = pd.DataFrame(
        [[0, 0], [1, 0], [2, 0], [3, 0]], index=leaves, columns=["x", "y"]
    )
    sub = sample_spatial(tdata, bounding_box=[(0, 1), (0, 10)], spatial_key="coords")
    assert len(sub.obs_names) == 2


def test_spatial_warn_bad_scale():
    tdata = _make_spatial_tdata(n_leaves=4)
    leaves = list(tdata.obs_names)
    tdata.obsm["spatial"] = pd.DataFrame(
        [[0.1, 0.1], [0.1, 0.2], [0.2, 0.1], [0.2, 0.2]],
        index=leaves, columns=["x", "y"],
    )
    space = np.zeros((1000, 1000), dtype=bool)
    space[:100, :100] = True
    with pytest.warns(LeafSubsamplerWarning):
        sample_spatial(tdata, space=space)


def test_spatial_no_removed_params():
    sig = inspect.signature(sample_spatial)
    assert "merge_cells" not in sig.parameters
    assert "number_of_leaves" not in sig.parameters
    assert "ratio" not in sig.parameters
    assert "random_seed" not in sig.parameters


def test_spatial_composed_pixel_merge():
    """sample_spatial → sample_supercellular(spatial_key=...) pipeline."""
    tdata = _make_tdata(n_leaves=4, with_chars=False)
    leaves = list(tdata.obs_names)
    tdata.obsm["characters"] = pd.DataFrame({"c1": ["1", "2", "3", "4"]}, index=leaves)
    tdata.obsm["spatial"] = pd.DataFrame(
        [[1, 1], [1, 1], [2, 2], [3, 3]], index=leaves, columns=["x", "y"]
    )
    space = np.ones((5, 5), dtype=bool)
    filtered = sample_spatial(tdata, space=space)
    merged = sample_supercellular(filtered, spatial_key="spatial")
    assert len(merged.obs_names) == 3
    pixel_merged = [l for l in merged.obs_names if "-" in l]
    assert len(pixel_merged) == 1


# ============================================================
# sample_supercellular — iterative mode
# ============================================================


def test_supercellular_bad_params_both():
    tdata = _make_tdata()
    with pytest.raises(LeafSubsamplerError):
        sample_supercellular(tdata, ratio=0.5, number_of_merges=2)


def test_supercellular_bad_params_neither():
    tdata = _make_tdata()
    with pytest.raises(LeafSubsamplerError):
        sample_supercellular(tdata)


def test_supercellular_too_many_merges():
    tdata = _make_tdata(n_leaves=4)
    with pytest.raises(LeafSubsamplerError):
        sample_supercellular(tdata, number_of_merges=4)


def test_supercellular_zero_merges():
    tdata = _make_tdata(n_leaves=4)
    with pytest.raises(LeafSubsamplerError):
        sample_supercellular(tdata, number_of_merges=0)


def test_supercellular_basic_merge_count():
    tdata = _make_tdata(n_leaves=8)
    sub = sample_supercellular(tdata, number_of_merges=3, random_seed=10)
    assert len(sub.obs_names) == 5


def test_supercellular_ratio():
    tdata = _make_tdata(n_leaves=8)
    sub = sample_supercellular(tdata, ratio=0.5, random_seed=10)
    assert len(sub.obs_names) == 4


def test_supercellular_reproducibility():
    tdata = _make_tdata(n_leaves=8)
    sub1 = sample_supercellular(tdata, number_of_merges=3, random_seed=42)
    sub2 = sample_supercellular(tdata, number_of_merges=3, random_seed=42)
    assert set(sub1.obs_names) == set(sub2.obs_names)


def test_supercellular_merged_leaves_named_correctly():
    tdata = _make_tdata(n_leaves=8)
    sub = sample_supercellular(tdata, number_of_merges=2, random_seed=7)
    assert any("-" in l for l in sub.obs_names)


def test_supercellular_characters_preserved_for_unmerged():
    tdata = _make_tdata(n_leaves=8, with_chars=True)
    sub = sample_supercellular(tdata, number_of_merges=1, random_seed=5)
    for leaf in sub.obs_names:
        if "-" not in leaf:
            assert tdata.obsm["characters"].loc[leaf].tolist() == sub.obsm["characters"].loc[leaf].tolist()


def test_supercellular_merged_states_contain_pipe():
    tdata = _make_tdata(n_leaves=4, with_chars=False)
    leaves = list(tdata.obs_names)
    tdata.obsm["characters"] = pd.DataFrame({"c1": ["1", "2", "3", "4"]}, index=leaves)
    sub = sample_supercellular(tdata, number_of_merges=1, random_seed=1)
    merged = [l for l in sub.obs_names if "-" in l]
    assert "|" in sub.obsm["characters"].loc[merged[0], "c1"]


def test_supercellular_collapse_duplicates():
    tdata = _make_tdata(n_leaves=4, with_chars=False)
    leaves = list(tdata.obs_names)
    tdata.obsm["characters"] = pd.DataFrame({"c1": ["1", "1", "3", "4"]}, index=leaves)
    sub = sample_supercellular(tdata, number_of_merges=1, collapse_duplicates=True, random_seed=1)
    for leaf in sub.obs_names:
        parts = sub.obsm["characters"].loc[leaf, "c1"].split("|")
        assert len(parts) == len(set(parts))


def test_supercellular_tree_valid():
    tdata = _make_tdata(n_leaves=8)
    sub = sample_supercellular(tdata, number_of_merges=3, random_seed=1)
    tree = sub.obst["tree"]
    assert {n for n in tree if tree.out_degree(n) == 0} == set(sub.obs_names)
    root = next(n for n in tree if tree.in_degree(n) == 0)
    for node in tree:
        if tree.out_degree(node) != 0 and node != root:
            assert tree.out_degree(node) >= 2


def test_supercellular_merged_time_is_mean():
    tdata = _make_tdata(n_leaves=4)
    orig_tree = tdata.obst["tree"]
    orig_times = {n: orig_tree.nodes[n]["time"] for n in orig_tree.nodes}
    sub = sample_supercellular(tdata, number_of_merges=1, random_seed=3)
    tree = sub.obst["tree"]
    for merged in [n for n in sub.obs_names if "-" in n]:
        parts = merged.split("-")
        if all(p in orig_times for p in parts) and merged in tree.nodes:
            expected = np.mean([orig_times[p] for p in parts])
            assert np.isclose(tree.nodes[merged]["time"], expected)


def test_supercellular_inverse_distance_weighting():
    """Per pair, siblings (dist 2) should be merged more than second-cousins (dist 6)."""
    tdata = _make_tdata(n_leaves=8)
    orig_tree = tdata.obst["tree"]
    leaves = list(tdata.obs_names)
    pair_counts: dict[tuple, int] = {}

    for trial in range(500):
        sub = sample_supercellular(tdata, number_of_merges=1, random_seed=trial)
        merged = [l for l in sub.obs_names if "-" in l]
        if not merged:
            continue
        parts = merged[0].split("-")
        if len(parts) == 2 and all(p in leaves for p in parts):
            pair = tuple(sorted(parts))
            pair_counts[pair] = pair_counts.get(pair, 0) + 1

    if not pair_counts:
        return

    sibling_rates, second_cousin_rates = [], []
    for (l1, l2), count in pair_counts.items():
        lca = nx.lowest_common_ancestor(orig_tree, l1, l2)
        d = orig_tree.nodes[l1]["time"] + orig_tree.nodes[l2]["time"] - 2 * orig_tree.nodes[lca]["time"]
        if d <= 2:
            sibling_rates.append(count)
        elif d >= 6:
            second_cousin_rates.append(count)

    if sibling_rates and second_cousin_rates:
        assert np.mean(sibling_rates) > np.mean(second_cousin_rates)


def test_supercellular_obsm_shape():
    tdata = _make_tdata(n_leaves=8, with_chars=True)
    sub = sample_supercellular(tdata, number_of_merges=2, random_seed=1)
    assert sub.obsm["characters"].shape == (6, 3)


def test_supercellular_uns_preserved():
    tdata = _make_tdata(n_leaves=8, with_chars=True)
    sub = sample_supercellular(tdata, number_of_merges=2, random_seed=1)
    assert sub.uns.get("cassette_size") == 3


# ============================================================
# sample_supercellular — pixel mode
# ============================================================


def test_pixel_mode_bad_params_with_ratio():
    tdata = _make_spatial_tdata(n_leaves=4)
    with pytest.raises(LeafSubsamplerError):
        sample_supercellular(tdata, spatial_key="spatial", ratio=0.5)


def test_pixel_mode_bad_params_with_n_merges():
    tdata = _make_spatial_tdata(n_leaves=4)
    with pytest.raises(LeafSubsamplerError):
        sample_supercellular(tdata, spatial_key="spatial", number_of_merges=1)


def test_pixel_mode_missing_spatial_key():
    tdata = _make_tdata(n_leaves=4)
    with pytest.raises(LeafSubsamplerError):
        sample_supercellular(tdata, spatial_key="spatial")


def test_pixel_mode_no_overlap():
    """All leaves at distinct pixels → no merges."""
    tdata = _make_tdata(n_leaves=4, with_chars=False)
    leaves = list(tdata.obs_names)
    tdata.obsm["spatial"] = pd.DataFrame(
        [[0, 0], [1, 0], [2, 0], [3, 0]], index=leaves, columns=["x", "y"]
    )
    sub = sample_supercellular(tdata, spatial_key="spatial")
    assert set(sub.obs_names) == set(leaves)
    assert not any("-" in l for l in sub.obs_names)


def test_pixel_mode_with_overlap():
    """Leaves sharing a pixel are merged into one observation."""
    tdata = _make_tdata(n_leaves=4, with_chars=False)
    leaves = list(tdata.obs_names)
    tdata.obsm["spatial"] = pd.DataFrame(
        [[2, 2], [2, 2], [3, 3], [4, 4]], index=leaves, columns=["x", "y"]
    )
    sub = sample_supercellular(tdata, spatial_key="spatial")
    assert len(sub.obs_names) == 3
    merged = [l for l in sub.obs_names if "-" in l]
    assert len(merged) == 1
    assert sub.obsm["spatial"].loc[merged[0]].tolist() == [2, 2]


def test_pixel_mode_characters_combined():
    tdata = _make_tdata(n_leaves=4, with_chars=False)
    leaves = list(tdata.obs_names)
    tdata.obsm["characters"] = pd.DataFrame({"c1": ["1", "2", "3", "4"]}, index=leaves)
    tdata.obsm["spatial"] = pd.DataFrame(
        [[1, 1], [1, 1], [2, 2], [3, 3]], index=leaves, columns=["x", "y"]
    )
    sub = sample_supercellular(tdata, spatial_key="spatial")
    merged = [l for l in sub.obs_names if "-" in l]
    merged_c1 = sub.obsm["characters"].loc[merged[0], "c1"]
    assert set(merged_c1.split("|")) == {"1", "2"}


def test_pixel_mode_collapse_duplicates():
    tdata = _make_tdata(n_leaves=4, with_chars=False)
    leaves = list(tdata.obs_names)
    tdata.obsm["characters"] = pd.DataFrame({"c1": ["1", "1", "3", "4"]}, index=leaves)
    tdata.obsm["spatial"] = pd.DataFrame(
        [[1, 1], [1, 1], [2, 2], [3, 3]], index=leaves, columns=["x", "y"]
    )
    sub = sample_supercellular(tdata, spatial_key="spatial", collapse_duplicates=True)
    merged = [l for l in sub.obs_names if "-" in l]
    assert sub.obsm["characters"].loc[merged[0], "c1"] == "1"


def test_pixel_mode_no_collapse_duplicates():
    tdata = _make_tdata(n_leaves=4, with_chars=False)
    leaves = list(tdata.obs_names)
    tdata.obsm["characters"] = pd.DataFrame({"c1": ["1", "1", "3", "4"]}, index=leaves)
    tdata.obsm["spatial"] = pd.DataFrame(
        [[1, 1], [1, 1], [2, 2], [3, 3]], index=leaves, columns=["x", "y"]
    )
    sub = sample_supercellular(tdata, spatial_key="spatial", collapse_duplicates=False)
    merged = [l for l in sub.obs_names if "-" in l]
    assert sub.obsm["characters"].loc[merged[0], "c1"] == "1|1"


def test_pixel_mode_tree_valid():
    tdata = _make_tdata(n_leaves=4)
    leaves = list(tdata.obs_names)
    tdata.obsm["spatial"] = pd.DataFrame(
        [[1, 1], [1, 1], [2, 2], [3, 3]], index=leaves, columns=["x", "y"]
    )
    sub = sample_supercellular(tdata, spatial_key="spatial")
    tree = sub.obst["tree"]
    assert {n for n in tree if tree.out_degree(n) == 0} == set(sub.obs_names)


def test_pixel_mode_composed_after_spatial():
    """Compose sample_spatial (filter) with sample_supercellular (pixel merge)."""
    tdata = _make_tdata(n_leaves=8, with_chars=False)
    leaves = list(tdata.obs_names)
    tdata.obsm["spatial"] = pd.DataFrame(
        [[i, i % 3] for i in range(8)], index=leaves, columns=["x", "y"]
    )
    tdata.obsm["spatial"].iloc[0] = [5, 5]
    tdata.obsm["spatial"].iloc[1] = [5, 5]
    tdata.obsm["spatial"].iloc[2] = [5, 5]
    tdata.obsm["spatial"].iloc[3] = [5, 5]

    space = np.ones((10, 10), dtype=bool)
    filtered = sample_spatial(tdata, space=space)
    merged = sample_supercellular(filtered, spatial_key="spatial")
    assert len(merged.obs_names) == 5


# ============================================================
# Deprecation stubs
# ============================================================


def test_leaf_subsampler_deprecated():
    with pytest.warns(DeprecationWarning, match="sample_uniform"):
        sampler = LeafSubsampler()
    with pytest.raises(NotImplementedError):
        sampler.subsample_leaves(None)


def test_uniform_leaf_subsampler_deprecated():
    with pytest.warns(DeprecationWarning, match="sample_uniform"):
        sampler = UniformLeafSubsampler()
    with pytest.raises(NotImplementedError):
        sampler.subsample_leaves(None)


def test_spatial_leaf_subsampler_deprecated():
    with pytest.warns(DeprecationWarning, match="sample_spatial"):
        sampler = SpatialLeafSubsampler()
    with pytest.raises(NotImplementedError):
        sampler.subsample_leaves(None)


def test_supercellular_sampler_deprecated():
    with pytest.warns(DeprecationWarning, match="sample_supercellular"):
        sampler = SupercellularSampler()
    with pytest.raises(NotImplementedError):
        sampler.subsample_leaves(None)
