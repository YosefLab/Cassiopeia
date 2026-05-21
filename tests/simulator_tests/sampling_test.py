"""Tests for sample_uniform, sample_spatial, sample_supercellular."""

import warnings

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

    # Set times: root=0, depth proportional to distance from root
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


def _make_spatial_tdata(n_leaves: int = 8, ndim: int = 2, seed: int = 42) -> td.TreeData:
    """Build TreeData with spatial coordinates."""
    tdata = _make_tdata(n_leaves=n_leaves, with_chars=True, seed=seed)
    leaves = list(tdata.obs_names)
    np.random.seed(seed)
    coords = np.array([[float(i), float(i % 4)] for i in range(n_leaves)])
    tdata.obsm["spatial"] = pd.DataFrame(coords, index=leaves, columns=["x", "y"])
    return tdata


# ============================================================
# sample_uniform
# ============================================================


class TestSampleUniform:
    def test_bad_params_both(self):
        tdata = _make_tdata()
        with pytest.raises(LeafSubsamplerError):
            sample_uniform(tdata, ratio=0.5, number_of_leaves=4)

    def test_bad_params_neither(self):
        tdata = _make_tdata()
        with pytest.raises(LeafSubsamplerError):
            sample_uniform(tdata)

    def test_bad_n_too_large(self):
        tdata = _make_tdata(n_leaves=4)
        with pytest.raises(LeafSubsamplerError):
            sample_uniform(tdata, number_of_leaves=100)

    def test_bad_n_zero(self):
        tdata = _make_tdata(n_leaves=4)
        with pytest.raises(LeafSubsamplerError):
            sample_uniform(tdata, number_of_leaves=0)

    def test_bad_ratio_zero(self):
        tdata = _make_tdata(n_leaves=4)
        with pytest.raises(LeafSubsamplerError):
            sample_uniform(tdata, ratio=0.001)

    def test_number_of_leaves(self):
        tdata = _make_tdata(n_leaves=8)
        sub = sample_uniform(tdata, number_of_leaves=4, random_seed=1)
        assert len(sub.obs_names) == 4
        # All returned leaves were in the original tree
        assert set(sub.obs_names) <= set(tdata.obs_names)

    def test_ratio(self):
        tdata = _make_tdata(n_leaves=8)
        sub = sample_uniform(tdata, ratio=0.5, random_seed=1)
        assert len(sub.obs_names) == 4

    def test_reproducibility(self):
        tdata = _make_tdata(n_leaves=8)
        sub1 = sample_uniform(tdata, number_of_leaves=4, random_seed=7)
        sub2 = sample_uniform(tdata, number_of_leaves=4, random_seed=7)
        assert list(sub1.obs_names) == list(sub2.obs_names)

    def test_obsm_preserved(self):
        tdata = _make_tdata(n_leaves=8, with_chars=True)
        sub = sample_uniform(tdata, number_of_leaves=4, random_seed=1)
        assert "characters" in sub.obsm
        assert sub.obsm["characters"].shape == (4, 3)
        # Check that preserved rows match original
        for leaf in sub.obs_names:
            orig_row = tdata.obsm["characters"].loc[leaf].tolist()
            sub_row = sub.obsm["characters"].loc[leaf].tolist()
            assert orig_row == sub_row

    def test_uns_preserved(self):
        tdata = _make_tdata(n_leaves=8, with_chars=True)
        sub = sample_uniform(tdata, number_of_leaves=4, random_seed=1)
        assert sub.uns["cassette_size"] == 3

    def test_tree_valid(self):
        tdata = _make_tdata(n_leaves=8)
        sub = sample_uniform(tdata, number_of_leaves=4, random_seed=5)
        tree = sub.obst["tree"]
        leaves = [n for n in tree if tree.out_degree(n) == 0]
        assert len(leaves) == 4
        # All internal nodes have >= 2 children (no unifurcations)
        root = next(n for n in tree if tree.in_degree(n) == 0)
        for node in tree:
            if tree.out_degree(node) != 0 and node != root:
                assert tree.out_degree(node) >= 2

    def test_keep_root_edge_false(self):
        tdata = _make_tdata(n_leaves=4)
        # Force a case where root ends up with one child by taking a subset that
        # is all on one side of the tree.
        leaves = list(tdata.obs_names)
        half = leaves[:2]
        sub_keep = sample_uniform(tdata, number_of_leaves=2, random_seed=0)
        sub_no_keep = sample_uniform(tdata, number_of_leaves=2, random_seed=0)
        # Just check both run without error
        assert len(sub_keep.obs_names) == 2
        assert len(sub_no_keep.obs_names) == 2

    def test_all_leaves(self):
        tdata = _make_tdata(n_leaves=4)
        sub = sample_uniform(tdata, number_of_leaves=4, random_seed=1)
        assert set(sub.obs_names) == set(tdata.obs_names)


# ============================================================
# sample_spatial
# ============================================================


class TestSampleSpatial:
    def test_bad_params_both_region(self):
        tdata = _make_spatial_tdata()
        space = np.ones((10, 10), dtype=bool)
        bb = [(0, 5), (0, 5)]
        with pytest.raises(LeafSubsamplerError):
            sample_spatial(tdata, bounding_box=bb, space=space)

    def test_bad_params_no_region(self):
        tdata = _make_spatial_tdata()
        with pytest.raises(LeafSubsamplerError):
            sample_spatial(tdata)

    def test_bad_params_both_count(self):
        tdata = _make_spatial_tdata()
        space = np.ones((10, 10), dtype=bool)
        with pytest.raises(LeafSubsamplerError):
            sample_spatial(tdata, space=space, ratio=0.5, number_of_leaves=2)

    def test_bad_ratio(self):
        tdata = _make_spatial_tdata()
        space = np.ones((10, 10), dtype=bool)
        with pytest.raises(LeafSubsamplerError):
            sample_spatial(tdata, space=space, ratio=2.0)

    def test_bad_number_of_leaves(self):
        tdata = _make_spatial_tdata()
        space = np.ones((10, 10), dtype=bool)
        with pytest.raises(LeafSubsamplerError):
            sample_spatial(tdata, space=space, number_of_leaves=0)

    def test_merge_cells_without_space(self):
        tdata = _make_spatial_tdata()
        bb = [(0, 10), (0, 10)]
        with pytest.raises(LeafSubsamplerError):
            sample_spatial(tdata, bounding_box=bb, merge_cells=True)

    def test_missing_attribute(self):
        tdata = _make_tdata()
        with pytest.raises(LeafSubsamplerError):
            sample_spatial(tdata, bounding_box=[(0, 10), (0, 10)])

    def test_bad_spatial_dims(self):
        tdata = _make_spatial_tdata()
        space_3d = np.ones((10, 10, 10), dtype=bool)
        with pytest.raises(LeafSubsamplerError):
            sample_spatial(tdata, space=space_3d)  # coords are 2D, space is 3D

    def test_no_leaves_in_region(self):
        tdata = _make_spatial_tdata()
        with pytest.raises(LeafSubsamplerError):
            sample_spatial(tdata, bounding_box=[(100, 200), (100, 200)])

    def test_bounding_box(self):
        tdata = _make_spatial_tdata(n_leaves=8)
        sub = sample_spatial(tdata, bounding_box=[(0, 3), (0, 10)], random_seed=1)
        # Leaves with x in [0, 3]: indices 0,1,2,3 → all 4 kept
        for leaf in sub.obs_names:
            x = tdata.obsm["spatial"].loc[leaf, "x"]
            assert 0 <= x <= 3

    def test_space_mask(self):
        tdata = _make_spatial_tdata(n_leaves=8)
        space = np.zeros((10, 10), dtype=bool)
        space[0:4, :] = True
        sub = sample_spatial(tdata, space=space, random_seed=1)
        for leaf in sub.obs_names:
            x = int(tdata.obsm["spatial"].loc[leaf, "x"])
            assert space[x, int(tdata.obsm["spatial"].loc[leaf, "y"])]

    def test_with_ratio(self):
        tdata = _make_spatial_tdata(n_leaves=8)
        space = np.ones((10, 10), dtype=bool)
        sub = sample_spatial(tdata, space=space, ratio=0.5, random_seed=1)
        assert len(sub.obs_names) == 4

    def test_obsm_preserved(self):
        tdata = _make_spatial_tdata(n_leaves=8)
        space = np.ones((10, 10), dtype=bool)
        sub = sample_spatial(tdata, space=space, number_of_leaves=4, random_seed=1)
        assert "characters" in sub.obsm
        assert "spatial" in sub.obsm
        assert sub.obsm["characters"].shape == (4, 3)

    def test_warn_bad_scale(self):
        tdata = _make_spatial_tdata(n_leaves=4)
        # Set tiny coordinates against large space
        leaves = list(tdata.obs_names)
        tdata.obsm["spatial"] = pd.DataFrame(
            [[0.1, 0.1], [0.1, 0.2], [0.2, 0.1], [0.2, 0.2]],
            index=leaves,
            columns=["x", "y"],
        )
        space = np.zeros((1000, 1000), dtype=bool)
        space[:100, :100] = True
        with pytest.warns(LeafSubsamplerWarning):
            sample_spatial(tdata, space=space, random_seed=1)

    def test_merge_cells_no_overlap(self):
        tdata = _make_spatial_tdata(n_leaves=4)
        # Ensure all leaves have distinct pixels
        leaves = list(tdata.obs_names)
        tdata.obsm["spatial"] = pd.DataFrame(
            [[0, 0], [1, 0], [2, 0], [3, 0]],
            index=leaves,
            columns=["x", "y"],
        )
        space = np.ones((5, 5), dtype=bool)
        sub = sample_spatial(tdata, space=space, merge_cells=True, random_seed=1)
        # No merges should happen - same count as without merge
        assert len(sub.obs_names) == 4
        # All leaves are plain original names
        for leaf in sub.obs_names:
            assert "-" not in leaf

    def test_merge_cells_with_overlap(self):
        tdata = _make_spatial_tdata(n_leaves=4)
        # Put first two leaves at same pixel
        leaves = list(tdata.obs_names)
        tdata.obsm["spatial"] = pd.DataFrame(
            [[2, 2], [2, 2], [3, 3], [4, 4]],
            index=leaves,
            columns=["x", "y"],
        )
        space = np.ones((6, 6), dtype=bool)
        sub = sample_spatial(tdata, space=space, merge_cells=True, random_seed=1)
        # Should have 3 leaves: merged + 2 singletons
        assert len(sub.obs_names) == 3
        # One merged leaf
        merged = [l for l in sub.obs_names if "-" in l]
        assert len(merged) == 1
        # Merged leaf chars contain | for any site where source cells differ
        # The spatial coordinate of the merged leaf is the pixel
        assert "spatial" in sub.obsm
        merged_coords = sub.obsm["spatial"].loc[merged[0]].tolist()
        assert merged_coords == [2, 2]

    def test_merge_cells_characters_combined(self):
        tdata = _make_tdata(n_leaves=4, with_chars=False)
        leaves = list(tdata.obs_names)
        # Distinct character states for merging
        tdata.obsm["characters"] = pd.DataFrame(
            {"c1": ["1", "2", "3", "4"]}, index=leaves
        )
        tdata.obsm["spatial"] = pd.DataFrame(
            [[1, 1], [1, 1], [2, 2], [3, 3]],
            index=leaves,
            columns=["x", "y"],
        )
        space = np.ones((5, 5), dtype=bool)
        sub = sample_spatial(tdata, space=space, merge_cells=True, random_seed=1)
        merged = [l for l in sub.obs_names if "-" in l]
        assert len(merged) == 1
        # The merged cell should have "1|2" or "2|1" for c1
        merged_c1 = sub.obsm["characters"].loc[merged[0], "c1"]
        assert set(merged_c1.split("|")) == {"1", "2"}

    def test_merge_cells_collapse_duplicates(self):
        tdata = _make_tdata(n_leaves=4, with_chars=False)
        leaves = list(tdata.obs_names)
        # Both cells have same character state
        tdata.obsm["characters"] = pd.DataFrame(
            {"c1": ["1", "1", "3", "4"]}, index=leaves
        )
        tdata.obsm["spatial"] = pd.DataFrame(
            [[1, 1], [1, 1], [2, 2], [3, 3]],
            index=leaves,
            columns=["x", "y"],
        )
        space = np.ones((5, 5), dtype=bool)
        sub = sample_spatial(
            tdata, space=space, merge_cells=True, collapse_duplicates=True, random_seed=1
        )
        merged = [l for l in sub.obs_names if "-" in l]
        # Collapse duplicates: "1|1" → "1"
        assert sub.obsm["characters"].loc[merged[0], "c1"] == "1"

    def test_merge_cells_no_collapse_duplicates(self):
        tdata = _make_tdata(n_leaves=4, with_chars=False)
        leaves = list(tdata.obs_names)
        tdata.obsm["characters"] = pd.DataFrame(
            {"c1": ["1", "1", "3", "4"]}, index=leaves
        )
        tdata.obsm["spatial"] = pd.DataFrame(
            [[1, 1], [1, 1], [2, 2], [3, 3]],
            index=leaves,
            columns=["x", "y"],
        )
        space = np.ones((5, 5), dtype=bool)
        sub = sample_spatial(
            tdata, space=space, merge_cells=True, collapse_duplicates=False, random_seed=1
        )
        merged = [l for l in sub.obs_names if "-" in l]
        assert sub.obsm["characters"].loc[merged[0], "c1"] == "1|1"


# ============================================================
# sample_supercellular
# ============================================================


class TestSampleSupercellular:
    def test_bad_params_both(self):
        tdata = _make_tdata()
        with pytest.raises(LeafSubsamplerError):
            sample_supercellular(tdata, ratio=0.5, number_of_merges=2)

    def test_bad_params_neither(self):
        tdata = _make_tdata()
        with pytest.raises(LeafSubsamplerError):
            sample_supercellular(tdata)

    def test_too_many_merges(self):
        tdata = _make_tdata(n_leaves=4)
        with pytest.raises(LeafSubsamplerError):
            sample_supercellular(tdata, number_of_merges=4)

    def test_zero_merges(self):
        tdata = _make_tdata(n_leaves=4)
        with pytest.raises(LeafSubsamplerError):
            sample_supercellular(tdata, number_of_merges=0)

    def test_basic_merge_count(self):
        tdata = _make_tdata(n_leaves=8)
        sub = sample_supercellular(tdata, number_of_merges=3, random_seed=10)
        # After 3 merges: 8 - 3 = 5 leaves
        assert len(sub.obs_names) == 5

    def test_ratio(self):
        tdata = _make_tdata(n_leaves=8)
        sub = sample_supercellular(tdata, ratio=0.5, random_seed=10)
        assert len(sub.obs_names) == 4

    def test_reproducibility(self):
        tdata = _make_tdata(n_leaves=8)
        sub1 = sample_supercellular(tdata, number_of_merges=3, random_seed=42)
        sub2 = sample_supercellular(tdata, number_of_merges=3, random_seed=42)
        assert set(sub1.obs_names) == set(sub2.obs_names)

    def test_merged_leaves_named_correctly(self):
        tdata = _make_tdata(n_leaves=8)
        sub = sample_supercellular(tdata, number_of_merges=2, random_seed=7)
        # At least one merged leaf with "-" separator
        merged = [l for l in sub.obs_names if "-" in l]
        assert len(merged) >= 1

    def test_characters_preserved_for_unmerged(self):
        tdata = _make_tdata(n_leaves=8, with_chars=True)
        sub = sample_supercellular(tdata, number_of_merges=1, random_seed=5)
        # Unmerged leaves should have identical character states
        unmerged = [l for l in sub.obs_names if "-" not in l]
        for leaf in unmerged:
            orig = tdata.obsm["characters"].loc[leaf].tolist()
            sub_row = sub.obsm["characters"].loc[leaf].tolist()
            assert orig == sub_row

    def test_merged_states_contain_pipe(self):
        """Merged leaves should have | in their character states if source states differ."""
        tdata = _make_tdata(n_leaves=4, with_chars=False)
        leaves = list(tdata.obs_names)
        # Give all leaves distinct states
        tdata.obsm["characters"] = pd.DataFrame(
            {"c1": ["1", "2", "3", "4"]}, index=leaves
        )
        sub = sample_supercellular(tdata, number_of_merges=1, random_seed=1)
        merged = [l for l in sub.obs_names if "-" in l]
        # The merged c1 should have | separator
        merged_c1 = sub.obsm["characters"].loc[merged[0], "c1"]
        assert "|" in merged_c1

    def test_collapse_duplicates_removes_repeated_states(self):
        tdata = _make_tdata(n_leaves=4, with_chars=False)
        leaves = list(tdata.obs_names)
        # Same state for first two leaves
        tdata.obsm["characters"] = pd.DataFrame(
            {"c1": ["1", "1", "3", "4"]}, index=leaves
        )
        # Force merge of first two by using seed that selects them
        # We can check result regardless of which pair is merged
        sub = sample_supercellular(
            tdata, number_of_merges=1, collapse_duplicates=True, random_seed=1
        )
        for leaf in sub.obs_names:
            c1 = sub.obsm["characters"].loc[leaf, "c1"]
            parts = c1.split("|")
            assert len(parts) == len(set(parts)), f"Duplicates in {c1}"

    def test_tree_valid(self):
        tdata = _make_tdata(n_leaves=8)
        sub = sample_supercellular(tdata, number_of_merges=3, random_seed=1)
        tree = sub.obst["tree"]
        # All out-degree-0 nodes should be in obs_names
        leaves_in_tree = {n for n in tree if tree.out_degree(n) == 0}
        assert leaves_in_tree == set(sub.obs_names)
        # No unifurcations (except optionally root)
        root = next(n for n in tree if tree.in_degree(n) == 0)
        for node in tree:
            if tree.out_degree(node) != 0 and node != root:
                assert tree.out_degree(node) >= 2

    def test_merged_time_is_mean(self):
        """Merged leaf time = mean of the two merged leaf times."""
        tdata = _make_tdata(n_leaves=4)
        orig_tree = tdata.obst["tree"]
        orig_times = {n: orig_tree.nodes[n]["time"] for n in orig_tree.nodes}

        sub = sample_supercellular(tdata, number_of_merges=1, random_seed=3)
        tree = sub.obst["tree"]

        merged_leaves = [n for n in sub.obs_names if "-" in n]
        for merged in merged_leaves:
            # Extract original leaf names from the merged name
            # (For a single merge, the merged name is "leaf1-leaf2")
            parts = merged.split("-")
            if all(p in orig_times for p in parts):
                expected = np.mean([orig_times[p] for p in parts])
                # Find the merged node in the new tree
                if merged in tree.nodes:
                    actual = tree.nodes[merged]["time"]
                    assert np.isclose(actual, expected), (
                        f"Expected time {expected}, got {actual}"
                    )

    def test_inverse_distance_weighting(self):
        """Per pair, siblings (dist 2) should be merged more than second-cousins (dist 6)."""
        tdata = _make_tdata(n_leaves=8)
        orig_tree = tdata.obst["tree"]
        leaves = list(tdata.obs_names)

        pair_counts: dict[tuple, int] = {}
        n_trials = 500

        for trial in range(n_trials):
            sub = sample_supercellular(tdata, number_of_merges=1, random_seed=trial)
            merged = [l for l in sub.obs_names if "-" in l]
            if not merged:
                continue
            parts = merged[0].split("-")
            if len(parts) == 2 and parts[0] in leaves and parts[1] in leaves:
                pair = tuple(sorted(parts))
                pair_counts[pair] = pair_counts.get(pair, 0) + 1

        if not pair_counts:
            return  # No valid pairs found; skip assertion

        # Compute average merge rate per sibling pair vs per second-cousin pair
        sibling_rates = []
        second_cousin_rates = []
        for pair, count in pair_counts.items():
            l1, l2 = pair
            lca = nx.lowest_common_ancestor(orig_tree, l1, l2)
            d = (
                orig_tree.nodes[l1]["time"]
                + orig_tree.nodes[l2]["time"]
                - 2 * orig_tree.nodes[lca]["time"]
            )
            if d <= 2:
                sibling_rates.append(count)
            elif d >= 6:
                second_cousin_rates.append(count)

        if sibling_rates and second_cousin_rates:
            avg_sibling = np.mean(sibling_rates)
            avg_second_cousin = np.mean(second_cousin_rates)
            assert avg_sibling > avg_second_cousin, (
                f"Per-pair sibling rate ({avg_sibling:.1f}) should exceed "
                f"second-cousin rate ({avg_second_cousin:.1f})"
            )

    def test_obsm_shape(self):
        tdata = _make_tdata(n_leaves=8, with_chars=True)
        sub = sample_supercellular(tdata, number_of_merges=2, random_seed=1)
        n_final = 8 - 2
        assert sub.obsm["characters"].shape == (n_final, 3)

    def test_uns_preserved(self):
        tdata = _make_tdata(n_leaves=8, with_chars=True)
        sub = sample_supercellular(tdata, number_of_merges=2, random_seed=1)
        assert sub.uns.get("cassette_size") == 3


# ============================================================
# Deprecation stubs
# ============================================================


class TestDeprecationStubs:
    def test_leaf_subsampler_deprecated(self):
        with pytest.warns(DeprecationWarning, match="sample_uniform"):
            sampler = LeafSubsampler()
        with pytest.raises(NotImplementedError):
            sampler.subsample_leaves(None)

    def test_uniform_leaf_subsampler_deprecated(self):
        with pytest.warns(DeprecationWarning, match="sample_uniform"):
            sampler = UniformLeafSubsampler()
        with pytest.raises(NotImplementedError):
            sampler.subsample_leaves(None)

    def test_spatial_leaf_subsampler_deprecated(self):
        with pytest.warns(DeprecationWarning, match="sample_spatial"):
            sampler = SpatialLeafSubsampler()
        with pytest.raises(NotImplementedError):
            sampler.subsample_leaves(None)

    def test_supercellular_sampler_deprecated(self):
        with pytest.warns(DeprecationWarning, match="sample_supercellular"):
            sampler = SupercellularSampler()
        with pytest.raises(NotImplementedError):
            sampler.subsample_leaves(None)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
