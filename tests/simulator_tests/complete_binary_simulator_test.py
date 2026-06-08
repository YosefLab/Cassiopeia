"""Tests for complete_binary()."""

import networkx as nx
import pytest

from cassiopeia.mixins import TreeSimulatorError
from cassiopeia.simulator import complete_binary


@pytest.fixture
def tdata():
    return complete_binary(depth=2)


# --- complete_binary() tests ---


@pytest.mark.parametrize("kwargs", [{}, {"num_cells": 3}, {"depth": 0}])
def test_complete_binary_raises(kwargs):
    with pytest.raises(TreeSimulatorError):
        complete_binary(**kwargs)


def test_complete_binary_num_cells_sets_depth():
    tdata = complete_binary(num_cells=4)
    tree = tdata.obst["simulated"]
    leaves = [n for n in tree if tree.out_degree(n) == 0]
    assert len(leaves) == 4


def test_complete_binary_tree_structure(tdata):
    tree = tdata.obst["simulated"]
    assert set(tree.nodes) == {"root", "1", "2", "3", "4", "5", "6", "7"}
    assert set(tree.edges) == {
        ("root", "1"),
        ("1", "2"),
        ("1", "3"),
        ("2", "4"),
        ("2", "5"),
        ("3", "6"),
        ("3", "7"),
    }


def test_complete_binary_branch_lengths(tdata):
    tree = tdata.obst["simulated"]
    assert nx.get_node_attributes(tree, "time") == {
        "root": 0.0,
        "1": 1 / 3,
        "2": 2 / 3,
        "3": 2 / 3,
        "4": 1.0,
        "5": 1.0,
        "6": 1.0,
        "7": 1.0,
    }
    assert nx.get_node_attributes(tree, "depth") == {
        "root": 0,
        "1": 1,
        "2": 2,
        "3": 2,
        "4": 3,
        "5": 3,
        "6": 3,
        "7": 3,
    }


def test_complete_binary_custom_key_added():
    tdata = complete_binary(depth=2, key_added="mytree")
    assert "mytree" in tdata.obst
    assert "tree" not in tdata.obst


# --- alignment ---


def test_complete_binary_alignment_leaves_default():
    tdata = complete_binary(depth=2)
    tree = tdata.obst["simulated"]
    leaves = {n for n in tree if tree.out_degree(n) == 0}
    assert set(tdata.obs_names) == leaves
    # time/depth populated for every observation (no NaN).
    assert tdata.obs["time"].notna().all()
    assert tdata.obs["depth"].notna().all()


def test_complete_binary_alignment_nodes():
    tdata = complete_binary(depth=2, alignment="nodes")
    tree = tdata.obst["simulated"]
    # obs spans every node (internal nodes included)...
    assert set(tdata.obs_names) == set(tree.nodes)
    # ...and time/depth are populated for all of them, not just leaves.
    assert tdata.obs["time"].notna().all()
    assert tdata.obs["depth"].notna().all()


if __name__ == "__main__":
    pytest.main([__file__])
