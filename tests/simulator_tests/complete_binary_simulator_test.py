import networkx as nx
import pytest

from cassiopeia.mixins import TreeSimulatorError
from cassiopeia.simulator import CompleteBinarySimulator


@pytest.fixture
def tdata():
    """A precomputed tree from the depth-2 simulator."""
    return CompleteBinarySimulator(depth=2).simulate_tree()


@pytest.mark.parametrize("kwargs", [{}, {"num_cells": 3}, {"depth": 0}])
def test_init_raises(kwargs):
    with pytest.raises(TreeSimulatorError):
        CompleteBinarySimulator(**kwargs)


def test_init_num_cells_sets_depth():
    simulator = CompleteBinarySimulator(num_cells=4)
    assert simulator.depth == 2


def test_binary_tree_structure(tdata):
    tree = tdata.obst["tree"]
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


def test_binary_branch_lengths(tdata):
    tree = tdata.obst["tree"]
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


if __name__ == "__main__":
    pytest.main([__file__])
