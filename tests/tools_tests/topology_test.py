"""Tests for cassiopeia.tl.get_root and cassiopeia.tl.get_leaves."""

import networkx as nx
import pandas as pd
import pytest
import treedata as td

import cassiopeia as cas


def test_get_root_identifies_unique_root():
    g = nx.DiGraph()
    g.add_edges_from([("0", "1"), ("0", "2"), ("1", "3"), ("2", "4"), ("2", "5")])
    assert cas.tl.get_root(g) == "0"

    # No root (cycle)
    cycle = nx.DiGraph()
    cycle.add_edges_from([("A", "B"), ("B", "C"), ("C", "A")])
    with pytest.raises(ValueError):
        cas.tl.get_root(cycle)

    # Multiple roots
    multi_root = nx.DiGraph()
    multi_root.add_edges_from([("A", "B"), ("C", "D")])
    with pytest.raises(ValueError):
        cas.tl.get_root(multi_root)


def test_get_root_treedata():
    g = nx.DiGraph()
    g.add_edges_from([("root", "a"), ("root", "b")])
    tdata = td.TreeData(obs=pd.DataFrame(index=["a", "b"]), obst={"tree": g})
    assert cas.tl.get_root(tdata, tree_key="tree") == "root"


def test_get_leaves_traversal_order():
    # Successors are visited in insertion order, so the depth-first leaf order
    # reflects the topology rather than the (sorted) node labels.
    g = nx.DiGraph()
    g.add_edges_from([("root", "X"), ("root", "A"), ("X", "Z"), ("X", "Y")])
    leaves = cas.tl.get_leaves(g)
    assert leaves == ["Z", "Y", "A"]
    assert leaves != sorted(leaves)


def test_get_leaves_treedata():
    g = nx.DiGraph()
    g.add_edges_from([("root", "X"), ("root", "A"), ("X", "Z"), ("X", "Y")])
    tdata = td.TreeData(obs=pd.DataFrame(index=["Z", "Y", "A"]), obst={"tree": g})
    assert cas.tl.get_leaves(tdata, tree_key="tree") == ["Z", "Y", "A"]
