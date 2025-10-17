import networkx as nx
import pytest
from treedata import TreeData

import cassiopeia as cas
from cassiopeia import utils


@pytest.fixture
def tree():
    g = nx.DiGraph()
    g.add_edges_from(
        [
            ("0", "1"),
            ("0", "2"),
            ("1", "3"),
            ("2", "4"),
            ("2", "5"),
        ]
    )
    g["0"]["1"]["length"] = 1.0
    g["0"]["2"]["length"] = 2.0
    g["1"]["3"]["length"] = 1.0
    g["2"]["4"]["length"] = 2.0
    g["2"]["5"]["length"] = 1.0
    return g


def test_get_digraph_from_cassiopeia_tree(tree):
    cas_tree = cas.data.CassiopeiaTree(tree=tree)
    with pytest.warns(DeprecationWarning):
        result, _ = utils._get_digraph(cas_tree)
    assert isinstance(result, nx.DiGraph)
    assert set(result.nodes) == set(cas_tree.nodes)


def test_get_digraph_from_treedata(tree):
    tdata = TreeData(obst={"tree": tree})
    result, tree_key = utils._get_digraph(tdata)
    assert tree_key == "tree"
    assert isinstance(result, nx.DiGraph)
    assert set(result.nodes) == set(tree.nodes)
    # Multiple trees
    tdata2 = TreeData(obst={"tree1": tree, "tree2": tree})
    with pytest.raises(ValueError):
        utils._get_digraph(tdata2)
    with pytest.raises(ValueError):
        utils._get_digraph(tdata2, tree_key="bad")
    result2, _ = utils._get_digraph(tdata2, tree_key="tree1")
    assert isinstance(result2, nx.DiGraph)
    assert set(result2.nodes) == set(tree.nodes)
    # Bad input
    tdata3 = TreeData(obst={})
    with pytest.raises(ValueError):
        utils._get_digraph(tdata3)
    with pytest.raises(TypeError):
        utils._get_digraph(tree="bad")


def test_get_leaves_sorted(tree):
    leaves = utils.get_leaves(tree)
    assert leaves == ["3", "4", "5"]


def test_get_root_identifies_unique_root(tree):
    root = utils.get_root(tree)
    assert root == "0"
    # Does not have a root
    cycle = nx.DiGraph()
    cycle.add_edges_from([("A", "B"), ("B", "C"), ("C", "A")])
    with pytest.raises(ValueError):
        utils.get_root(cycle)
    # Multiple roots
    multi_root = nx.DiGraph()
    multi_root.add_edges_from([("A", "B"), ("C", "D")])
    with pytest.raises(ValueError):
        utils.get_root(multi_root)


def test_collapse_unifurcations(tree):
    collapsed = utils.collapse_unifurcations(tree)
    assert "1" not in collapsed
    assert set(collapsed.successors("0")) == {"2", "3"}
    assert collapsed["0"]["2"]["length"] == pytest.approx(2.0)
    assert collapsed["0"]["3"]["length"] == pytest.approx(2.0)
    # Inplace
    tdata = TreeData(obst={"tree": tree})
    utils.collapse_unifurcations(tdata, tree_key="tree", inplace=True)
    assert "1" not in tdata.obst["tree"]
    assert set(tdata.obst["tree"].successors("0")) == {"2", "3"}
    assert tdata.obst["tree"]["0"]["2"]["length"] == pytest.approx(2.0)
    assert tdata.obst["tree"]["0"]["3"]["length"] == pytest.approx(2.0)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
