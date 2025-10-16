import networkx as nx
import pytest
from treedata import TreeData

import cassiopeia as cas
from cassiopeia import utils


def _build_sample_graph():
    graph = nx.DiGraph()
    graph.add_edges_from(
        [
            ("root", "inner"),
            ("inner", "leaf_a"),
            ("inner", "mid"),
            ("mid", "leaf_b"),
        ]
    )
    graph["root"]["inner"]["length"] = 1.0
    graph["inner"]["leaf_a"]["length"] = 1.0
    graph["inner"]["mid"]["length"] = 2.0
    graph["mid"]["leaf_b"]["length"] = 1.0
    return graph


def test_to_networkx_from_cassiopeia_tree():
    graph = _build_sample_graph()
    cas_tree = cas.data.CassiopeiaTree(tree=graph)

    result = utils._to_networkx(cas_tree)

    assert isinstance(result, nx.DiGraph)
    assert set(result.nodes) == set(cas_tree.nodes)


def test_to_networkx_from_treedata_requires_key():
    graph = _build_sample_graph()
    tdata = TreeData(obst={"g1": graph, "g2": graph.copy()}, alignment="subset")

    with pytest.raises(ValueError):
        utils._to_networkx(tdata)

    with pytest.raises(ValueError):
        utils._to_networkx(tdata, key="missing")

    selected = utils._to_networkx(tdata, key="g1")
    assert set(selected.edges) == set(graph.edges)


def test_get_leaves_sorted():
    graph = _build_sample_graph()

    leaves = utils.get_leaves(graph)

    assert leaves == ["leaf_a", "leaf_b"]


def test_get_root_identifies_unique_root():
    graph = _build_sample_graph()

    assert utils.get_root(graph) == "root"


def test_collapse_unifurcations_collapses_chain():
    graph = _build_sample_graph()

    collapsed = utils.collapse_unifurcations(graph)

    assert "inner" not in collapsed
    assert "mid" not in collapsed
    assert set(collapsed.successors("root")) == {"leaf_a", "leaf_b"}
    assert collapsed["root"]["leaf_a"]["length"] == pytest.approx(2.0)
    assert collapsed["root"]["leaf_b"]["length"] == pytest.approx(4.0)


def test_collapse_unifurcations_matches_cassiopeia_tree():
    graph = _build_sample_graph()
    cas_tree = cas.data.CassiopeiaTree(tree=graph)
    for parent, child in cas_tree.edges:
        cas_tree.set_branch_length(parent, child, graph[parent][child]["length"])

    collapsed_from_tree = utils.collapse_unifurcations(cas_tree)
    collapsed_from_graph = utils.collapse_unifurcations(graph)

    assert set(collapsed_from_tree.edges) == set(collapsed_from_graph.edges)
    for edge in collapsed_from_graph.edges:
        assert collapsed_from_tree[edge[0]][edge[1]]["length"] == pytest.approx(
            collapsed_from_graph[edge[0]][edge[1]]["length"]
        )
