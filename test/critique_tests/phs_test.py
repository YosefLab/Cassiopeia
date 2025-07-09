"""
Tests for the cassiopeia.critique.phs module.
"""

import pytest

import networkx as nx
import treedata as td
import pandas as pd

import cassiopeia as cas

@pytest.fixture
def tree():
    t = nx.DiGraph()
    t.add_edges_from([("A", "B"), ("A", "C"), ("C", "D"), ("C", "E")])
    node_depths = {"A": 0, "B": 2, "C": 1, "D": 2, "E": 2}
    node_characters = {"A": [0,0], "B": [1,0], "C": [1,2], "D": [1,2], "E": [1,2]}
    nx.set_node_attributes(t, node_depths, "time")
    nx.set_node_attributes(t, node_characters, "characters")
    yield t


@pytest.fixture
def tdata(tree):
    tdata = td.TreeData(obs = pd.DataFrame(index = ["B","D","E"]),
        obsm = {"characters":pd.DataFrame(index = ["B","D","E"], data = [[1,0],[1,2],[1,2]])},
        obst = {"tree":tree})
    yield tdata


@pytest.fixture
def cas_tree(tree):
    cas_tree = cas.data.CassiopeiaTree(tree=tree)
    cas_tree.character_matrix = pd.DataFrame(index = ["B","D","E"], data = [[1,0],[1,2],[1,2]])
    node_depths = {"A": 0, "B": 2, "C": 1, "D": 2, "E": 2}
    node_characters = {"A": [0,0], "B": [1,0], "C": [1,2], "D": [1,2], "E": [1,2]}
    for node in cas_tree.nodes:
        cas_tree.set_attribute(node,"character_states",node_characters[node])
        cas_tree.set_attribute(node,"time",node_depths[node])
    cas_tree.priors = {1:.7, 2:.3}
    yield cas_tree


def test_phs(tdata, cas_tree):
    # TreeData input
    phs = cas.critique.cPHS(tdata,  mutation_rate=0.5, collision_probability=0.1)
    assert phs == pytest.approx(0.04608590676153751)
    assert cas.critique.cPHS(cas_tree,priors={1:.7, 2:.3}) == pytest.approx(0.96498842592592)
    # CassiopeiaTree input
    phs = cas.critique.cPHS(cas_tree,  mutation_rate=0.5, collision_probability=0.1)
    assert phs == pytest.approx(0.04608590676153751)
    assert cas.critique.cPHS(cas_tree) == pytest.approx(0.96498842592592)
    

if __name__ == "__main__":
    pytest.main(["-v", __file__])
