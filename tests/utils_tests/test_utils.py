import networkx as nx
import numpy as np
import pandas as pd
import pytest
import treedata as td
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


def test_collapse_unifurcations(tree):
    collapsed = utils._collapse_unifurcations(tree)
    assert "1" not in collapsed
    assert set(collapsed.successors("0")) == {"2", "3"}
    assert collapsed["0"]["2"]["length"] == pytest.approx(2.0)
    assert collapsed["0"]["3"]["length"] == pytest.approx(2.0)
    # Inplace
    tdata = TreeData(obst={"tree": tree})
    utils._collapse_unifurcations(tdata, tree_key="tree", inplace=True)
    assert "1" not in tdata.obst["tree"]
    assert set(tdata.obst["tree"].successors("0")) == {"2", "3"}
    assert tdata.obst["tree"]["0"]["2"]["length"] == pytest.approx(2.0)
    assert tdata.obst["tree"]["0"]["3"]["length"] == pytest.approx(2.0)


def test_get_characters_cassiopeia_layer():
    """Test getting character matrix from CassiopeiaTree layer."""
    tree = nx.DiGraph()
    tree.add_edges_from([("root", "A")])
    cm = pd.DataFrame({"A": [0, 1, -1]}).T
    cas_tree = cas.data.CassiopeiaTree(tree=tree, character_matrix=cm)
    cas_tree.layers["alternative"] = cm * 2
    result = utils._get_characters(cas_tree, "alternative")
    assert isinstance(result, pd.DataFrame)


def test_get_characters_treedata(tree):
    """Test getting character matrix from TreeData obsm."""
    tdata = TreeData(obst={"tree": tree})
    obs_order = tdata.obs_names.tolist()
    cm = pd.DataFrame([[0, 1, -1], [1, 0, -1], [1, 1, 0]], index=obs_order)
    tdata.obsm["characters"] = cm
    result = utils._get_characters(tdata)
    assert isinstance(result, pd.DataFrame)


def test_get_characters_converts_numpy_array():
    """Test that numpy arrays are converted to DataFrames."""
    tree = nx.DiGraph()
    tree.add_edges_from([("root", "A"), ("root", "B")])
    leaves = sorted([n for n in tree.nodes() if tree.out_degree(n) == 0])
    cm_array = np.array([[0, 1, -1], [1, 0, -1]])
    tdata = TreeData(
        obst={"tree": tree}, obsm={"characters": cm_array}, obs=pd.DataFrame(index=leaves)
    )
    result = utils._get_characters(tdata)
    assert isinstance(result, pd.DataFrame)
    assert result.shape == (2, 3)


def test_get_parameter(tree):
    """Test getting parameter from TreeData and non-tree objects."""
    tdata = TreeData(obst={"tree": tree})
    tdata.uns["mutation_rate"] = 0.75
    result = utils._get_parameter(tdata, "mutation_rate")
    assert result == 0.75
    result = utils._get_parameter(None, "param", value="default_val")
    assert result == "default_val"


def test_check_tree_has_key_missing():
    """Test that _check_tree_has_key raises ValueError when key is missing."""
    tree = nx.DiGraph()
    tree.add_edges_from([("root", "A"), ("root", "B")])

    with pytest.raises(ValueError, match="One or more nodes do not have 'depth' attribute"):
        utils._check_tree_has_key(tree, "depth")


if __name__ == "__main__":
    pytest.main(["-v", __file__])


@pytest.mark.parametrize(
    ("stored", "priors", "expected"),
    [
        ({0: {1: 0.5}}, None, {0: {1: 0.5}}),
        (None, None, None),
        ({0: {1: 0.5}}, True, {0: {1: 0.5}}),
        ({0: {1: 0.5}}, False, None),
        ({0: {1: 0.5}}, {1: {2: 0.1}}, {1: {2: 0.1}}),
        ([{1: 0.5}, {2: 0.1}], None, {0: {1: 0.5}, 1: {2: 0.1}}),
    ],
)
def test_resolve_priors(stored, priors, expected):
    tdata = TreeData(obs=pd.DataFrame(index=["a"]), uns={"priors": stored} if stored else {})
    assert utils._resolve_priors(tdata, priors) == expected


def test_resolve_priors_true_without_stored_raises():
    with pytest.raises(ValueError, match="priors=True"):
        utils._resolve_priors(TreeData(obs=pd.DataFrame(index=["a"])), True)


@pytest.mark.parametrize(
    "priors",
    [{0: {1: 0.5}, 1: {"2": 0.25}}, {"0": {"1": 0.5}, "1": {"2": 0.25}}, [{1: 0.5}, {"2": 0.25}]],
)
def test_format_priors(priors):
    assert utils._format_priors(priors) == {"0": {"1": 0.5}, "1": {"2": 0.25}}


@pytest.mark.parametrize("priors", [0.5, {0: 0.1}, [0.5]])
def test_malformed_priors_raise(priors):
    with pytest.raises(ValueError, match="state probabilities"):
        utils._format_priors(priors)
    with pytest.raises(ValueError, match="state probabilities"):
        utils._resolve_priors(TreeData(obs=pd.DataFrame(index=["a"])), priors)


@pytest.mark.parametrize(
    ("column", "expected"),
    [
        (["*", "1", "-"], {"1": 0.5, "3": 0.5}),  # string states; unseen "3" stays a string
        ([0, 1, -1], {1: 0.5, 3: 0.5}),  # integer states
        ([0.0, 1.0, np.nan], {1.0: 0.5, 3: 0.5}),  # float states match their integer form
        (["*", ("1", "2"), "-"], {"1": 0.5, "3": 0.5}),  # ambiguous (tuple) states
        (pd.Categorical(["*", "*", "-"], categories=["*", "1", "-"]), {"1": 0.5, "3": 0.5}),
    ],
)
def test_resolve_priors_matches_matrix_states(column, expected):
    chars = pd.DataFrame({"c0": column}, index=["a", "b", "c"])
    tdata = TreeData(obs=pd.DataFrame(index=chars.index), obsm={"characters": chars})
    tdata.uns["priors"] = utils._format_priors({0: {1: 0.5, 3: 0.5}})
    resolved = utils._resolve_priors(tdata, characters=chars)
    assert resolved == {0: expected}
    assert all(type(k) is type(e) for k, e in zip(resolved[0], expected, strict=True))


def test_stored_priors_round_trip_h5td(tmp_path):
    chars = pd.DataFrame({"c0": ["*", "1"], "c1": ["2", "-"]}, index=["a", "b"])
    tdata = TreeData(obs=pd.DataFrame(index=chars.index), obsm={"characters": chars})
    tdata.uns["priors"] = utils._format_priors([{"1": 0.5}, {"2": 1.0}])
    tdata.write_h5td(tmp_path / "t.h5td")
    back = td.read_h5td(tmp_path / "t.h5td")
    expected = {0: {"1": 0.5}, 1: {"2": 1.0}}
    assert utils._resolve_priors(back, True, back.obsm["characters"]) == expected


def test_multiple_trees_error_lists_keys():
    tdata = TreeData(obs=pd.DataFrame(index=["a"]))
    tdata.obst["t1"] = nx.DiGraph([("r", "a")])
    tdata.obst["t2"] = nx.DiGraph([("r", "a")])
    with pytest.raises(ValueError, match="'t1', 't2'.*tree_key"):
        utils._get_digraph(tdata)


def test_default_parameter_warns_once(monkeypatch):
    import warnings

    monkeypatch.setattr(utils, "_WARNED_DEFAULT_PARAMETERS", set())
    tdata = TreeData(obs=pd.DataFrame(index=["a"]))
    with pytest.warns(UserWarning, match="missing_state"):
        utils._get_parameter(tdata, "missing_state")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert utils._get_parameter(tdata, "missing_state") == (-1, "-1", "NA", "-")


def test_node_name_generator_unique_with_coarse_clock(monkeypatch):
    """Names stay unique when the clock does not advance (macOS has a microsecond clock)."""
    import time

    monkeypatch.setattr(time, "time", lambda: 0.0)
    gen = utils._node_name_generator()
    names = [next(gen) for _ in range(1000)]
    assert len(set(names)) == len(names)
    assert all(name.startswith("cassiopeia_internal_node") for name in names)
    assert next(utils._node_name_generator()) not in names
