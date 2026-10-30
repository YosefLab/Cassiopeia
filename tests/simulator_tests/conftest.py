"""Shared fixtures for the simulator tests."""

import networkx as nx
import numpy as np
import pandas as pd
import pytest
import treedata as td


def _binary_tree(depth: int = 3) -> nx.DiGraph:
    """Balanced binary tree with integer ``time`` equal to node depth."""
    tree = nx.balanced_tree(2, depth, create_using=nx.DiGraph)
    tree = nx.relabel_nodes(tree, {i: str(i) for i in tree.nodes})
    root = "0"
    for node in tree.nodes:
        tree.nodes[node]["time"] = float(nx.shortest_path_length(tree, root, node))
    return tree


@pytest.fixture
def lineage_tree() -> td.TreeData:
    """A small ultrametric binary lineage tree (8 leaves) with a ``time`` attribute."""
    tree = _binary_tree(depth=3)
    leaves = [n for n in tree if tree.out_degree(n) == 0]
    return td.TreeData(obs=pd.DataFrame(index=leaves), obst={"simulated": tree})


@pytest.fixture
def trajectory() -> nx.DiGraph:
    """A small fate/factor tree with ``time`` and ``X_latent`` node attributes."""
    rng = np.random.default_rng(0)
    latent_dim = 5
    g = nx.DiGraph()
    g.add_node("r", time=0.0, X_latent=np.zeros(latent_dim))
    g.add_node("a", time=1.0, X_latent=rng.normal(size=latent_dim))
    g.add_node("b", time=1.0, X_latent=rng.normal(size=latent_dim))
    g.add_node("aa", time=2.0, X_latent=rng.normal(size=latent_dim))
    g.add_node("ab", time=2.0, X_latent=rng.normal(size=latent_dim))
    g.add_edges_from([("r", "a"), ("r", "b"), ("a", "aa"), ("a", "ab")])
    return g


@pytest.fixture
def char_tdata() -> td.TreeData:
    """A TreeData with a small categorical character matrix, priors, and missing data."""
    tree = _binary_tree(depth=3)
    leaves = [n for n in tree if tree.out_degree(n) == 0]  # 8 leaves
    rng = np.random.default_rng(0)
    states = ["0", "1", "2"]  # "0" unmodified, "1"/"2" edited
    data = {f"c{c}": [rng.choice(states) for _ in leaves] for c in range(3)}
    matrix = pd.DataFrame(data, index=leaves)
    # Introduce a couple of missing entries that must be left untouched.
    matrix.iloc[0, 0] = "-"
    matrix.iloc[3, 2] = "-"
    matrix = matrix.astype(pd.CategoricalDtype(categories=["0", "1", "2", "-"], ordered=True))

    tdata = td.TreeData(obs=pd.DataFrame(index=leaves), obst={"simulated": tree})
    tdata.obsm["characters"] = matrix
    tdata.uns["priors"] = {i: {"1": 0.5, "2": 0.5} for i in range(3)}
    tdata.uns["missing_state"] = "-"
    tdata.uns["unmodified_state"] = "0"
    return tdata
