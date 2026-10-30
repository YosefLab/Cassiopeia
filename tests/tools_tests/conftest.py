"""Shared TreeData fixtures for the tools tests."""

from __future__ import annotations

import networkx as nx
import pandas as pd
import pytest
import treedata as td

# The canonical small test topology used across the tools tests.
SMALL_NET_EDGES = [
    ("node5", "node0"),
    ("node5", "node1"),
    ("node6", "node2"),
    ("node6", "node3"),
    ("node6", "node4"),
    ("node7", "node5"),
    ("node7", "node6"),
]


def build_tree(
    edges,
    character_matrix,
    priors=None,
    missing_state=-1,
    unmodified_state=0,
    branch_lengths=None,
    uns=None,
):
    """Build a :class:`treedata.TreeData` from an edge list and character matrix.

    Edge ``length`` attributes default to 1.0 (override via ``branch_lengths``),
    and node ``time``/``depth`` attributes are computed as the cumulative branch
    length from the root.
    """
    g = nx.DiGraph()
    g.add_edges_from(edges)

    branch_lengths = branch_lengths or {}
    for u, v in g.edges:
        g.edges[u, v]["length"] = float(branch_lengths.get((u, v), 1.0))

    root = next(n for n in g.nodes if g.in_degree(n) == 0)
    times = {root: 0.0}
    for u, v in nx.bfs_edges(g, root):
        times[v] = times[u] + g.edges[u, v]["length"]
    for n in g.nodes:
        g.nodes[n]["time"] = times[n]
        g.nodes[n]["depth"] = times[n]

    obs = pd.DataFrame(index=list(character_matrix.index))
    uns_dict = {"missing_state": missing_state, "unmodified_state": unmodified_state}
    if priors is not None:
        uns_dict["priors"] = priors
    if uns:
        uns_dict.update(uns)

    return td.TreeData(
        obs=obs,
        obst={"tree": g},
        obsm={"characters": character_matrix},
        uns=uns_dict,
    )


@pytest.fixture
def tree_factory():
    """Return the :func:`build_tree` factory for ad-hoc TreeData construction."""
    return build_tree


@pytest.fixture
def discrete_tree():
    """Discrete (per-generation) small tree mirroring the legacy fixture."""
    cm = pd.DataFrame.from_dict(
        {
            "node0": [0, -1, -1],
            "node1": [1, 1, -1],
            "node2": [1, -1, -1],
            "node3": [1, -1, -1],
            "node4": [1, -1, -1],
        },
        orient="index",
    )
    priors = {0: {1: 1}, 1: {1: 1}, 2: {1: 1}}
    return build_tree(SMALL_NET_EDGES, cm, priors=priors)


@pytest.fixture
def continuous_tree():
    """Continuous (branch-length) small tree mirroring the legacy fixture."""
    cm = pd.DataFrame.from_dict(
        {
            "node0": [1, 0],
            "node1": [1, 1],
            "node2": [2, 3],
            "node3": [-1, 2],
            "node4": [-1, 1],
        },
        orient="index",
    )
    priors = {
        0: {1: 0.2, 2: 0.7, 3: 0.1},
        1: {1: 0.2, 2: 0.7, 3: 0.1},
        2: {1: 0.2, 2: 0.7, 3: 0.1},
    }
    branch_lengths = {("node5", "node0"): 1.5, ("node6", "node3"): 2.0}
    return build_tree(SMALL_NET_EDGES, cm, priors=priors, branch_lengths=branch_lengths)
