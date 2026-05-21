"""Tests for simple_fit_subclone."""

import numpy as np
import pytest
import treedata as td

from cassiopeia.simulator import simple_fit_subclone


# ============================================================
# simple_fit_subclone
# ============================================================


def test_deterministic_nodes_and_edges():
    tdata = simple_fit_subclone(
        branch_length_neutral=1,
        branch_length_fit=0.5,
        experiment_duration=1.9,
        generations_until_fit_subclone=1,
    )
    tree = tdata.obst["tree"]
    assert list(tree.nodes) == [
        "0_neutral", "1_neutral", "2_fit", "3_neutral", "4_fit", "5_fit"
    ]
    assert list(tree.edges) == [
        ("0_neutral", "1_neutral"),
        ("1_neutral", "2_fit"),
        ("1_neutral", "3_neutral"),
        ("2_fit", "4_fit"),
        ("2_fit", "5_fit"),
    ]


def test_deterministic_times():
    tdata = simple_fit_subclone(
        branch_length_neutral=1,
        branch_length_fit=0.5,
        experiment_duration=1.9,
        generations_until_fit_subclone=1,
    )
    tree = tdata.obst["tree"]
    times = {n: tree.nodes[n]["time"] for n in tree.nodes}
    assert times == {
        "0_neutral": 0.0,
        "1_neutral": 1.0,
        "2_fit": 1.5,
        "3_neutral": 1.9,
        "4_fit": 1.9,
        "5_fit": 1.9,
    }


def test_returns_treedata():
    tdata = simple_fit_subclone(
        branch_length_neutral=1,
        branch_length_fit=0.5,
        experiment_duration=1.9,
        generations_until_fit_subclone=1,
    )
    assert isinstance(tdata, td.TreeData)


def test_leaves_at_experiment_duration():
    experiment_duration = 3.0
    tdata = simple_fit_subclone(
        branch_length_neutral=1,
        branch_length_fit=0.5,
        experiment_duration=experiment_duration,
        generations_until_fit_subclone=2,
    )
    tree = tdata.obst["tree"]
    leaves = [n for n in tree if tree.out_degree(n) == 0]
    for leaf in leaves:
        assert tree.nodes[leaf]["time"] == experiment_duration


def test_leaf_labels_neutral_and_fit():
    tdata = simple_fit_subclone(
        branch_length_neutral=1,
        branch_length_fit=0.5,
        experiment_duration=3.0,
        generations_until_fit_subclone=2,
    )
    tree = tdata.obst["tree"]
    leaves = [n for n in tree if tree.out_degree(n) == 0]
    assert all(l.endswith("_neutral") or l.endswith("_fit") for l in leaves)
    assert any(l.endswith("_fit") for l in leaves)
    assert any(l.endswith("_neutral") for l in leaves)


def test_exactly_one_fit_subclone_root():
    """The fit subclone starts from exactly one node at the split generation."""
    tdata = simple_fit_subclone(
        branch_length_neutral=1,
        branch_length_fit=0.5,
        experiment_duration=4.0,
        generations_until_fit_subclone=2,
    )
    tree = tdata.obst["tree"]
    # Find internal nodes whose children include a "_fit" node
    fit_parents = [
        n for n in tree
        if any(c.endswith("_fit") for c in tree.successors(n))
        and not n.endswith("_fit")
    ]
    assert len(fit_parents) == 1


def test_stochastic_branch_lengths():
    """Callable branch lengths produce distinct internal branch lengths."""
    np.random.seed(1)

    def bl_neutral() -> float:
        return np.random.exponential(1.0)

    def bl_fit() -> float:
        return np.random.exponential(0.5)

    tdata = simple_fit_subclone(
        branch_length_neutral=bl_neutral,
        branch_length_fit=bl_fit,
        experiment_duration=4.9,
        generations_until_fit_subclone=2,
    )
    tree = tdata.obst["tree"]
    internal_bls = [
        tree.nodes[c]["time"] - tree.nodes[p]["time"]
        for p, c in tree.edges
        if tree.out_degree(c) != 0
    ]
    assert len(internal_bls) == len(set(internal_bls))


def test_custom_tree_key():
    tdata = simple_fit_subclone(
        branch_length_neutral=1,
        branch_length_fit=0.5,
        experiment_duration=1.9,
        generations_until_fit_subclone=1,
        tree_key="mytree",
    )
    assert "mytree" in tdata.obst
    assert "tree" not in tdata.obst


def test_all_nodes_have_time():
    tdata = simple_fit_subclone(
        branch_length_neutral=1,
        branch_length_fit=0.5,
        experiment_duration=3.0,
        generations_until_fit_subclone=2,
    )
    tree = tdata.obst["tree"]
    for node in tree.nodes:
        assert "time" in tree.nodes[node]


def test_later_subclone_generation():
    """generations_until_fit_subclone=3 means the subclone appears later."""
    tdata_early = simple_fit_subclone(
        branch_length_neutral=1,
        branch_length_fit=0.5,
        experiment_duration=5.0,
        generations_until_fit_subclone=1,
    )
    tdata_late = simple_fit_subclone(
        branch_length_neutral=1,
        branch_length_fit=0.5,
        experiment_duration=5.0,
        generations_until_fit_subclone=3,
    )
    # More fit leaves when subclone starts earlier (more generations to expand)
    tree_early = tdata_early.obst["tree"]
    tree_late = tdata_late.obst["tree"]
    fit_early = sum(1 for n in tree_early if tree_early.out_degree(n) == 0 and n.endswith("_fit"))
    fit_late = sum(1 for n in tree_late if tree_late.out_degree(n) == 0 and n.endswith("_fit"))
    assert fit_early >= fit_late
