# tests/test_small_parsimony_pytest.py

import networkx as nx
import pandas as pd
import pytest
import treedata as td

import cassiopeia as cas
from cassiopeia.mixins import (
    CassiopeiaError,
    FitchCountError,
)
from cassiopeia.tools.small_parsimony import (
    fitch_hartigan_bottom_up,
    fitch_hartigan_top_down,
)


@pytest.fixture
def binary_tree():
    G = nx.DiGraph()
    G.add_edges_from(
        [
            ("0", "1"),
            ("0", "2"),
            ("1", "3"),
            ("1", "4"),
            ("2", "5"),
            ("2", "6"),
            ("3", "7"),
            ("3", "8"),
            ("4", "9"),
            ("4", "10"),
            ("5", "11"),
            ("5", "12"),
            ("6", "13"),
            ("6", "14"),
        ]
    )
    return G


@pytest.fixture
def binary_meta():
    return pd.DataFrame.from_dict(
        {
            "7": ["A", 10],
            "8": ["G", 2],
            "9": ["A", 2],
            "10": ["A", 12],
            "11": ["C", 1],
            "12": ["A", 5],
            "13": ["G", 8],
            "14": ["G", 9],
        },
        orient="index",
        columns=["nucleotide", "quality"],
    )


@pytest.fixture
def binary_tdata(binary_tree, binary_meta):
    return td.TreeData(obs=binary_meta, obst={"binary": binary_tree})


@pytest.fixture
def tree():
    G = nx.DiGraph()
    G.add_edges_from(
        [
            ("0", "1"),
            ("0", "2"),
            ("1", "3"),
            ("1", "4"),
            ("2", "5"),
            ("2", "14"),
            ("4", "8"),
            ("4", "9"),
            ("4", "10"),
            ("5", "11"),
            ("5", "12"),
            ("5", "13"),
            ("1", "15"),
            ("4", "16"),
        ]
    )
    return G


@pytest.fixture
def meta():
    return pd.DataFrame.from_dict(
        {
            "3": ["A", 10],
            "8": ["G", 2],
            "9": ["G", 2],
            "10": ["A", 12],
            "11": ["C", 1],
            "12": ["A", 5],
            "13": ["G", 8],
            "14": ["G", 9],
            "15": ["G", 22],
            "16": ["A", 5],
        },
        orient="index",
        columns=["nucleotide", "quality"],
    )


def test_fitch_hartigan_bottom_up_errors(binary_tree):
    with pytest.raises(CassiopeiaError):
        fitch_hartigan_bottom_up(binary_tree, "quality")
    with pytest.raises(CassiopeiaError):
        fitch_hartigan_bottom_up(binary_tree, "imaginary_column")


def test_fitch_hartigan_bottom_up(binary_tree, binary_meta):
    fitch_tree = fitch_hartigan_bottom_up(binary_tree, "nucleotide", meta_df=binary_meta, copy=True)

    expected_sets = {
        "0": ["A"],
        "1": ["A"],
        "2": ["C", "A", "G"],
        "3": ["A", "G"],
        "4": ["A"],
        "5": ["C", "A"],
        "6": ["G"],
        "7": ["A"],
        "8": ["G"],
        "9": ["A"],
        "10": ["A"],
        "11": ["C"],
        "12": ["A"],
        "13": ["G"],
        "14": ["G"],
    }

    for n in fitch_tree.nodes:
        node_states = fitch_tree.nodes[n]["S1"]
        assert set(node_states) == set(expected_sets[n])

    # Custom key name
    fitch_hartigan_bottom_up(
        binary_tree, "nucleotide", meta_df=binary_meta, add_key="possible_states"
    )
    for n in fitch_tree.nodes:
        with pytest.raises(KeyError):
            binary_tree.nodes[n]["S1"]
        node_states = binary_tree.nodes[n]["possible_states"]
        assert set(node_states) == set(expected_sets[n])


def test_fitch_hartigan_top_down(binary_tree, binary_meta):
    fitch_tree = fitch_hartigan_bottom_up(binary_tree, "nucleotide", meta_df=binary_meta, copy=True)
    fitch_hartigan_top_down(fitch_tree)

    expected_labels = {
        "0": "A",
        "1": "A",
        "2": "A",
        "3": "A",
        "4": "A",
        "5": "A",
        "6": "G",
        "7": "A",
        "8": "G",
        "9": "A",
        "10": "A",
        "11": "C",
        "12": "A",
        "13": "G",
        "14": "G",
    }

    for n in fitch_tree.nodes:
        node_state = fitch_tree.nodes[n]["label"]
        assert node_state == expected_labels[n]

    # Custom label key
    fitch_hartigan_bottom_up(binary_tree, "nucleotide")
    fitch_hartigan_top_down(binary_tree, label_key="nucleotide_assignment")

    for n in binary_tree.nodes:
        with pytest.raises(KeyError):
            binary_tree.nodes[n]["label"]
        node_state = binary_tree.nodes[n]["nucleotide_assignment"]
        assert node_state == expected_labels[n]


def test_fitch_hartigan(binary_tdata):
    cas.tl.fitch_hartigan(binary_tdata, "nucleotide")

    expected_labels = {
        "0": "A",
        "1": "A",
        "2": "A",
        "3": "A",
        "4": "A",
        "5": "A",
        "6": "G",
        "7": "A",
        "8": "G",
        "9": "A",
        "10": "A",
        "11": "C",
        "12": "A",
        "13": "G",
        "14": "G",
    }

    for n, node_state in nx.get_node_attributes(binary_tdata.trees["binary"], "label").items():
        assert node_state == expected_labels[n]


def test_score_parsimony(binary_tdata):
    with pytest.raises(CassiopeiaError):
        cas.tl.score_small_parsimony(binary_tdata, "nucleotide", None, False, "label")

    parsimony = cas.tl.score_small_parsimony(
        binary_tdata, "nucleotide", infer_ancestral_states=True
    )
    assert parsimony == 3


def test_tree_fitch_bottom_up(tree, meta):
    fitch_hartigan_bottom_up(tree, "nucleotide", meta_df=meta)

    expected_sets = {
        "0": ["G"],
        "1": ["G", "A"],
        "2": ["G"],
        "3": ["A"],
        "4": ["G", "A"],
        "5": ["C", "A", "G"],
        "8": ["G"],
        "9": ["G"],
        "10": ["A"],
        "11": ["C"],
        "12": ["A"],
        "13": ["G"],
        "14": ["G"],
        "15": ["G"],
        "16": ["A"],
    }

    for n, node_states in nx.get_node_attributes(tree, "S1").items():
        assert set(node_states) == set(expected_sets[n])


def test_tree_fitch_hartigan(tree, meta):
    cas.tl.fitch_hartigan(tree, "nucleotide", meta_df=meta)

    expected_labels = {
        "0": "G",
        "1": "G",
        "2": "G",
        "3": "A",
        "4": "G",
        "5": "G",
        "8": "G",
        "9": "G",
        "10": "A",
        "11": "C",
        "12": "A",
        "13": "G",
        "14": "G",
        "15": "G",
        "16": "A",
    }

    for n, node_states in nx.get_node_attributes(tree, "label").items():
        assert set(node_states) == set(expected_labels[n])


def test_tree_parsimony(tree, meta):
    parsimony = cas.tl.score_small_parsimony(
        tree, "nucleotide", infer_ancestral_states=True, meta_df=meta
    )
    assert parsimony == 5


def test_fitch_count_basic_binary(binary_tdata):
    fitch_matrix = cas.tl.fitch_count(binary_tdata, "nucleotide")

    num_nucleotides = binary_tdata.obs["nucleotide"].nunique()
    assert fitch_matrix.shape == (num_nucleotides, num_nucleotides)

    expected_matrix = pd.DataFrame.from_dict(
        {"A": [9, 2, 1], "G": [0, 2, 0], "C": [0, 0, 0]},
        orient="index",
        columns=["A", "G", "C"],
    ).astype(float)

    pd.testing.assert_frame_equal(expected_matrix, fitch_matrix)

    # If ancestral states already assigned
    fitch_hartigan_bottom_up(binary_tdata, "nucleotide", add_key="nucleotide_sets")
    fitch_matrix_no_infer = cas.tl.fitch_count(
        binary_tdata,
        "nucleotide",
        infer_ancestral_states=False,
        state_key="nucleotide_sets",
    )
    pd.testing.assert_frame_equal(expected_matrix, fitch_matrix_no_infer)


def test_fitch_count_basic_binary_custom_state_space(binary_tdata):
    fitch_matrix = cas.tl.fitch_count(
        binary_tdata, "nucleotide", unique_states=["A", "G", "C", "N"]
    )

    expected_matrix = pd.DataFrame.from_dict(
        {"A": [9, 2, 1, 0], "G": [0, 2, 0, 0], "C": [0, 0, 0, 0], "N": [0, 0, 0, 0]},
        orient="index",
        columns=["A", "G", "C", "N"],
    ).astype(float)

    pd.testing.assert_frame_equal(expected_matrix, fitch_matrix)

    with pytest.raises(FitchCountError):
        cas.tl.fitch_count(binary_tdata, "nucleotide", unique_states=["A", "G"])


def test_fitch_count_basic_binary_internal_node(binary_tdata):
    fitch_matrix = cas.tl.fitch_count(binary_tdata, "nucleotide", root="5")

    num_nucleotides = binary_tdata.obs["nucleotide"].nunique()
    assert fitch_matrix.shape == (num_nucleotides, num_nucleotides)

    expected_matrix = pd.DataFrame.from_dict(
        {"A": [1, 0, 1], "G": [0, 0, 0], "C": [1, 0, 1]},
        orient="index",
        columns=["A", "G", "C"],
    ).astype(float)

    pd.testing.assert_frame_equal(expected_matrix, fitch_matrix)


if __name__ == "__main__":
    pytest.main([__file__])
