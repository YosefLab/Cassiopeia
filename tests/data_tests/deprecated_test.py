"""Tests for the deprecated utilities in cassiopeia/data/deprecated.py."""

import pytest

from cassiopeia.data import deprecated as data_deprecated
from cassiopeia.data import utilities as data_utilities


def test_utilities_module_reexports_deprecated():
    assert data_utilities.to_newick is data_deprecated.to_newick
    assert data_utilities.get_lca_characters is data_deprecated.get_lca_characters


# ── Removed functions raise ──────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("name", "replacement"),
    [
        ("compute_phylogenetic_weight_matrix", "pycea.tl.tree_distance"),
        ("net_relatedness_index", "pycea.tl.ancestral_linkage"),
        ("compute_inter_cluster_distances", "pycea.tl.ancestral_linkage"),
        ("to_newick", "treedata.io.write_nwk"),
        ("sample_bootstrap_character_matrices", "removed in v3.0.0"),
        ("sample_bootstrap_allele_tables", "removed in v3.0.0"),
    ],
)
def test_removed_functions_raise(name, replacement):
    with pytest.raises(NotImplementedError, match=replacement):
        getattr(data_deprecated, name)(None)


# ── Warn + delegate ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("vecs", "expected"),
    [
        ([[1, 0, 3, 4, 5], [1, -1, -1, 3, -1], [1, 2, 3, 2, -1]], [1, 0, 3, 0, 5]),
        (
            [[(1, 1), (0, 2), (3,), (4,), (5,)], [1, -1, -1, 3, -1], [1, 2, 3, 2, -1]],
            [1, 2, 3, 0, 5],
        ),
        (
            [
                [(4, 62), (3, 10), (3, 10, 16), (0, 3), (0, 2, 3), (0, 2, 3), (0, 4, 7)],
                [4, 3, -1, 0, 0, 0, (0, 7)],
            ],
            [4, 3, (3, 10, 16), 0, 0, 0, (0, 7)],
        ),
        (
            [
                [(1, 1), (0, 2), (3, 0), (4,), (5,)],
                [1, -1, -1, 3, -1],
                [1, -1, (3, 0), 2, -1],
            ],
            [1, (0, 2), (3, 0), 0, 5],
        ),
    ],
)
def test_get_lca_characters(vecs, expected):
    with pytest.warns(DeprecationWarning, match="ancestral_characters"):
        assert data_deprecated.get_lca_characters(vecs, missing_state_indicator=-1) == expected


def test_newick_to_networkx():
    with pytest.warns(DeprecationWarning, match="read_nwk"):
        g = data_deprecated.newick_to_networkx("(node1:0.2,node2:0.3)node0;")
    assert sorted(g.edges(data="length")) == [("node0", "node1", 0.2), ("node0", "node2", 0.3)]
