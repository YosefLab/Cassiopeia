"""
Tests for the cassiopeia.critique.compare module.
"""

import networkx as nx
import pytest
from treedata import TreeData

import cassiopeia as cas

# ---------- Fixtures ----------


@pytest.fixture
def ground_truth_tree():
    g = nx.balanced_tree(2, 3, create_using=nx.DiGraph)
    g = nx.relabel_nodes(g, lambda x: str(x))
    return g


@pytest.fixture
def tree1():
    g = nx.DiGraph()
    g.add_nodes_from(list(range(15)))
    g.add_edges_from(
        [
            (0, 1),
            (0, 2),
            (1, 3),
            (1, 4),
            (2, 5),
            (2, 6),
            (3, 9),
            (3, 8),
            (4, 7),
            (4, 10),
            (5, 13),
            (5, 12),
            (6, 14),
            (6, 11),
        ]
    )
    g = nx.relabel_nodes(g, lambda x: str(x))
    return g


@pytest.fixture
def multifurcating_ground_truth():
    g = nx.DiGraph()
    g.add_edges_from(
        [
            (0, 1),
            (0, 2),
            (1, 3),
            (1, 4),
            (1, 5),
            (3, 8),
            (3, 9),
            (4, 10),
            (4, 11),
            (5, 12),
            (5, 13),
            (2, 6),
            (2, 7),
            (6, 14),
            (6, 15),
            (7, 16),
            (7, 17),
        ]
    )
    g = nx.relabel_nodes(g, lambda x: str(x))
    return g


@pytest.fixture
def tree2():
    g = nx.DiGraph()
    g.add_edges_from(
        [
            (0, 1),
            (0, 2),
            (1, 3),
            (1, 4),
            (1, 5),
            (3, 8),
            (3, 9),
            (4, 10),
            (4, 11),
            (5, 12),
            (5, 13),
            (2, 6),
            (2, 7),
            (6, 14),
            (6, 17),
            (7, 16),
            (7, 15),
        ]
    )
    g = nx.relabel_nodes(g, lambda x: str(x))
    return g


@pytest.fixture
def ground_truth_rake():
    g = nx.DiGraph()
    g.add_edges_from([(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6)])
    g = nx.relabel_nodes(g, lambda x: str(x))
    return g


@pytest.fixture
def tdata(ground_truth_tree, tree1, multifurcating_ground_truth, tree2, ground_truth_rake):
    return TreeData(
        obst={
            "ground_truth": ground_truth_tree,
            "tree1": tree1,
            "multifurcating": multifurcating_ground_truth,
            "tree2": tree2,
            "rake": ground_truth_rake,
        },
    )


# ---------- Tests ----------
@pytest.mark.skip
def test_out_group(tree1, ground_truth_rake):
    out_group = cas.critique.critique_utilities.get_outgroup(tree1, ("11", "14", "9"))
    assert out_group == "9"

    out_group = cas.critique.critique_utilities.get_outgroup(ground_truth_rake, ("4", "5", "6"))
    assert out_group == "None"


@pytest.mark.skip
def test_same_tree_gives_perfect_triplets_correct(ground_truth_tree):
    (
        all_triplets,
        resolvable_triplets_correct,
        unresolved_triplets_correct,
        proportion_unresolvable,
    ) = cas.critique.triplets_correct(ground_truth_tree, ground_truth_tree, number_of_trials=10)

    for depth in all_triplets.keys():
        assert all_triplets[depth] == 1.0
    for depth in resolvable_triplets_correct.keys():
        assert resolvable_triplets_correct[depth] == 1.0
    for depth in proportion_unresolvable.keys():
        assert proportion_unresolvable[depth] == 0.0


@pytest.mark.skip
def test_triplets_correct_different_trees(ground_truth_tree, tree1):
    (
        all_triplets,
        resolvable_triplets_correct,
        unresolved_triplets_correct,
        proportion_unresolvable,
    ) = cas.critique.triplets_correct(ground_truth_tree, tree1, number_of_trials=10)

    assert all_triplets[0] == 1.0
    assert all_triplets[1] == 0.0
    assert proportion_unresolvable[0] == 0.0
    assert proportion_unresolvable[1] == 0.0


@pytest.mark.skip
def test_triplets_correct_multifurcating_same_tree(multifurcating_ground_truth):
    (
        all_triplets,
        resolvable_triplets_correct,
        unresolved_triplets_correct,
        proportion_unresolvable,
    ) = cas.critique.triplets_correct(
        multifurcating_ground_truth,
        multifurcating_ground_truth,
        number_of_trials=1000,
    )

    for depth in all_triplets.keys():
        assert all_triplets[depth] == 1.0
        assert unresolved_triplets_correct[depth] == 1.0

    prob_of_sampling_left = 0.833333
    prob_of_sampling_unresolvable_from_left = 0.4
    expected_unresolvable_triplets = prob_of_sampling_left * prob_of_sampling_unresolvable_from_left

    assert proportion_unresolvable[0] == 0
    assert pytest.approx(proportion_unresolvable[1], abs=0.05) == expected_unresolvable_triplets


@pytest.mark.skip
def test_triplets_correct_multifurcating_different_trees(multifurcating_ground_truth, tree2):
    (
        all_triplets,
        resolvable_triplets_correct,
        unresolved_triplets_correct,
        proportion_unresolvable,
    ) = cas.critique.triplets_correct(multifurcating_ground_truth, tree2, number_of_trials=1000)

    assert all_triplets[0] == 1.0
    for depth in unresolved_triplets_correct.keys():
        assert unresolved_triplets_correct[depth] == 1.0

    prob_of_sampling_left = 0.833
    assert pytest.approx(all_triplets[1], abs=0.05) == prob_of_sampling_left


@pytest.mark.skip
def test_rake_tree(ground_truth_rake):
    (
        all_triplets,
        resolvable_triplets_correct,
        unresolved_triplets_correct,
        proportion_unresolvable,
    ) = cas.critique.triplets_correct(
        ground_truth_rake,
        ground_truth_rake,
        number_of_trials=1000,
    )
    assert all_triplets[0] == 1.0
    assert unresolved_triplets_correct[0] == 1.0
    assert proportion_unresolvable[0] == 1.0


def test_robinson_foulds_bifurcating_same_tree(ground_truth_tree):
    rf, max_rf = cas.critique.robinson_foulds(ground_truth_tree, ground_truth_tree)
    assert rf == 0
    assert max_rf == 10


def test_robinson_foulds_different_trees_bifurcating(ground_truth_tree, tree1):
    rf, max_rf = cas.critique.robinson_foulds(ground_truth_tree, tree1)
    assert rf == 8
    assert max_rf == 10


def test_robinson_foulds_different_trees_multifurcating(tree2, multifurcating_ground_truth):
    rf, max_rf = cas.critique.robinson_foulds(tree2, multifurcating_ground_truth)
    assert rf == 4
    assert max_rf == 12


def test_robinson_foulds_same_tree_multifurcating(multifurcating_ground_truth):
    rf, max_rf = cas.critique.robinson_foulds(
        multifurcating_ground_truth, multifurcating_ground_truth
    )
    assert rf == 0
    assert max_rf == 12


def test_robinson_foulds_different_leaf_sets_error(ground_truth_tree):
    different_leaves_tree = nx.DiGraph()
    different_leaves_tree.add_edges_from([(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)])

    with pytest.raises(ValueError) as exc:
        cas.critique.robinson_foulds(ground_truth_tree, different_leaves_tree)
    assert "identical leaf sets" in str(exc.value)


def test_robinson_foulds_with_string_keys(tdata):
    rf, max_rf = cas.critique.robinson_foulds(tdata, key1="ground_truth", key2="ground_truth")
    assert rf == 0
    assert max_rf > 0


def test_robinson_foulds_mixed_types(tdata, ground_truth_tree):
    rf, max_rf = cas.critique.robinson_foulds(
        tdata,
        ground_truth_tree,
        key1="ground_truth",
    )
    assert rf == 0
    assert max_rf > 0

    def test_triplets_correct_different_trees_bifurcating(self):
        all_tc, res_tc, unres_tc, prop_unres = cas.critique.triplets_correct(
            self.ground_truth_tree, self.tree1
        )
        # Not a perfect match across all depths
        self.assertTrue(any(v < 1.0 for v in all_tc.values()))
        # Sanity: values are within [0,1]
        self.assertTrue(all(0.0 <= v <= 1.0 for v in all_tc.values()))
        self.assertTrue(all(0.0 <= v <= 1.0 for v in res_tc.values()))
        self.assertTrue(all(0.0 <= v <= 1.0 for v in unres_tc.values()))
        self.assertTrue(all(0.0 <= v <= 1.0 for v in prop_unres.values()))

    def test_triplets_correct_different_trees_multifurcating(self):
        all_tc, res_tc, unres_tc, prop_unres = cas.critique.triplets_correct(
            self.tree2, self.multifurcating_ground_truth
        )
        # Multifurcations → some unresolved; not all depths perfect
        self.assertTrue(any(v < 1.0 for v in all_tc.values()))
        # Sanity bounds
        self.assertTrue(all(0.0 <= v <= 1.0 for v in all_tc.values()))
        self.assertTrue(all(0.0 <= v <= 1.0 for v in res_tc.values()))
        self.assertTrue(all(0.0 <= v <= 1.0 for v in unres_tc.values()))
        self.assertTrue(all(0.0 <= v <= 1.0 for v in prop_unres.values()))

    def test_triplets_correct_same_tree_multifurcating(self):
        all_tc, res_tc, unres_tc, prop_unres = cas.critique.triplets_correct(
            self.multifurcating_ground_truth, self.multifurcating_ground_truth
        )
        # Identical trees → perfect agreement at every depth
        self.assertTrue(len(all_tc) > 0)
        self.assertTrue(all(v == 1.0 for v in all_tc.values()))
        # Sanity bounds on other outputs
        self.assertTrue(all(0.0 <= v <= 1.0 for v in res_tc.values()))
        self.assertTrue(all(0.0 <= v <= 1.0 for v in unres_tc.values()))
        self.assertTrue(all(0.0 <= v <= 1.0 for v in prop_unres.values()))

    def test_triplets_correct_different_leaf_sets_error(self):
        different_leaves_tree = nx.DiGraph()
        different_leaves_tree.add_edges_from([(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)])

        with self.assertRaises(ValueError) as context:
            cas.critique.triplets_correct(
                self.ground_truth_tree.get_tree_topology(), different_leaves_tree
            )
        self.assertIn("identical leaf sets", str(context.exception))

    def test_triplets_correct_with_nx_digraph(self):
        tree1_graph = self.ground_truth_tree.get_tree_topology()
        tree2_graph = self.ground_truth_tree.get_tree_topology()

        all_tc, res_tc, unres_tc, prop_unres = cas.critique.triplets_correct(
            tree1_graph, tree2_graph
        )
        self.assertTrue(len(all_tc) > 0)
        self.assertTrue(all(v == 1.0 for v in all_tc.values()))

    def test_triplets_correct_with_string_keys(self):
        all_tc, res_tc, unres_tc, prop_unres = cas.critique.triplets_correct(
            "ground_truth", "ground_truth", tdata=self.tdata
        )
        self.assertTrue(len(all_tc) > 0)
        self.assertTrue(all(v == 1.0 for v in all_tc.values()))

    def test_triplets_correct_type_mismatch_error(self):
        with self.assertRaises(TypeError) as context:
            cas.critique.triplets_correct(
                self.ground_truth_tree, self.ground_truth_tree.get_tree_topology()
            )
        self.assertIn("must be the same type", str(context.exception))

    def test_triplets_correct_string_without_tdata_error(self):
        with self.assertRaises(ValueError) as context:
            cas.critique.triplets_correct("tree1", "tree2")
        self.assertIn("tdata must be provided", str(context.exception))

    def test_triplets_correct_string_with_missing_key_error(self):
        with self.assertRaises(ValueError) as context:
            cas.critique.triplets_correct("tree1", "nonexistent_tree", tdata=self.tdata)
        self.assertIn("Tree keys must exist in tdata.obst", str(context.exception))

    def test_triplets_correct_wrong_type(self):
        with self.assertRaises(TypeError) as context:
            cas.critique.triplets_correct(["unsupported_list"], ["unsupported_list"])
        self.assertIn("Unsupported", str(context.exception))

    def test_triplets_correct_missing_tdata_obst(self):
        with self.assertRaises(ValueError) as context:
            cas.critique.triplets_correct("tree1", "tree2", tdata=self.emptytdata)
        self.assertIn("does not have an 'obst' attribute", str(context.exception))


if __name__ == "__main__":
    pytest.main(["-v", __file__])
