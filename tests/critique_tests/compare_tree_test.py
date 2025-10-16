"""
Tests for the cassiopeia.critique.compare module.
"""

import unittest

import networkx as nx
from treedata import TreeData

import cassiopeia as cas


class TestTreeComparisons(unittest.TestCase):
    def setUp(self):
        self.ground_truth_tree = cas.data.CassiopeiaTree(tree=nx.balanced_tree(2, 3, create_using=nx.DiGraph))

        tree1 = nx.DiGraph()
        tree1.add_nodes_from(list(range(15)))
        tree1.add_edges_from(
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

        self.tree1 = cas.data.CassiopeiaTree(tree=tree1)

        # create tests with some unresolvable triplets

        multifurcating_ground_truth = nx.DiGraph()
        multifurcating_ground_truth.add_edges_from(
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
        self.multifurcating_ground_truth = cas.data.CassiopeiaTree(tree=multifurcating_ground_truth)

        tree2 = nx.DiGraph()
        tree2.add_edges_from(
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
        self.tree2 = cas.data.CassiopeiaTree(tree=tree2)

        ground_truth_rake = nx.DiGraph()
        ground_truth_rake.add_edges_from([(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6)])

        self.ground_truth_rake = cas.data.CassiopeiaTree(tree=ground_truth_rake)

        self.tdata = TreeData(
            obst={
                "ground_truth": self.ground_truth_tree.get_tree_topology(),
                "tree1": tree1,
                "multifurcating": multifurcating_ground_truth,
                "tree2": tree2,
                "rake": ground_truth_rake,
            },
            alignment="subset",
        )

        self.emptytdata = TreeData(alignment="subset")

    def test_out_group(self):
        out_group = cas.critique.critique_utilities.get_outgroup(self.tree1, ("11", "14", "9"))

        self.assertEqual("9", out_group)

        out_group = cas.critique.critique_utilities.get_outgroup(self.ground_truth_rake, ("4", "5", "6"))
        self.assertEqual("None", out_group)

    def test_same_tree_gives_perfect_triplets_correct(self):
        (
            all_triplets,
            resolvable_triplets_correct,
            unresolved_triplets_correct,
            proportion_unresolvable,
        ) = cas.critique.triplets_correct(self.ground_truth_tree, self.ground_truth_tree, number_of_trials=10)

        for depth in all_triplets.keys():
            self.assertEqual(all_triplets[depth], 1.0)

        for depth in resolvable_triplets_correct.keys():
            self.assertEqual(resolvable_triplets_correct[depth], 1.0)

        for depth in proportion_unresolvable.keys():
            self.assertEqual(proportion_unresolvable[depth], 0.0)

    def test_triplets_correct_different_trees(self):
        (
            all_triplets,
            resolvable_triplets_correct,
            unresolved_triplets_correct,
            proportion_unresolvable,
        ) = cas.critique.triplets_correct(self.ground_truth_tree, self.tree1, number_of_trials=10)

        self.assertEqual(all_triplets[0], 1.0)
        self.assertEqual(all_triplets[1], 0.0)

        self.assertEqual(proportion_unresolvable[0], 0.0)
        self.assertEqual(proportion_unresolvable[1], 0.0)

    def test_triplets_correct_multifurcating_same_tree(self):
        (
            all_triplets,
            resolvable_triplets_correct,
            unresolved_triplets_correct,
            proportion_unresolvable,
        ) = cas.critique.triplets_correct(
            self.multifurcating_ground_truth,
            self.multifurcating_ground_truth,
            number_of_trials=1000,
        )

        for depth in all_triplets.keys():
            self.assertEqual(all_triplets[depth], 1.0)
            self.assertEqual(unresolved_triplets_correct[depth], 1.0)
        # expected proportion of unresolvable triplets at depth 1
        # this is simply:
        #   prob. of sampling the multifurcating child * \
        #   the prob of sampling a leaf from each of its children
        prob_of_sampling_left = 0.833333
        prob_of_sampling_unresolvable_from_left = 0.4
        expected_unresolvable_triplets = prob_of_sampling_left * prob_of_sampling_unresolvable_from_left

        self.assertEqual(proportion_unresolvable[0], 0)
        self.assertAlmostEqual(
            proportion_unresolvable[1],
            expected_unresolvable_triplets,
            delta=0.05,
        )

    def test_triplets_correct_multifurcating_different_trees(self):
        (
            all_triplets,
            resolvable_triplets_correct,
            unresolved_triplets_correct,
            proportion_unresolvable,
        ) = cas.critique.triplets_correct(self.multifurcating_ground_truth, self.tree2, number_of_trials=1000)

        self.assertEqual(all_triplets[0], 1.0)

        # we only changed triplets on the resolvable part of the tree so make
        # sure that the unresolvable triplets correct is 1.0
        for depth in unresolved_triplets_correct.keys():
            self.assertEqual(unresolved_triplets_correct[depth], 1.0)

        # all of the triplets on the right side of the tree at depth 1 are
        # incorrect, so the overall triplets correct should be just the prob.
        # of sampling the left
        prob_of_sampling_left = 0.833
        self.assertAlmostEqual(all_triplets[1], prob_of_sampling_left, delta=0.05)

    def test_rake_tree(self):
        (
            all_triplets,
            resolvable_triplets_correct,
            unresolved_triplets_correct,
            proportion_unresolvable,
        ) = cas.critique.triplets_correct(
            self.ground_truth_rake,
            self.ground_truth_rake,
            number_of_trials=1000,
        )

        self.assertEqual(all_triplets[0], 1.0)
        self.assertEqual(unresolved_triplets_correct[0], 1.0)
        self.assertEqual(proportion_unresolvable[0], 1.0)

    def test_robinson_foulds_bifurcating_same_tree(self):
        rf, max_rf = cas.critique.robinson_foulds(self.ground_truth_tree, self.ground_truth_tree)

        self.assertEqual(rf, 0)
        self.assertEqual(max_rf, 10)

    def test_robinson_foulds_different_trees_bifurcating(self):
        rf, max_rf = cas.critique.robinson_foulds(self.ground_truth_tree, self.tree1)
        self.assertEqual(rf, 8)
        self.assertEqual(max_rf, 10)

    def test_robinson_foulds_different_trees_multifurcating(self):
        rf, max_rf = cas.critique.robinson_foulds(self.tree2, self.multifurcating_ground_truth)
        self.assertEqual(rf, 4)
        self.assertEqual(max_rf, 12)

    def test_robinson_foulds_same_tree_multifurcating(self):
        rf, max_rf = cas.critique.robinson_foulds(self.multifurcating_ground_truth, self.multifurcating_ground_truth)
        self.assertEqual(rf, 0)
        self.assertEqual(max_rf, 12)

    def test_robinson_foulds_different_leaf_sets_error(self):
        different_leaves_tree = nx.DiGraph()
        different_leaves_tree.add_edges_from([(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)])

        with self.assertRaises(ValueError) as context:
            cas.critique.robinson_foulds(self.ground_truth_tree.get_tree_topology(), different_leaves_tree)

        self.assertIn("identical leaf sets", str(context.exception))

    def test_robinson_foulds_with_nx_digraph(self):
        tree1_graph = self.ground_truth_tree.get_tree_topology()
        tree2_graph = self.ground_truth_tree.get_tree_topology()

        rf, max_rf = cas.critique.robinson_foulds(tree1_graph, tree2_graph)

        self.assertEqual(rf, 0)
        self.assertGreater(max_rf, 0)

    def test_robinson_foulds_with_string_keys(self):
        rf, max_rf = cas.critique.robinson_foulds("ground_truth", "ground_truth", tdata=self.tdata)

        self.assertEqual(rf, 0)
        self.assertGreater(max_rf, 0)

    def test_robinson_foulds_type_mismatch_error(self):
        with self.assertRaises(TypeError) as context:
            cas.critique.robinson_foulds(self.ground_truth_tree, self.ground_truth_tree.get_tree_topology())

        self.assertIn("must be the same type", str(context.exception))

    def test_robinson_foulds_string_without_tdata_error(self):
        with self.assertRaises(ValueError) as context:
            cas.critique.robinson_foulds("tree1", "tree2")

        self.assertIn("tdata must be provided", str(context.exception))

    def test_robinson_foulds_string_with_missing_key_error(self):
        with self.assertRaises(ValueError) as context:
            cas.critique.robinson_foulds("tree1", "nonexistent_tree", tdata=self.tdata)

        self.assertIn("Tree keys must exist in tdata.obst", str(context.exception))

    def test_robinson_foulds_wrong_type(self):
        with self.assertRaises(TypeError) as context:
            cas.critique.robinson_foulds(["unsupported_list"], ["unsupported_list"])

        self.assertIn("Unsupported tree type", str(context.exception))

    def test_robinson_foulds_missing_tdata_obst(self):
        with self.assertRaises(ValueError) as context:
            cas.critique.robinson_foulds("tree1", "tree2", tdata=self.emptytdata)

        self.assertIn("does not have an 'obst' attribute.", str(context.exception))


if __name__ == "__main__":
    unittest.main()
