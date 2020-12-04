"""
Tests for the cassiopeia.critique.compare module.
"""

import unittest

import networkx as nx
import numpy as np
import pandas as pd

import cassiopeia as cas


class TestResolveUMISequence(unittest.TestCase):
    def setUp(self):

        self.ground_truth_tree = nx.balanced_tree(2, 3, create_using=nx.DiGraph)

        self.tree1 = nx.DiGraph()
        self.tree1.add_nodes_from([i for i in range(15)])
        self.tree1.add_edges_from(
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

        # create tests with some unresolvable triplets

        self.multifurcating_ground_truth = nx.DiGraph()
        self.multifurcating_ground_truth.add_edges_from(
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

        self.tree2 = nx.DiGraph()
        self.tree2.add_edges_from(
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

        self.ground_truth_rake = nx.DiGraph()
        self.ground_truth_rake.add_edges_from(
            [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6)]
        )

    def test_same_tree_gives_perfect_triplets_correct(self):

        (
            all_triplets,
            resolvable_triplets_correct,
            unresolved_triplets_correct,
            proportion_unresolvable,
        ) = cas.critique.triplets_correct(
            self.ground_truth_tree, self.ground_truth_tree, number_of_trials=10
        )

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
        ) = cas.critique.triplets_correct(
            self.ground_truth_tree, self.tree1, number_of_trials=10
        )

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
        expected_unresolvable_triplets = (
            prob_of_sampling_left * prob_of_sampling_unresolvable_from_left
        )

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
        ) = cas.critique.triplets_correct(
            self.multifurcating_ground_truth, self.tree2, number_of_trials=1000
        )

        self.assertEqual(all_triplets[0], 1.0)

        # we only changed triplets on the resolvable part of the tree so make
        # sure that the unresolvable triplets correct is 1.0
        for depth in unresolved_triplets_correct.keys():
            self.assertEqual(unresolved_triplets_correct[depth], 1.0)

        # all of the triplets on the right side of the tree at depth 1 are
        # incorrect, so the overall triplets correct should be just the prob.
        # of sampling the left
        prob_of_sampling_left = 0.833
        self.assertAlmostEqual(
            all_triplets[1], prob_of_sampling_left, delta=0.05
        )

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

    def test_robinson_foulds_same_tree_bifurcating(self):

        rf, max_rf = cas.critique.robinson_foulds(
            self.ground_truth_tree, self.ground_truth_tree
        )

        self.assertEqual(rf, 0)
        self.assertEqual(max_rf, 10)


if __name__ == "__main__":
    unittest.main()
