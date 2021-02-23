import unittest

import networkx as nx
import numpy as np
import pandas as pd

import cassiopeia as cas
from cassiopeia.solver.VanillaGreedySolver import VanillaGreedySolver
from cassiopeia.solver import missing_data_methods
from cassiopeia.data import utilities as tree_utilities
from cassiopeia.solver import solver_utilities


class VanillaGreedySolverTest(unittest.TestCase):
    def test_basic_freq_dict(self):
        cm = pd.DataFrame.from_dict(
            {
                "c1": [5, 0, 1, 2, -1],
                "c2": [0, 0, 3, 2, -1],
                "c3": [-1, 4, 0, 2, 2],
                "c4": [4, 4, 1, 2, 0],
            },
            orient="index",
            columns=["a", "b", "c", "d", "e"],
        )

        vg_tree = cas.data.CassiopeiaTree(cm, missing_state_indicator=-1)

        vgsolver = VanillaGreedySolver()
        unique_character_matrix = (
            vg_tree.get_original_character_matrix().drop_duplicates()
        )
        freq_dict = vgsolver.compute_mutation_frequencies(
            ["c1", "c2", "c3", "c4"],
            unique_character_matrix,
            vg_tree.missing_state_indicator,
        )

        self.assertEqual(len(freq_dict), 5)
        self.assertEqual(len(freq_dict[0]), 4)
        self.assertEqual(len(freq_dict[1]), 3)
        self.assertEqual(len(freq_dict[2]), 4)
        self.assertEqual(len(freq_dict[3]), 2)
        self.assertEqual(len(freq_dict[4]), 3)
        self.assertEqual(freq_dict[0][5], 1)
        self.assertEqual(freq_dict[1][0], 2)
        self.assertEqual(freq_dict[2][-1], 0)
        self.assertNotIn(3, freq_dict[1].keys())

    def test_duplicate_freq_dict(self):
        cm = pd.DataFrame.from_dict(
            {
                "c1": [5, 0, 1, 2, -1],
                "c2": [5, 0, 1, 2, -1],
                "c3": [0, 0, 3, 2, -1],
                "c4": [-1, 4, 0, 2, 2],
                "c5": [4, 4, 1, 2, 0],
            },
            orient="index",
            columns=["a", "b", "c", "d", "e"],
        )

        vg_tree = cas.data.CassiopeiaTree(cm, missing_state_indicator=-1)

        vgsolver = VanillaGreedySolver()

        unique_character_matrix = (
            vg_tree.get_original_character_matrix().drop_duplicates()
        )
        freq_dict = vgsolver.compute_mutation_frequencies(
            ["c1", "c3", "c4", "c5"],
            unique_character_matrix,
            vg_tree.missing_state_indicator,
        )

        self.assertEqual(len(freq_dict), 5)
        self.assertEqual(len(freq_dict[0]), 4)
        self.assertEqual(len(freq_dict[1]), 3)
        self.assertEqual(len(freq_dict[2]), 4)
        self.assertEqual(len(freq_dict[3]), 2)
        self.assertEqual(len(freq_dict[4]), 3)
        self.assertEqual(freq_dict[0][5], 1)
        self.assertEqual(freq_dict[1][0], 2)
        self.assertEqual(freq_dict[2][-1], 0)
        self.assertNotIn(3, freq_dict[1].keys())

    def test_average_missing_data(self):
        cm = pd.DataFrame.from_dict(
            {
                "c1": [-1, 4, 0, 2, 2],
                "c2": [4, 4, 1, 2, 0],
                "c3": [4, 0, 3, -1, -1],
                "c4": [5, 0, 1, 2, -1],
                "c5": [5, 0, 1, 2, -1],
            },
            orient="index",
            columns=["a", "b", "c", "d", "e"],
        )

        left_set, right_set = missing_data_methods.assign_missing_average(
            cm, -1, ["c1", "c2"], ["c4", "c5"], ["c3"]
        )
        self.assertEqual(left_set, ["c1", "c2", "c3"])
        self.assertEqual(right_set, ["c4", "c5"])

    def test_average_missing_data_priors(self):
        cm = pd.DataFrame.from_dict(
            {
                "c1": [-1, 4, 0, 2, 2],
                "c2": [4, 4, 0, 2, 0],
                "c3": [4, 0, 1, -1, -1],
                "c4": [5, 0, 1, 2, -1],
                "c5": [5, 0, 1, 2, -1],
            },
            orient="index",
            columns=["a", "b", "c", "d", "e"],
        )
        priors = {
            0: {4: 0.5, 5: 0.5},
            1: {4: 1},
            2: {1: 1},
            3: {2: 1},
            4: {2: 1},
        }

        weights = solver_utilities.transform_priors(priors)

        left_set, right_set = missing_data_methods.assign_missing_average(
            cm, -1, ["c1", "c2"], ["c4", "c5"], ["c3"], weights
        )
        self.assertEqual(left_set, ["c1", "c2", "c3"])
        self.assertEqual(right_set, ["c4", "c5"])

    def test_all_duplicates_base_case(self):
        cm = pd.DataFrame(
            [
                [5, 0, 1, 2, 0],
                [5, 0, 1, 2, 0],
                [5, 0, 1, 2, 0],
            ]
        )

        vg_tree = cas.data.CassiopeiaTree(cm, missing_state_indicator=-1)
        vgsolver = VanillaGreedySolver()
        vgsolver.solve(vg_tree)
        expected_newick_string = "(0,1,2);"
        observed_newick_string = vg_tree.get_newick()
        self.assertEqual(expected_newick_string, observed_newick_string)

    def test_case_1(self):
        cm = pd.DataFrame(
            [
                [5, 0, 1, 2, 0],
                [5, 0, 0, 2, -1],
                [4, 0, 3, 2, -1],
                [-1, 4, 0, 2, 2],
                [0, 4, 1, 2, 2],
                [4, 0, 0, 2, 2],
            ]
        )

        vg_tree = cas.data.CassiopeiaTree(cm, missing_state_indicator=-1)

        vgsolver = VanillaGreedySolver()

        unique_character_matrix = (
            vg_tree.get_original_character_matrix().drop_duplicates()
        )

        left, right = vgsolver.perform_split(
            unique_character_matrix, list(range(6))
        )

        self.assertListEqual(left, [3, 4, 5, 2])
        self.assertListEqual(right, [0, 1])

        vgsolver.solve(vg_tree)
        expected_newick_string = "(((2,5),(4,3)),(0,1));"
        observed_newick_string = vg_tree.get_newick()
        self.assertEqual(expected_newick_string, observed_newick_string)

    def test_case_2(self):

        cm = pd.DataFrame.from_dict(
            {
                "c1": [0, 0, 1, 2, 0],
                "c2": [0, 0, 1, 2, 0],
                "c3": [1, 2, 0, 2, -1],
                "c4": [1, 2, 3, 2, -1],
                "c5": [1, 0, 3, 4, 5],
                "c6": [1, 0, -1, 4, 5],
                "c7": [1, 0, -1, -1, 5],
            },
            orient="index",
            columns=["a", "b", "c", "d", "e"],
        )

        vg_tree = cas.data.CassiopeiaTree(cm, missing_state_indicator=-1)
        vgsolver = VanillaGreedySolver()

        vgsolver.solve(vg_tree)
        expected_newick_string = "(((c4,c3),(c5,c6,c7)),(c1,c2));"
        observed_newick_string = vg_tree.get_newick()
        self.assertEqual(expected_newick_string, observed_newick_string)

    def test_weighted_case_trivial(self):
        cm = pd.DataFrame.from_dict(
            {
                "c1": [0, 0, 1, 2, 0],
                "c2": [0, 0, 1, 2, 0],
                "c3": [1, 2, 0, 2, -1],
                "c4": [1, 2, 3, 2, -1],
                "c5": [1, 0, 3, 4, 5],
                "c6": [1, 0, -1, 4, 5],
                "c7": [1, 0, -1, -1, 5],
            },
            orient="index",
            columns=["a", "b", "c", "d", "e"],
        )

        priors = {
            0: {1: 0.5},
            1: {2: 0.5},
            2: {1: 0.5, 3: 0.5},
            3: {2: 0.5, 4: 0.5},
            4: {5: 0.5},
        }

        vg_tree = cas.data.CassiopeiaTree(
            cm, missing_state_indicator=-1, priors=priors
        )

        vgsolver = VanillaGreedySolver()

        vgsolver.solve(vg_tree)
        expected_newick_string = "(((c4,c3),(c5,c6,c7)),(c1,c2));"
        observed_newick_string = vg_tree.get_newick()
        self.assertEqual(expected_newick_string, observed_newick_string)

    def test_priors_case(self):
        cm = pd.DataFrame.from_dict(
            {
                "c1": [0, 0, 1, 2, 0],
                "c2": [0, 0, 1, 2, 0],
                "c3": [1, 2, 0, 2, -1],
                "c4": [1, 2, 3, 2, -1],
                "c5": [1, 0, 3, 4, 5],
                "c6": [1, 0, -1, 4, 5],
                "c7": [1, 0, -1, -1, 5],
            },
            orient="index",
            columns=["a", "b", "c", "d", "e"],
        )

        priors = {
            0: {1: 0.99, 2: 0.01},
            1: {2: 0.99, 3: 0.01},
            2: {1: 0.8, 3: 0.2},
            3: {2: 0.9, 4: 0.1},
            4: {5: 0.99, 6: 0.01},
        }

        vg_tree = cas.data.CassiopeiaTree(
            cm, missing_state_indicator=-1, priors=priors
        )
        vgsolver = VanillaGreedySolver()

        vgsolver.solve(vg_tree)
        expected_newick_string = "((c5,c6,c7),(c4,c3,(c1,c2)));"
        observed_newick_string = vg_tree.get_newick()
        self.assertEqual(expected_newick_string, observed_newick_string)


if __name__ == "__main__":
    unittest.main()
