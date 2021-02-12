import unittest

import networkx as nx
import pandas as pd

import cassiopeia as cas
from cassiopeia.data import utilities as tree_utilities
from cassiopeia.solver.SpectralGreedySolver import SpectralGreedySolver
from cassiopeia.solver.MaxCutGreedySolver import MaxCutGreedySolver
from cassiopeia.solver import graph_utilities
from cassiopeia.solver import solver_utilities


class GreedyVariantsTest(unittest.TestCase):
    def test_spectral_sparse_case(self):
        cm = pd.DataFrame(
            [
                [5, 3, 0, 0, 0],
                [0, 3, 4, 2, 1],
                [5, 0, 0, 0, 1],
                [5, 0, 4, 2, 0],
            ]
        )

        sg_tree = cas.data.CassiopeiaTree(cm, missing_state_indicator=-1)

        sgsolver = SpectralGreedySolver()

        character_matrix = sg_tree.get_original_character_matrix()
        unique_character_matrix = character_matrix.drop_duplicates()

        left, right = sgsolver.perform_split(unique_character_matrix, list(range(4)))
        self.assertListEqual(left, [0, 2, 3])
        self.assertListEqual(right, [1])

        sgsolver.solve(sg_tree)
        expected_newick_string = "((0,3,2),1);"
        observed_newick_string = sg_tree.get_newick()
        self.assertEqual(expected_newick_string, observed_newick_string)

    def test_spectral_base_case(self):
        cm = pd.DataFrame.from_dict(
            {
                "c1": [5, 3, 0, 4, 0],
                "c2": [5, 3, 0, 2, 1],
                "c3": [5, 0, 0, 4, 1],
                "c4": [5, 0, 0, 4, 1],
                "c5": [5, 0, 0, 4, 1],
                "c6": [5, 3, 0, 2, 0],
            },
            orient="index",
            columns=["a", "b", "c", "d", "e"],
        )

        sg_tree = cas.data.CassiopeiaTree(cm, missing_state_indicator=-1)
        sgsolver = SpectralGreedySolver()

        unique_cm = cm.drop_duplicates()
        left, right = sgsolver.perform_split(unique_cm, unique_cm.index)
        self.assertEqual(left, ["c2", "c6"])
        self.assertEqual(right, ["c1", "c3"])

        sgsolver.solve(sg_tree)
        expected_newick_string = "((c2,c6),(c1,(c3,c4,c5)));"
        observed_newick_string = sg_tree.get_newick()
        self.assertEqual(expected_newick_string, observed_newick_string)

    def test_spectral_base_case_weights_almost_one(self):
        cm = pd.DataFrame.from_dict(
            {
                "c1": [5, 3, 0, 4, 0],
                "c2": [5, 3, 0, 2, 1],
                "c3": [5, 0, 0, 4, 1],
                "c4": [5, 0, 0, 4, 1],
                "c5": [5, 0, 0, 4, 1],
                "c6": [5, 3, 0, 2, 0],
            },
            orient="index",
            columns=["a", "b", "c", "d", "e"],
        )

        priors = {
            0: {5: 0.367879},
            1: {3: 0.367879},
            2: {},
            3: {2: 0.367879, 4: 0.367879},
            4: {1: 0.367879},
        }

        sg_tree = cas.data.CassiopeiaTree(cm, missing_state_indicator=-1, priors=priors)

        weights = solver_utilities.transform_priors(priors, 'negative_log')
        sgsolver = SpectralGreedySolver()
        unique_cm = cm.drop_duplicates()
        left, right = sgsolver.perform_split(unique_cm, unique_cm.index, weights=weights)
        self.assertEqual(left, ["c2", "c6"])
        self.assertEqual(right, ["c1", "c3"])

        sgsolver.solve(sg_tree)
        expected_newick_string = "((c2,c6),(c1,(c3,c4,c5)));"
        observed_newick_string = sg_tree.get_newick()
        self.assertEqual(expected_newick_string, observed_newick_string)

    def test_maxcut_base_case(self):
        # A case in which the connectivity only has negative weights, so the
        # hill-climbing procedure favors a cut with 0 weight
        cm = pd.DataFrame.from_dict(
            {
                "c1": [5, 3, 0, 0, 0],
                "c2": [0, 3, 4, 2, 1],
                "c3": [5, 0, 0, 0, 1],
                "c4": [5, 0, 4, 2, 0],
                "c5": [5, 0, 4, 2, 0],
            },
            orient="index",
            columns=["a", "b", "c", "d", "e"],
        )

        mcg_tree = cas.data.CassiopeiaTree(cm, missing_state_indicator=-1)

        mcgsolver = MaxCutGreedySolver()
        unique_cm = cm.drop_duplicates()
        left, right = mcgsolver.perform_split(unique_cm, unique_cm.index)
        self.assertListEqual(left, ["c1", "c3", "c4", "c2"])
        self.assertListEqual(right, [])

        mcgsolver.solve(mcg_tree)
        expected_newick_string = "(c1,c3,c2,(c4,c5));"
        observed_newick_string = mcg_tree.get_newick()
        self.assertEqual(expected_newick_string, observed_newick_string)

    def test_maxcut_base_case_weights_trivial(self):
        # A case in which the connectivity only has negative weights, so the
        # hill-climbing procedure favors a cut with 0 weight
        cm = pd.DataFrame.from_dict(
            {
                "c1": [5, 3, 0, 0, 0],
                "c2": [0, 3, 4, 2, 1],
                "c3": [5, 0, 0, 0, 1],
                "c4": [5, 0, 4, 2, 0],
                "c5": [5, 0, 4, 2, 0],
            },
            orient="index",
            columns=["a", "b", "c", "d", "e"],
        )

        priors = {
            0: {5: 0.5},
            1: {3: 0.5},
            2: {4: 0.5},
            3: {2: 0.5},
            4: {1: 0.5},
        }

        weights = solver_utilities.transform_priors(priors, 'negative_log')

        mcg_tree = cas.data.CassiopeiaTree(cm, missing_state_indicator=-1, priors=priors)

        mcgsolver = MaxCutGreedySolver()
        unique_cm = cm.drop_duplicates()
        left, right = mcgsolver.perform_split(unique_cm, unique_cm.index, weights=weights)
        self.assertListEqual(left, ["c1", "c3", "c4", "c2"])
        self.assertListEqual(right, [])

        mcgsolver.solve(mcg_tree)
        expected_newick_string = "(c1,c3,c2,(c4,c5));"
        observed_newick_string = mcg_tree.get_newick()
        self.assertEqual(expected_newick_string, observed_newick_string)


if __name__ == "__main__":
    unittest.main()
