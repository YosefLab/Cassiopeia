import unittest

import itertools
import networkx as nx
import pandas as pd

import cassiopeia as cas
from cassiopeia.solver.SpectralGreedySolver import SpectralGreedySolver
from cassiopeia.solver.MaxCutGreedySolver import MaxCutGreedySolver
from cassiopeia.solver import solver_utilities


def find_triplet_structure(triplet, T):
    a, b, c = triplet[0], triplet[1], triplet[2]
    a_ancestors = [node for node in nx.ancestors(T, a)]
    b_ancestors = [node for node in nx.ancestors(T, b)]
    c_ancestors = [node for node in nx.ancestors(T, c)]
    ab_common = len(set(a_ancestors) & set(b_ancestors))
    ac_common = len(set(a_ancestors) & set(c_ancestors))
    bc_common = len(set(b_ancestors) & set(c_ancestors))
    structure = "-"
    if ab_common > bc_common and ab_common > ac_common:
        structure = "ab"
    elif ac_common > bc_common and ac_common > ab_common:
        structure = "ac"
    elif bc_common > ab_common and bc_common > ac_common:
        structure = "bc"
    return structure


class GreedyVariantsTest(unittest.TestCase):
    def test_spectral_sparse_case(self):
        cm = pd.DataFrame.from_dict(
            {
                "c1": [5, 3, 0, 0, 0],
                "c2": [0, 3, 4, 2, 1],
                "c3": [5, 0, 0, 0, 1],
                "c4": [5, 0, 4, 2, 0],
            },
            orient="index",
            columns=["a", "b", "c", "d", "e"],
        )

        sg_tree = cas.data.CassiopeiaTree(cm, missing_state_indicator=-1)

        sgsolver = SpectralGreedySolver()

        character_matrix = sg_tree.character_matrix.copy()
        unique_cm = character_matrix.drop_duplicates()

        left, right = sgsolver.perform_split(unique_cm, unique_cm.index)
        self.assertListEqual(left, ["c1", "c3", "c4"])
        self.assertListEqual(right, ["c2"])

        sgsolver.solve(sg_tree, collapse_mutationless_edges=True)
        expected_tree = nx.DiGraph()
        expected_tree.add_edges_from(
            [
                (4, "c1"),
                (4, "c3"),
                (4, "c4"),
                (5, 4),
                (5, "c2"),
            ]
        )
        observed_tree = sg_tree.get_tree_topology()
        triplets = itertools.combinations(["c1", "c2", "c3", "c4"], 3)
        for triplet in triplets:
            expected_triplet = find_triplet_structure(triplet, expected_tree)
            observed_triplet = find_triplet_structure(triplet, observed_tree)
            self.assertEqual(expected_triplet, observed_triplet)

        sgsolver.solve(sg_tree, collapse_mutationless_edges=True)
        expected_tree = nx.DiGraph()
        expected_tree.add_edges_from(
            [
                (4, "c1"),
                (4, "c3"),
                (4, "c4"),
                (5, 4),
                (5, "c2"),
            ]
        )
        observed_tree = sg_tree.get_tree_topology()
        triplets = itertools.combinations(["c1", "c2", "c3", "c4"], 3)
        for triplet in triplets:
            expected_triplet = find_triplet_structure(triplet, expected_tree)
            observed_triplet = find_triplet_structure(triplet, observed_tree)
            self.assertEqual(expected_triplet, observed_triplet)

        sgsolver.solve(sg_tree)
        observed_tree = sg_tree.get_tree_topology()
        for triplet in triplets:
            expected_triplet = find_triplet_structure(triplet, expected_tree)
            observed_triplet = find_triplet_structure(triplet, observed_tree)
            self.assertEqual(expected_triplet, observed_triplet)

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

        sgsolver.solve(sg_tree, collapse_mutationless_edges=True)
        expected_tree = nx.DiGraph()
        expected_tree.add_edges_from(
            [
                (4, "c3"),
                (4, "c4"),
                (4, "c5"),
                (5, 4),
                (5, "c1"),
                (6, "c2"),
                (6, "c6"),
                (6, 7),
            ]
        )
        observed_tree = sg_tree.get_tree_topology()
        triplets = itertools.combinations(
            ["c1", "c2", "c3", "c4", "c5", "c6"], 3
        )
        for triplet in triplets:
            expected_triplet = find_triplet_structure(triplet, expected_tree)
            observed_triplet = find_triplet_structure(triplet, observed_tree)
            self.assertEqual(expected_triplet, observed_triplet)

        sgsolver.solve(sg_tree, collapse_mutationless_edges=True)
        observed_tree = sg_tree.get_tree_topology()
        for triplet in triplets:
            expected_triplet = find_triplet_structure(triplet, expected_tree)
            observed_triplet = find_triplet_structure(triplet, observed_tree)
            self.assertEqual(expected_triplet, observed_triplet)

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

        sg_tree = cas.data.CassiopeiaTree(
            cm, missing_state_indicator=-1, priors=priors
        )

        weights = solver_utilities.transform_priors(priors, "negative_log")
        sgsolver = SpectralGreedySolver()
        unique_cm = cm.drop_duplicates()
        left, right = sgsolver.perform_split(
            unique_cm, unique_cm.index, weights=weights
        )
        self.assertEqual(left, ["c2", "c6"])
        self.assertEqual(right, ["c1", "c3"])

        sgsolver.solve(sg_tree, collapse_mutationless_edges=True)
        expected_tree = nx.DiGraph()
        expected_tree.add_edges_from(
            [
                (4, "c3"),
                (4, "c4"),
                (4, "c5"),
                (5, 4),
                (5, "c1"),
                (6, "c2"),
                (6, "c6"),
                (6, 7),
            ]
        )
        observed_tree = sg_tree.get_tree_topology()
        triplets = itertools.combinations(
            ["c1", "c2", "c3", "c4", "c5", "c6"], 3
        )
        for triplet in triplets:
            expected_triplet = find_triplet_structure(triplet, expected_tree)
            observed_triplet = find_triplet_structure(triplet, observed_tree)
            self.assertEqual(expected_triplet, observed_triplet)

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

        mcgsolver.solve(mcg_tree, collapse_mutationless_edges=True)
        expected_tree = nx.DiGraph()
        expected_tree.add_edges_from(
            [
                (5, "c1"),
                (5, "c2"),
                (5, "c3"),
                (5, 4),
                (4, "c4"),
                (4, "c5"),
            ]
        )
        observed_tree = mcg_tree.get_tree_topology()
        triplets = itertools.combinations(["c1", "c2", "c3", "c4"], 3)
        for triplet in triplets:
            expected_triplet = find_triplet_structure(triplet, expected_tree)
            observed_triplet = find_triplet_structure(triplet, observed_tree)
            self.assertEqual(expected_triplet, observed_triplet)

        mcgsolver.solve(mcg_tree)
        observed_tree = mcg_tree.get_tree_topology()
        for triplet in triplets:
            expected_triplet = find_triplet_structure(triplet, expected_tree)
            observed_triplet = find_triplet_structure(triplet, observed_tree)
            self.assertEqual(expected_triplet, observed_triplet)

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

        weights = solver_utilities.transform_priors(priors, "negative_log")

        mcg_tree = cas.data.CassiopeiaTree(
            cm, missing_state_indicator=-1, priors=priors
        )

        mcgsolver = MaxCutGreedySolver()
        unique_cm = cm.drop_duplicates()
        left, right = mcgsolver.perform_split(
            unique_cm, unique_cm.index, weights=weights
        )
        self.assertListEqual(left, ["c1", "c3", "c4", "c2"])
        self.assertListEqual(right, [])

        mcgsolver.solve(mcg_tree, collapse_mutationless_edges=True)
        expected_tree = nx.DiGraph()
        expected_tree.add_edges_from(
            [
                (5, "c1"),
                (5, "c2"),
                (5, "c3"),
                (5, 4),
                (4, "c4"),
                (4, "c5"),
            ]
        )
        observed_tree = mcg_tree.get_tree_topology()
        triplets = itertools.combinations(["c1", "c2", "c3", "c4"], 3)
        for triplet in triplets:
            expected_triplet = find_triplet_structure(triplet, expected_tree)
            observed_triplet = find_triplet_structure(triplet, observed_tree)
            self.assertEqual(expected_triplet, observed_triplet)


if __name__ == "__main__":
    unittest.main()
