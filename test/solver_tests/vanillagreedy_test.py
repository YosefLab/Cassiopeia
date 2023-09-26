import unittest

import itertools
import networkx as nx
import pandas as pd

import cassiopeia as cas
from cassiopeia.mixins import GreedySolverError
from cassiopeia.solver.VanillaGreedySolver import VanillaGreedySolver
from cassiopeia.solver import missing_data_methods
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


class VanillaGreedySolverTest(unittest.TestCase):
    def test_raises_error_on_ambiguous(self):
        cm = pd.DataFrame.from_dict(
            {
                "c1": [5, (0, 1), 1, 2, -1],
                "c2": [0, 0, 3, 2, -1],
                "c3": [-1, 4, 0, 2, 2],
                "c4": [4, 4, 1, 2, 0],
            },
            orient="index",
            columns=["a", "b", "c", "d", "e"],
        )

        tree = cas.data.CassiopeiaTree(cm, missing_state_indicator=-1)
        with self.assertRaises(GreedySolverError):
            solver = VanillaGreedySolver()
            solver.solve(tree)

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
        unique_character_matrix = vg_tree.character_matrix.drop_duplicates()
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

        unique_character_matrix = vg_tree.character_matrix.drop_duplicates()
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

    def test_ambiguous_freq_dict(self):
        cm = pd.DataFrame.from_dict(
            {
                "c1": [5, (0, 1), 1, 2, -1],
                "c2": [0, 0, 3, 2, -1],
                "c3": [-1, 4, 0, (2, 3), 2],
                "c4": [4, 4, 1, 2, 0],
            },
            orient="index",
            columns=["a", "b", "c", "d", "e"],
        )

        vg_tree = cas.data.CassiopeiaTree(cm, missing_state_indicator=-1)

        vgsolver = VanillaGreedySolver()
        unique_character_matrix = vg_tree.character_matrix.drop_duplicates()
        freq_dict = vgsolver.compute_mutation_frequencies(
            ["c1", "c2", "c3", "c4"],
            unique_character_matrix,
            vg_tree.missing_state_indicator,
        )

        self.assertEqual(len(freq_dict), 5)
        self.assertEqual(len(freq_dict[0]), 4)
        self.assertEqual(len(freq_dict[1]), 4)
        self.assertEqual(len(freq_dict[2]), 4)
        self.assertEqual(len(freq_dict[3]), 3)
        self.assertEqual(len(freq_dict[4]), 3)
        self.assertEqual(freq_dict[0][5], 1)
        self.assertEqual(freq_dict[1][0], 2)
        self.assertEqual(freq_dict[1][1], 1)
        self.assertEqual(freq_dict[3][3], 1)
        self.assertEqual(freq_dict[2][-1], 0)
        self.assertNotIn(3, freq_dict[1].keys())

    def test_ambiguous_duplicate_freq_dict(self):
        cm = pd.DataFrame.from_dict(
            {
                "c1": [5, (0, 1), 1, 2, -1],
                "c1_dup": [5, (1, 0), 1, 2, -1],
                "c2": [0, 0, 3, 2, -1],
                "c3": [-1, 4, 0, (2, 3), 2],
                "c4": [4, 4, 1, 2, 0],
            },
            orient="index",
            columns=["a", "b", "c", "d", "e"],
        )

        vg_tree = cas.data.CassiopeiaTree(cm, missing_state_indicator=-1)

        vgsolver = VanillaGreedySolver()
        keep_rows = (
            cm.apply(
                lambda x: [
                    set(s) if type(s) == tuple else set([s]) for s in x.values
                ],
                axis=0,
            )
            .apply(tuple, axis=1)
            .drop_duplicates()
            .index.values
        )
        unique_character_matrix = cm.loc[keep_rows].copy()

        freq_dict = vgsolver.compute_mutation_frequencies(
            unique_character_matrix.index,
            unique_character_matrix,
            vg_tree.missing_state_indicator,
        )

        self.assertEqual(len(freq_dict), 5)
        self.assertEqual(len(freq_dict[0]), 4)
        self.assertEqual(len(freq_dict[1]), 4)
        self.assertEqual(len(freq_dict[2]), 4)
        self.assertEqual(len(freq_dict[3]), 3)
        self.assertEqual(len(freq_dict[4]), 3)
        self.assertEqual(freq_dict[0][5], 1)
        self.assertEqual(freq_dict[1][0], 2)
        self.assertEqual(freq_dict[1][1], 1)
        self.assertEqual(freq_dict[3][3], 1)
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
        cm = pd.DataFrame.from_dict(
            {
                "c1": [5, 0, 1, 2, 0],
                "c2": [5, 0, 1, 2, 0],
                "c3": [5, 0, 1, 2, 0],
            },
            orient="index",
            columns=["a", "b", "c", "d", "e"],
        )

        vg_tree = cas.data.CassiopeiaTree(cm, missing_state_indicator=-1)
        vgsolver = VanillaGreedySolver()

        vgsolver.solve(vg_tree, collapse_mutationless_edges=True)
        expected_tree = nx.DiGraph()
        expected_tree.add_edges_from([(3, "c1"), (3, "c3"), (3, "c2")])
        observed_tree = vg_tree.get_tree_topology()
        triplets = itertools.combinations(["c1", "c2", "c3"], 3)
        for triplet in triplets:
            expected_triplet = find_triplet_structure(triplet, expected_tree)
            observed_triplet = find_triplet_structure(triplet, observed_tree)
            self.assertEqual(expected_triplet, observed_triplet)

        vg_tree = cas.data.CassiopeiaTree(cm, missing_state_indicator=-1)
        vgsolver.solve(vg_tree)
        observed_tree = vg_tree.get_tree_topology()
        for triplet in triplets:
            expected_triplet = find_triplet_structure(triplet, expected_tree)
            observed_triplet = find_triplet_structure(triplet, observed_tree)
            self.assertEqual(expected_triplet, observed_triplet)

    def test_case_1(self):
        cm = pd.DataFrame.from_dict(
            {
                "c1": [5, 0, 1, 2, 0],
                "c2": [5, 0, 0, 2, -1],
                "c3": [4, 0, 3, 2, -1],
                "c4": [-1, 4, 0, 2, 2],
                "c5": [0, 4, 1, 2, 2],
                "c6": [4, 0, 0, 2, 2],
            },
            orient="index",
            columns=["a", "b", "c", "d", "e"],
        )

        vg_tree = cas.data.CassiopeiaTree(cm, missing_state_indicator=-1)
        vgsolver = VanillaGreedySolver()

        unique_character_matrix = vg_tree.character_matrix.drop_duplicates()

        left, right = vgsolver.perform_split(
            unique_character_matrix, unique_character_matrix.index
        )

        self.assertListEqual(left, ["c4", "c5", "c6", "c3"])
        self.assertListEqual(right, ["c1", "c2"])

        vgsolver.solve(vg_tree, collapse_mutationless_edges=True)
        expected_tree = nx.DiGraph()
        expected_tree.add_edges_from(
            [
                (6, "c1"),
                (6, "c2"),
                (7, "c4"),
                (7, "c5"),
                (8, "c3"),
                (8, "c6"),
                (9, 7),
                (9, 8),
                (10, 6),
                (10, 9),
            ]
        )
        observed_tree = vg_tree.get_tree_topology()
        triplets = itertools.combinations(
            ["c1", "c2", "c3", "c4", "c5", "c6"], 3
        )
        for triplet in triplets:
            expected_triplet = find_triplet_structure(triplet, expected_tree)
            observed_triplet = find_triplet_structure(triplet, observed_tree)
            self.assertEqual(expected_triplet, observed_triplet)

        vgsolver.solve(vg_tree)
        expected_tree = nx.DiGraph()
        expected_tree.add_edges_from(
            [
                (6, "c1"),
                (6, "c2"),
                (7, "c4"),
                (7, "c5"),
                (8, "c3"),
                (8, "c6"),
                (9, 7),
                (9, 8),
                (10, 6),
                (10, 9),
            ]
        )
        observed_tree = vg_tree.get_tree_topology()
        for triplet in triplets:
            expected_triplet = find_triplet_structure(triplet, expected_tree)
            observed_triplet = find_triplet_structure(triplet, observed_tree)
            self.assertEqual(expected_triplet, observed_triplet)

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

        vgsolver.solve(vg_tree, collapse_mutationless_edges=True)
        expected_tree = nx.DiGraph()
        expected_tree.add_edges_from(
            [
                (6, "c1"),
                (6, "c2"),
                (7, "c3"),
                (7, "c4"),
                (8, "c5"),
                (8, "c6"),
                (8, "c7"),
                (9, 7),
                (9, 8),
                (10, 6),
                (10, 9),
            ]
        )
        observed_tree = vg_tree.get_tree_topology()
        triplets = itertools.combinations(
            ["c1", "c2", "c3", "c4", "c5", "c6", "c7"], 3
        )
        for triplet in triplets:
            expected_triplet = find_triplet_structure(triplet, expected_tree)
            observed_triplet = find_triplet_structure(triplet, observed_tree)
            self.assertEqual(expected_triplet, observed_triplet)

        vgsolver.solve(vg_tree)
        expected_tree = nx.DiGraph()
        expected_tree.add_edges_from(
            [
                (6, "c1"),
                (6, "c2"),
                (7, "c3"),
                (7, "c4"),
                (8, "c5"),
                (8, "c6"),
                (8, "c7"),
                (9, 7),
                (9, 8),
                (10, 6),
                (10, 9),
            ]
        )
        observed_tree = vg_tree.get_tree_topology()
        for triplet in triplets:
            expected_triplet = find_triplet_structure(triplet, expected_tree)
            observed_triplet = find_triplet_structure(triplet, observed_tree)
            self.assertEqual(expected_triplet, observed_triplet)

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

        vgsolver.solve(vg_tree, collapse_mutationless_edges=True)
        expected_tree = nx.DiGraph()
        expected_tree.add_edges_from(
            [
                (6, "c1"),
                (6, "c2"),
                (7, "c3"),
                (7, "c4"),
                (8, "c5"),
                (8, "c6"),
                (8, "c7"),
                (9, 7),
                (9, 8),
                (10, 6),
                (10, 9),
            ]
        )
        observed_tree = vg_tree.get_tree_topology()
        triplets = itertools.combinations(
            ["c1", "c2", "c3", "c4", "c5", "c6", "c7"], 3
        )
        for triplet in triplets:
            expected_triplet = find_triplet_structure(triplet, expected_tree)
            observed_triplet = find_triplet_structure(triplet, observed_tree)
            self.assertEqual(expected_triplet, observed_triplet)

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

        vgsolver.solve(vg_tree, collapse_mutationless_edges=True)
        expected_tree = nx.DiGraph()
        expected_tree.add_edges_from(
            [
                (6, "c1"),
                (6, "c2"),
                (8, "c3"),
                (8, "c4"),
                (7, "c5"),
                (7, "c6"),
                (7, "c7"),
                (8, 6),
                (9, 7),
                (9, 8),
            ]
        )
        observed_tree = vg_tree.get_tree_topology()
        triplets = itertools.combinations(
            ["c1", "c2", "c3", "c4", "c5", "c6", "c7"], 3
        )

        for triplet in triplets:
            expected_triplet = find_triplet_structure(triplet, expected_tree)
            observed_triplet = find_triplet_structure(triplet, observed_tree)
            self.assertEqual(expected_triplet, observed_triplet)

        vgsolver.solve(vg_tree)
        expected_tree = nx.DiGraph()
        expected_tree.add_edges_from(
            [
                (6, "c1"),
                (6, "c2"),
                (8, "c3"),
                (8, 6),
                (7, "c5"),
                (7, "c6"),
                (7, "c7"),
                (9, 8),
                (9, "c4"),
                (10, 9),
                (10, 7),
            ]
        )
        observed_tree = vg_tree.get_tree_topology()
        for triplet in triplets:
            expected_triplet = find_triplet_structure(triplet, expected_tree)
            observed_triplet = find_triplet_structure(triplet, observed_tree)
            self.assertEqual(expected_triplet, observed_triplet)


if __name__ == "__main__":
    unittest.main()
