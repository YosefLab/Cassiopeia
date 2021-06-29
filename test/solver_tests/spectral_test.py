import unittest

import networkx as nx
import pandas as pd

import cassiopeia as cas
from cassiopeia.data import utilities as tree_utilities
from cassiopeia.solver.SpectralSolver import SpectralSolver
from cassiopeia.solver import graph_utilities
from cassiopeia.solver import dissimilarity_functions


class SpectralSolverTest(unittest.TestCase):
    def test_similarity(self):
        cm = pd.DataFrame.from_dict(
            {
                "c1": [5, -1, 1, 2, 0],
                "c2": [5, 4, 0, 2, 1],
                "c3": [4, 4, 3, 1, 1],
                "c4": [-1, 4, 0, 1, -1],
            },
            orient="index",
            columns=["a", "b", "c", "d", "e"],
        )

        self.assertEqual(
            dissimilarity_functions.hamming_similarity_without_missing(
                list(cm.loc["c1", :]), list(cm.loc["c3", :]), -1
            ),
            0,
        )
        self.assertEqual(
            dissimilarity_functions.hamming_similarity_without_missing(
                list(cm.loc["c2", :]), list(cm.loc["c3", :]), -1
            ),
            2,
        )
        self.assertEqual(
            dissimilarity_functions.hamming_similarity_without_missing(
                list(cm.loc["c3", :]), list(cm.loc["c4", :]), -1
            ),
            2,
        )

    def test_similarity_weighted(self):
        cm = pd.DataFrame.from_dict(
            {
                "c1": [5, -1, 1, 2, 0],
                "c2": [5, 4, 0, 2, 1],
                "c3": [4, 4, 3, 1, 1],
                "c4": [-1, 4, 0, 1, -1],
            },
            orient="index",
            columns=["a", "b", "c", "d", "e"],
        )

        weights = {
            0: {5: 2},
            1: {4: 2},
            2: {1: 1, 3: 1},
            3: {1: 1, 2: 1},
            4: {1: 3},
        }

        self.assertEqual(
            dissimilarity_functions.hamming_similarity_without_missing(
                list(cm.loc["c1", :]), list(cm.loc["c3", :]), -1, weights
            ),
            0,
        )
        self.assertEqual(
            dissimilarity_functions.hamming_similarity_without_missing(
                list(cm.loc["c2", :]), list(cm.loc["c3", :]), -1, weights
            ),
            5,
        )

    def test_graph_construction(self):
        cm = pd.DataFrame.from_dict(
            {
                "c1": [5, -1, 1, 2, 0],
                "c2": [5, -1, 1, 2, 0],
                "c3": [5, 4, 0, 2, 1],
                "c4": [4, 4, 3, 1, 1],
                "c5": [-1, 4, 0, 1, -1],
            },
            orient="index",
            columns=["a", "b", "c", "d", "e"],
        )

        unique_character_matrix = cm.drop_duplicates()
        G = graph_utilities.construct_similarity_graph(
            unique_character_matrix,
            -1,
            unique_character_matrix.index,
            similarity_function=dissimilarity_functions.hamming_similarity_without_missing,
        )

        self.assertEqual(G["c1"]["c3"]["weight"], 1)
        self.assertEqual(G["c3"]["c4"]["weight"], 1)
        self.assertEqual(G["c4"]["c5"]["weight"], 1)
        self.assertNotIn(["c1", "c4"], G.edges)
        self.assertNotIn(["c1", "c5"], G.edges)
        self.assertNotIn(["c3", "c5"], G.edges)

    def test_graph_construction_weighted(self):
        cm = pd.DataFrame.from_dict(
            {
                "c1": [5, -1, 1, 2, 0],
                "c2": [5, -1, 1, 2, 0],
                "c3": [5, 4, 0, 2, 1],
                "c4": [4, 4, 3, 1, 1],
                "c5": [-1, 4, 0, 1, -1],
            },
            orient="index",
            columns=["a", "b", "c", "d", "e"],
        )

        weights = {
            0: {5: 2},
            1: {4: 2},
            2: {1: 1, 3: 1},
            3: {1: 1, 2: 1},
            4: {1: 3},
        }

        unique_character_matrix = cm.drop_duplicates()

        G = graph_utilities.construct_similarity_graph(
            unique_character_matrix,
            -1,
            unique_character_matrix.index,
            similarity_function=dissimilarity_functions.hamming_similarity_without_missing,
            weights=weights,
        )

        self.assertEqual(G["c1"]["c3"]["weight"], 1)
        self.assertEqual(G["c3"]["c4"]["weight"], 3)
        self.assertEqual(G["c4"]["c5"]["weight"], 1)
        self.assertNotIn(["c1", "c4"], G.edges)
        self.assertNotIn(["c1", "c5"], G.edges)
        self.assertNotIn(["c3", "c5"], G.edges)

    def test_hill_climb(self):
        G = nx.Graph()
        for i in range(4):
            G.add_node(i)
        G.add_edge(0, 1, weight=3)
        G.add_edge(0, 2, weight=2)
        G.add_edge(0, 3, weight=2)
        G.add_edge(1, 2, weight=2)
        G.add_edge(1, 3, weight=10)
        G.add_edge(2, 3, weight=1)

        new_cut = graph_utilities.spectral_improve_cut(G, [0])
        self.assertEqual(new_cut, [0, 2])

    def test_simple_base_case(self):
        cm = pd.DataFrame.from_dict(
            {
                "c1": [2, -1, 3, -1, 0],
                "c2": [2, 3, 3, 0, 0],
                "c3": [3, 4, 3, 0, 0],
                "c4": [4, -1, 3, -1, 0],
                "c5": [4, 2, 3, 0, 0],
                "c6": [5, 1, -1, 0, 1],
                "c7": [1, -1, 3, 1, 2],
            },
            orient="index",
            columns=["a", "b", "c", "d", "e"],
        )

        sp_tree = cas.data.CassiopeiaTree(cm, missing_state_indicator=-1)

        spsolver = SpectralSolver()
        spsolver.solve(sp_tree)
        expected_newick_string = "(c3,(c1,c2),(c6,c7),(c4,c5));"
        observed_newick_string = sp_tree.get_newick()
        self.assertEqual(expected_newick_string, observed_newick_string)

    def test_simple_base_case_string(self):
        cm = pd.DataFrame.from_dict(
            {
                "c1": [2, -1, 3, -1, 0],
                "c2": [2, 3, 3, 0, 0],
                "c3": [3, 4, 3, 0, 0],
                "c4": [4, -1, 3, -1, 0],
                "c5": [4, 2, 3, 0, 0],
                "c6": [5, 1, -1, 0, 1],
                "c7": [1, -1, 3, 1, 2],
                "c8": [1, -1, 3, 1, 2],
            },
            orient="index",
            columns=["a", "b", "c", "d", "e"],
        )

        sp_tree = cas.data.CassiopeiaTree(cm, missing_state_indicator=-1)

        spsolver = SpectralSolver()
        spsolver.solve(sp_tree)
        expected_newick_string = "(c3,(c1,c2),(c6,(c7,c8)),(c4,c5));"
        observed_newick_string = sp_tree.get_newick()
        self.assertEqual(expected_newick_string, observed_newick_string)

    def test_simple_base_case2(self):
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

        sp_tree = cas.data.CassiopeiaTree(cm, missing_state_indicator=-1)

        spsolver = SpectralSolver()
        spsolver.solve(sp_tree)
        expected_newick_string = "((c1,c3),(c2,c4));"
        observed_newick_string = sp_tree.get_newick()
        self.assertEqual(expected_newick_string, observed_newick_string)

    def test_simple_base_case2_priors(self):
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

        priors = {
            0: {5: 0.8},
            1: {3: 0.5},
            2: {4: 0.5},
            3: {2: 0.5},
            4: {1: 0.1},
        }

        sp_tree = cas.data.CassiopeiaTree(
            cm, missing_state_indicator=-1, priors=priors
        )

        spsolver = SpectralSolver()
        spsolver.solve(sp_tree)
        expected_newick_string = "(c1,c2,(c3,c4));"
        observed_newick_string = sp_tree.get_newick()
        self.assertEqual(expected_newick_string, observed_newick_string)


if __name__ == "__main__":
    unittest.main()
