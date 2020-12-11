import unittest

import networkx as nx
import pandas as pd

from cassiopeia.solver.MaxCutSolver import MaxCutSolver
from cassiopeia.solver import graph_utilities
from cassiopeia.solver import solver_utilities


class MaxCutSolverTest(unittest.TestCase):
    def setUp(self):
        self.cm = pd.DataFrame(
            [["5", "0", "1"], ["5", "4", "0"], ["4", "4", "3"], ["-", "4", "0"]]
        )
        self.mcsolver = MaxCutSolver(character_matrix=self.cm, missing_char="-")
        self.mutation_frequencies = self.mcsolver.compute_mutation_frequencies(
            range(self.cm.shape[0])
        )

    def test_check_if_cut(self):
        self.assertTrue(graph_utilities.check_if_cut(2, 3, [0, 1, 2]))
        self.assertFalse(graph_utilities.check_if_cut(1, 2, [0, 1, 2]))

    def test_evaluate_cut(self):
        G = graph_utilities.construct_connectivity_graph(
            self.cm,
            self.mutation_frequencies,
            "-",
            list(range(self.cm.shape[0])),
        )
        cut_weight = self.mcsolver.evaluate_cut([1, 2], G)
        self.assertEqual(cut_weight, -4)

    def test_graph_construction(self):
        G = graph_utilities.construct_connectivity_graph(
            self.cm,
            self.mutation_frequencies,
            "-",
            list(range(self.cm.shape[0])),
        )

        self.assertEqual(G[0][1]["weight"], -1)
        self.assertEqual(G[0][2]["weight"], 3)
        self.assertEqual(G[0][3]["weight"], 2)
        self.assertEqual(G[1][2]["weight"], -2)
        self.assertEqual(G[1][3]["weight"], -3)
        self.assertEqual(G[2][3]["weight"], -3)

    def test_graph_construction_weights(self):
        weights = {0: {"4": 1, "5": 2}, 1: {"4": 2}, 2: {"1": 1, "3": 1}}

        G = graph_utilities.construct_connectivity_graph(
            self.cm,
            self.mutation_frequencies,
            "-",
            list(range(self.cm.shape[0])),
            w=weights,
        )

        self.assertEqual(G[0][1]["weight"], -2)
        self.assertEqual(G[0][2]["weight"], 6)
        self.assertEqual(G[0][3]["weight"], 4)
        self.assertEqual(G[1][2]["weight"], -4)
        self.assertEqual(G[1][3]["weight"], -6)
        self.assertEqual(G[2][3]["weight"], -6)

    def test_hill_climb(self):
        G = nx.DiGraph()
        for i in range(4):
            G.add_node(i)
        G.add_edge(0, 1, weight=3)
        G.add_edge(0, 2, weight=2)
        G.add_edge(0, 3, weight=2)
        G.add_edge(1, 2, weight=2)
        G.add_edge(1, 3, weight=10)
        G.add_edge(2, 3, weight=1)

        new_cut = graph_utilities.max_cut_improve_cut(G, [0, 2])
        self.assertEqual(new_cut, [0, 1])

    def test_simple_base_case(self):
        # A case where samples 1, 2, 3 cannot be resolved, so they are returned
        # as a polytomy
        self.mcsolver.solve()
        expected_newick_string = "(0,(1,2,3));"
        observed_newick_string = solver_utilities.to_newick(self.mcsolver.tree)
        self.assertEqual(expected_newick_string, observed_newick_string)


if __name__ == "__main__":
    unittest.main()
