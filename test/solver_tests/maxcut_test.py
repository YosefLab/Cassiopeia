import unittest

import networkx as nx
import pandas as pd

from cassiopeia.solver.MaxCutSolver import MaxCutSolver
from cassiopeia.solver import graph_utilities
from cassiopeia.solver import solver_utilities

class MaxCutSolverTest(unittest.TestCase):
    def test_graph_construction(self):
        # A case where samples 1, 2, 3 cannot be resolved, so they are returned
        # as a polytomy
        cm = pd.DataFrame(
            [
                ["5", "0", "1"],
                ["5", "4", "0"],
                ["4", "4", "3"],
                ["-", "4", "0"]
            ]
        )
        mcsolver = MaxCutSolver(character_matrix=cm, missing_char="-")
        mcsolver.solve()
        self.assertIn((4, 0), mcsolver.tree.edges)
        self.assertIn((4, 5), mcsolver.tree.edges)
        self.assertIn((5, 1), mcsolver.tree.edges)
        self.assertIn((5, 2), mcsolver.tree.edges)
        self.assertIn((5, 3), mcsolver.tree.edges)

        mutation_frequencies = mcsolver.compute_mutation_frequencies(range(cm.shape[0]))
        G = graph_utilities.construct_connectivity_graph(cm, mutation_frequencies, "-")

        self.assertEqual(G[0][1]['weight'], -1)
        self.assertEqual(G[0][2]['weight'], 3)
        self.assertEqual(G[0][3]['weight'], 2)
        self.assertEqual(G[1][2]['weight'], -2)
        self.assertEqual(G[1][3]['weight'], -3)
        self.assertEqual(G[2][3]['weight'], -3)

        cut_weight = mcsolver.evaluate_cut([1, 2], G)
        self.assertEqual(cut_weight, -4)


    def test_hill_climb(self):
        G = nx.DiGraph()
        for i in range(4):
            G.add_node(i)
        G.add_edge(0, 1, weight = 3)
        G.add_edge(0, 2, weight = 2)
        G.add_edge(0, 3, weight = 2)
        G.add_edge(1, 2, weight = 2)
        G.add_edge(1, 3, weight = 10)
        G.add_edge(2, 3, weight = 1)

        new_cut = graph_utilities.max_cut_improve_cut(G, [0, 2])
        self.assertEqual(new_cut, [0, 1])
    


if __name__ == "__main__":
    unittest.main()


