import unittest

import networkx as nx
import pandas as pd

from cassiopeia.solver.SpectralSolver import SpectralSolver
from cassiopeia.solver import graph_utilities
from cassiopeia.solver import solver_utilities


class SpectralSolverTest(unittest.TestCase):
    def test_similarity(self):
        cm = pd.DataFrame(
            [
                ["5", "-", "1", "2", "0"],
                ["5", "4", "0", "2", "1"],
                ["4", "4", "3", "1", "1"],
                ["-", "4", "0", "1", "-"],
            ]
        )

        self.assertEqual(graph_utilities.similarity(0, 2, cm, "-"), 0)
        self.assertEqual(graph_utilities.similarity(1, 2, cm, "-"), 2)

    def test_graph_construction(self):
        cm = pd.DataFrame(
            [
                ["5", "-", "1", "2", "0"],
                ["5", "4", "0", "2", "1"],
                ["4", "4", "3", "1", "1"],
                ["-", "4", "0", "1", "-"],
            ]
        )
        spsolver = SpectralSolver(character_matrix=cm, missing_char="-")

        mutation_frequencies = spsolver.compute_mutation_frequencies(
            range(cm.shape[0])
        )
        G = graph_utilities.construct_similarity_graph(
            cm, mutation_frequencies, "-"
        )

        self.assertEqual(G[0][1]["weight"], 1)
        self.assertEqual(G[1][2]["weight"], 1)
        self.assertEqual(G[2][3]["weight"], 1)
        self.assertNotIn([0, 2], G.edges)

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
        cm = pd.DataFrame(
            [
                ["2", "-", "3", "-", "0"],
                ["2", "3", "3", "0", "0"],
                ["3", "4", "3", "0", "0"],
                ["4", "-", "3", "-", "0"],
                ["4", "2", "3", "0", "0"],
                ["5", "1", "-", "0", "1"],
                ["1", "-", "3", "1", "2"],
            ]
        )
        spsolver = SpectralSolver(character_matrix=cm, missing_char="-")
        spsolver.solve()
        expected_newick_string = "(2,(0,1),(5,6),(3,4));"
        observed_newick_string = solver_utilities.to_newick(spsolver.tree)
        self.assertEqual(expected_newick_string, observed_newick_string)


if __name__ == "__main__":
    unittest.main()
