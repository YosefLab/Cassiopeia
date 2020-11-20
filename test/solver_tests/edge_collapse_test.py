"""
Tests for edge collapsing functionality in cassiopeia.solver.solver_utilities.py
"""

import unittest

import networkx as nx
import pandas as pd

from cassiopeia.solver import solver_utilities


class InferAncestorError(Exception):
    """An Exception class for solver utilities."""

    pass


class TestCollapseEdges(unittest.TestCase):
    def test1(self):
        T = nx.DiGraph()
        for i in range(7):
            T.add_node(i)
        T.add_edge(4, 0)
        T.add_edge(4, 1)
        T.add_edge(5, 2)
        T.add_edge(5, 3)
        T.add_edge(6, 4)
        T.add_edge(6, 5)
        table = [
            ["1", "0", "3", "4", "5"],
            ["1", "0", "3", "3", "-"],
            ["1", "2", "3", "0", "-"],
            ["1", "0", "3", "0", "-"],
        ]
        cm = pd.DataFrame(table)
        solver_utilities.collapse_tree(
            T,
            infer_ancestral_characters=True,
            character_matrix=cm,
            missing_char="-",
        )
        new_map = {}
        for i in T:
            new_map[i] = (
                "|".join([str(c) for c in T.nodes[i]["characters"]]) + f",{i}"
            )
        T = nx.relabel_nodes(T, new_map)

        expected_nodes = {
            "1|0|3|4|5,0",
            "1|0|3|3|-,1",
            "1|2|3|0|-,2",
            "1|0|3|0|-,3",
            "1|0|3|0|-,5",
            "1|0|3|0|5,6",
        }

        expected_edges = {
            ("1|0|3|0|5,6", "1|0|3|4|5,0"),
            ("1|0|3|0|5,6", "1|0|3|3|-,1"),
            ("1|0|3|0|-,5", "1|0|3|0|-,3"),
            ("1|0|3|0|-,5", "1|2|3|0|-,2"),
            ("1|0|3|0|5,6", "1|0|3|0|-,5"),
        }

        for i in T:
            self.assertIn(i, expected_nodes)

        for i in T.edges():
            self.assertIn(i, expected_edges)

    def test2(self):
        T = nx.DiGraph()
        for i in range(7):
            T.add_node(i)
        T.add_edge(4, 0)
        T.add_edge(4, 1)
        T.add_edge(5, 3)
        T.add_edge(5, 4)
        T.add_edge(6, 5)
        T.add_edge(6, 2)
        table = [
            ["1", "0", "3", "4", "5"],
            ["1", "0", "3", "3", "-"],
            ["1", "2", "3", "0", "-"],
            ["1", "0", "3", "0", "-"],
        ]
        cm = pd.DataFrame(table)
        solver_utilities.collapse_tree(
            T,
            infer_ancestral_characters=True,
            character_matrix=cm,
            missing_char="-",
        )
        new_map = {}
        for i in T:
            new_map[i] = (
                "|".join([str(c) for c in T.nodes[i]["characters"]]) + f",{i}"
            )
        T = nx.relabel_nodes(T, new_map)

        expected_nodes = {
            "1|0|3|4|5,0",
            "1|0|3|3|-,1",
            "1|2|3|0|-,2",
            "1|0|3|0|-,3",
            "1|0|3|0|5,6",
        }

        expected_edges = {
            ("1|0|3|0|5,6", "1|0|3|4|5,0"),
            ("1|0|3|0|5,6", "1|0|3|3|-,1"),
            ("1|0|3|0|5,6", "1|2|3|0|-,2"),
            ("1|0|3|0|5,6", "1|0|3|0|-,3"),
        }

        for i in T:
            self.assertIn(i, expected_nodes)

        for i in T.edges():
            self.assertIn(i, expected_edges)

    def test_newick_converter(self):
        T = nx.DiGraph()
        for i in range(7):
            T.add_node(i)
        T.add_edge(4, 0)
        T.add_edge(4, 1)
        T.add_edge(5, 3)
        T.add_edge(5, 4)
        T.add_edge(6, 5)
        T.add_edge(6, 2)

        expected_newick_string = "((3,(0,1)),2);"
        observed_newick_string = solver_utilities.to_newick(T)

        self.assertEqual(expected_newick_string, observed_newick_string)


if __name__ == "__main__":
    unittest.main()
