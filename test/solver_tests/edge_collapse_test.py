"""
Tests for edge collapsing functionality in cassiopeia.solver.solver_utilities.py
"""

import unittest

import ete3
import networkx as nx
import pandas as pd

import cassiopeia as cas
from cassiopeia.data import utilities as data_utilities
from cassiopeia.solver import solver_utilities


class InferAncestorError(Exception):
    """An Exception class for solver utilities."""

    pass


def to_newick_with_internal(tree: nx.DiGraph) -> str:
    """Converts a networkx graph to a newick string.

    Args:
        tree: A networkx tree

    Returns:
        A newick string representing the topology of the tree
    """

    def _to_newick_str(g, node):
        is_leaf = g.out_degree(node) == 0
        _name = str(node)
        return (
            "%s" % (_name,)
            if is_leaf
            else (
                "("
                + ",".join(
                    _to_newick_str(g, child) for child in g.successors(node)
                )
                + ")"
            )
            + _name
        )

    root = [node for node in tree if tree.in_degree(node) == 0][0]
    return _to_newick_str(tree, root) + ";"


class TestCollapseEdges(unittest.TestCase):
    def test_lca_characters(self):
        vecs = [[1, 0, 3, 4, 5], [1, -1, -1, 3, -1], [1, 2, 3, 2, -1]]
        ret_vec = data_utilities.get_lca_characters(
            vecs, missing_state_indicator=-1
        )
        self.assertEqual(ret_vec, [1, 0, 3, 0, 5])

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
            [1, 0, 3, 4, 5],
            [1, 0, 3, 3, -1],
            [1, 2, 3, 0, -1],
            [1, 0, 3, 0, -1],
        ]
        character_matrix = pd.DataFrame(table)
        solver_utilities.collapse_tree(
            T,
            infer_ancestral_characters=True,
            character_matrix=character_matrix,
            missing_state_indicator=-1,
        )
        new_map = {}
        for i in T:
            new_map[i] = (
                "|".join([str(c) for c in T.nodes[i]["characters"]]) + f",{i}"
            )
        T = nx.relabel_nodes(T, new_map)

        expected_nodes = {
            "1|0|3|4|5,0",
            "1|0|3|3|-1,1",
            "1|2|3|0|-1,2",
            "1|0|3|0|-1,3",
            "1|0|3|0|-1,5",
            "1|0|3|0|5,6",
        }

        expected_edges = {
            ("1|0|3|0|5,6", "1|0|3|4|5,0"),
            ("1|0|3|0|5,6", "1|0|3|3|-1,1"),
            ("1|0|3|0|-1,5", "1|0|3|0|-1,3"),
            ("1|0|3|0|-1,5", "1|2|3|0|-1,2"),
            ("1|0|3|0|5,6", "1|0|3|0|-1,5"),
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
        T.add_edge(7, 6)
        table = [
            [1, 0, 3, 4, 5],
            [1, 0, 3, 3, -1],
            [1, 2, 3, 0, -1],
            [1, 0, 3, 0, -1],
        ]
        character_matrix = pd.DataFrame(table)
        solver_utilities.collapse_tree(
            T,
            infer_ancestral_characters=True,
            character_matrix=character_matrix,
            missing_state_indicator=-1,
        )
        new_map = {}
        for i in T:
            new_map[i] = (
                "|".join([str(c) for c in T.nodes[i]["characters"]]) + f",{i}"
            )
        T = nx.relabel_nodes(T, new_map)

        expected_nodes = {
            "1|0|3|4|5,0",
            "1|0|3|3|-1,1",
            "1|2|3|0|-1,2",
            "1|0|3|0|-1,3",
            "1|0|3|0|5,7",
        }

        expected_edges = {
            ("1|0|3|0|5,7", "1|0|3|4|5,0"),
            ("1|0|3|0|5,7", "1|0|3|3|-1,1"),
            ("1|0|3|0|5,7", "1|2|3|0|-1,2"),
            ("1|0|3|0|5,7", "1|0|3|0|-1,3"),
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
        observed_newick_string = cas.data.to_newick(T)

        self.assertEqual(expected_newick_string, observed_newick_string)


if __name__ == "__main__":
    unittest.main()
