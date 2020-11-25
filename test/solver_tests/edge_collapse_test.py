"""
Tests for edge collapsing functionality in cassiopeia.solver.solver_utilities.py
"""

import unittest

import ete3
import networkx as nx
import pandas as pd

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

    def test_basic_unifurcation_collapsing(self):

        T = nx.DiGraph()
        T.add_edges_from([(0, 1), (0, 2), (2, 3), (3, 4), (3, 5)])

        tree = ete3.Tree(to_newick_with_internal(T), format=1)

        collapsed_tree = solver_utilities.collapse_unifurcations(tree)

        # make sure all leaves remain
        self.assertEqual(len(tree), len(collapsed_tree))
        for n in tree:
            self.assertIn(n.name, collapsed_tree.get_leaf_names())

        # make sure there are no singletons left
        for n in collapsed_tree.traverse():
            self.assertFalse(len(n.children) == 1)

        # make sure 0 is connected to 3 now
        children_of_root = [n.name for n in collapsed_tree.children]
        self.assertIn("3", children_of_root)

    def test_longer_caterpillar_tree_unifurcation_collapsing(self):

        T = nx.DiGraph()
        T.add_edges_from(
            [(0, 1), (0, 2), (2, 3), (3, 4), (4, 5), (5, 6), (5, 7)]
        )

        tree = ete3.Tree(to_newick_with_internal(T), format=1)

        collapsed_tree = solver_utilities.collapse_unifurcations(tree)

        # make sure all leaves remain
        self.assertEqual(len(tree), len(collapsed_tree))
        for n in tree:
            self.assertIn(n.name, collapsed_tree.get_leaf_names())

        # make sure there are no singletons left
        for n in collapsed_tree.traverse():
            self.assertFalse(len(n.children) == 1)

        self.assertEqual((collapsed_tree & "5").up.name, "0")


if __name__ == "__main__":
    unittest.main()
