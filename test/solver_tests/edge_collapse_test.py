import unittest

import networkx as nx
import pandas as pd

from cassiopeia.solver import solver_utilities


class InferAncestorError(Exception):
    """An Exception class for solver utilities."""

    pass


def collapse_tree(
    T: nx.DiGraph,
    infer_ancestral_characters: bool,
    cm: pd.DataFrame = None,
    missing_char: str = None,
):
    """Collapses mutationless edges in a tree in-place.

    Uses the internal node annotations of a tree to collapse edges with no
    mutations. Either takes in a tree with internal node annotations or
    a tree without annotations and infers the annotations bottom-up from the
    samples obeying Camin-Sokal Parsimony. If ground truth internal annotations
    exist, it is suggested that they are used directly and that the annotations
    are not inferred again using the parsimony method.

    Args:
        network: A networkx DiGraph object representing the tree

    Returns:
        None, operates on the tree destructively

    """
    leaves = [n for n in T if T.out_degree(n) == 0 and T.in_degree(n) == 1]
    root = [n for n in T if T.in_degree(n) == 0][0]
    char_map = {}

    # Populates the internal annotations using either the ground truth
    # annotations, or infers them
    if infer_ancestral_characters:
        if cm is None or missing_char is None:
            raise InferAncestorError()

        for i in leaves:
            char_map[i] = list(cm.iloc[i, :])
        solver_utilities.annotate_ancestral_characters(
            T, root, char_map, missing_char
        )
    else:
        for i in T.nodes():
            char_map[i] = i.char_vec

    # Calls helper function on root, passing in the mapping dictionary
    solver_utilities.collapse_edges(T, root, char_map)
    return char_map


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
        char_map = collapse_tree(T, True, cm, "-")
        new_map = {}
        for i in T:
            new_map[i] = "|".join([str(c) for c in char_map[i]]) + "," + str(i)
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
        char_map = collapse_tree(T, True, cm, "-")
        new_map = {}
        for i in T:
            new_map[i] = "|".join([str(c) for c in char_map[i]]) + "," + str(i)
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


if __name__ == "__main__":
    unittest.main()
