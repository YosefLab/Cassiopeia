import unittest

import networkx as nx
import pandas as pd

import cassiopeia.solver.solver_utilities as s_utils


def collapse_tree(
    T: nx.DiGraph,
    infer_ancestral_characters: bool,
    cm: pd.DataFrame = None,
    missing_char: str = None,
):
    leaves = [n for n in T if T.out_degree(n) == 0 and T.in_degree(n) == 1]
    root = [n for n in T if T.in_degree(n) == 0][0]
    char_map = {}

    # Populates the internal annotations using either the ground truth
    # annotations, or infers them
    if s_utils.infer_ancestral_characters:
        for i in leaves:
            char_map[i] = list(cm.iloc[i, :])
        s_utils.infer_ancestral_characters(T, root, char_map, missing_char)
    else:
        for i in T.nodes():
            char_map[i] = i.char_vec

    # Calls helper function on root, passing in the mapping dictionary
    s_utils.collapse_edges(T, root, char_map)
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
            [0, 0, 1, 2, 3],
            [0, 0, 1, 2, 3],
            [0, 0, 1, 0, 3],
            [0, 0, 1, 0, 3],
        ]
        cm = pd.DataFrame(table)
        char_map = collapse_tree(T, True, cm)
        new_map = {}
        for i in T:
            new_map[i] = "|".join([str(c) for c in char_map[i]])
        T = nx.relabel_nodes(T, new_map)

        expected_nodes = {"0|0|1|2|3", "0|0|1|0|3"}

        expected_edges = {("0|0|1|0|3", "0|0|1|2|3")}

        for i in T:
            self.assertIn(i, expected_nodes)

        for i in T.edges():
            self.assertIn(i, expected_edges)

    def test2(self):
        T = nx.DiGraph()
        for i in range(9):
            T.add_node(i)
        T.add_edge(5, 0)
        T.add_edge(5, 1)
        T.add_edge(6, 5)
        T.add_edge(6, 2)
        T.add_edge(7, 3)
        T.add_edge(7, 4)
        T.add_edge(8, 6)
        T.add_edge(8, 7)
        table = [
            [0, 0, 1, 2, 3],
            [0, 0, 1, 2, 2],
            [0, 0, 1, 2, 0],
            [0, 0, 1, 2, 2],
            [0, 0, 1, 2, 1],
        ]
        cm = pd.DataFrame(table)
        char_map = collapse_tree(T, True, cm)
        new_map = {}
        for i in T:
            new_map[i] = "|".join([str(c) for c in char_map[i]])
        T = nx.relabel_nodes(T, new_map)

        expected_nodes = {"0|0|1|2|0", "0|0|1|2|1", "0|0|1|2|2", "0|0|1|2|3"}

        expected_edges = {
            ("0|0|1|2|0", "0|0|1|2|1"),
            ("0|0|1|2|0", "0|0|1|2|2"),
            ("0|0|1|2|0", "0|0|1|2|3"),
        }

        for i in T:
            self.assertIn(i, expected_nodes)

        for i in T.edges():
            self.assertIn(i, expected_edges)


if __name__ == "__main__":
    unittest.main()
