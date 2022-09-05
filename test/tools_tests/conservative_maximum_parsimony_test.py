import unittest

import networkx as nx
import pandas as pd

from cassiopeia.data import CassiopeiaTree
from cassiopeia.tools import conservative_maximum_parsimony


class Test_conservative_maximum_parsimony(unittest.TestCase):
    def test_rooted_binary_tree(self):
        tree = nx.DiGraph()
        tree.add_nodes_from([str(i) for i in range(16)])
        tree.add_edges_from(
            [("0", "1")]
            + [(str(i), str(2 * i)) for i in range(1, 8)]
            + [(str(i), str(2 * i + 1)) for i in range(1, 8)]
        )
        tree = CassiopeiaTree(tree=tree)
        cm = pd.DataFrame.from_dict(
            {
                "8": [99, 99, 00, 00, 99, 99, 99],
                "9": [-1, -1, 11, -1, -1, -1, -1],
                "10": [99, -1, 22, -1, 99, 99, 99],
                "11": [-1, -1, 33, 99, -1, -1, 11],
                "12": [-1, 99, 44, -1, 99, 22, 99],
                "13": [-1, -1, -1, 99, -1, -1, -1],
                "14": [-1, -1, 55, 99, -1, -1, 11],
                "15": [-1, -1, -1, -1, -1, -1, 22],
            },
            orient="index",
        )
        tree.character_matrix = cm
        tree.set_character_states_at_leaves(cm)
        tree = conservative_maximum_parsimony(tree)

        def dfs(v):
            if v == 0:
                return [(0, tree.get_character_states("0"))] + dfs(1)
            elif 2 * v >= len(tree.nodes):
                return [(v, tree.get_character_states(str(v)))]
            else:
                return (
                    [(v, tree.get_character_states(str(v)))]
                    + dfs(2 * v)
                    + dfs(2 * v + 1)
                )

        cs = dfs(0)
        self.assertEqual(
            cs,
            [
                (0, [00, 00, 00, 00, 00, 00, 00]),
                (1, [-1, 99, 00, 00, 99, 00, 00]),
                (2, [99, 99, 00, 00, 99, 99, 00]),
                (4, [99, 99, 00, 00, 99, 99, -1]),
                (8, [99, 99, 00, 00, 99, 99, 99]),
                (9, [-1, -1, 11, -1, -1, -1, -1]),
                (5, [99, -1, 00, -1, 99, 99, 00]),
                (10, [99, -1, 22, -1, 99, 99, 99]),
                (11, [-1, -1, 33, 99, -1, -1, 11]),
                (3, [-1, 99, 00, 99, 99, -1, 00]),
                (6, [-1, 99, -1, 99, 99, -1, -1]),
                (12, [-1, 99, 44, -1, 99, 22, 99]),
                (13, [-1, -1, -1, 99, -1, -1, -1]),
                (7, [-1, -1, -1, 99, -1, -1, 00]),
                (14, [-1, -1, 55, 99, -1, -1, 11]),
                (15, [-1, -1, -1, -1, -1, -1, 22]),
            ],
        )

    def test_perfect_binary_tree(self):
        tree = nx.DiGraph()
        tree.add_nodes_from([str(i) for i in range(15)])
        tree.add_edges_from(
            [(str(i), str(2 * i + 1)) for i in range(7)]
            + [(str(i), str(2 * i + 2)) for i in range(7)]
        )
        tree = CassiopeiaTree(tree=tree)
        cm = pd.DataFrame.from_dict(
            {
                "7": [00, 99],
                "8": [00, -1],
                "9": [00, 99],
                "10": [00, -1],
                "11": [00, 99],
                "12": [00, -1],
                "13": [00, -1],
                "14": [00, -1],
            },
            orient="index",
        )
        tree.character_matrix = cm
        tree.set_character_states_at_leaves(cm)
        tree = conservative_maximum_parsimony(tree)

        def dfs(v):
            if 2 * v >= len(tree.nodes) - 1:
                return [(v, tree.get_character_states(str(v)))]
            else:
                return (
                    [(v, tree.get_character_states(str(v)))]
                    + dfs(2 * v + 1)
                    + dfs(2 * v + 2)
                )

        cs = dfs(0)
        self.assertEqual(
            cs,
            [
                (0, [00, 00]),
                (1, [00, 99]),
                (3, [00, 99]),
                (7, [00, 99]),
                (8, [00, -1]),
                (4, [00, 99]),
                (9, [00, 99]),
                (10, [00, -1]),
                (2, [00, -1]),
                (5, [00, -1]),
                (11, [00, 99]),
                (12, [00, -1]),
                (6, [00, -1]),
                (13, [00, -1]),
                (14, [00, -1]),
            ],
        )
