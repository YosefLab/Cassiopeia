import unittest

import networkx as nx
import numpy as np

from cassiopeia.data import CassiopeiaTree
from cassiopeia.simulator import IIDExponentialLineageTracer


class Test(unittest.TestCase):
    def test_smoke(self):
        r"""
        Just tests that lineage_tracing_simulator runs
        """
        np.random.seed(1)
        tree = nx.DiGraph()
        tree.add_nodes_from(["0", "1", "2", "3", "4", "5", "6"]),
        tree.add_edges_from(
            [
                ("0", "1"),
                ("0", "2"),
                ("1", "3"),
                ("1", "4"),
                ("2", "5"),
                ("2", "6"),
            ]
        )
        tree = CassiopeiaTree(tree=tree)
        tree.set_times(
            {"0": 0, "1": 0.1, "2": 0.9, "3": 1.0, "4": 1.0, "5": 1.0, "6": 1.0}
        )
        np.random.seed(1)
        IIDExponentialLineageTracer(
            mutation_rate=1.0, num_characters=10
        ).overlay_data(tree)
