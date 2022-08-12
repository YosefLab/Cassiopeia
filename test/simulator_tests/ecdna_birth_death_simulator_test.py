"""
Tests the functionality of cassiopeia.simulator.ecDNABirthDeathSimulator.
"""

from tkinter.tix import Tree
import unittest

import networkx as nx
import numpy as np

from typing import List, Tuple

from cassiopeia.data.CassiopeiaTree import CassiopeiaTree
from cassiopeia.mixins import ecDNABirthDeathSimulatorError
from cassiopeia.simulator.ecDNABirthDeathSimulator import (
    ecDNABirthDeathSimulator,
)

import cassiopeia.data.utilities as utilities


def extract_tree_statistics(
    tree: CassiopeiaTree,
) -> Tuple[List[float], int, bool]:
    """A helper function for testing simulated trees.

    Outputs the total lived time for each extant lineage, the number of extant
    lineages, and whether the tree has the expected node degrees (to ensure
    unifurcations were collapsed).

    Args:
        tree: The tree to test

    Returns:
        The total time lived for each leaf, the number of leaves, and if the
        degrees only have degree 0 or 2
    """

    times = []
    out_degrees = []
    for i in tree.nodes:
        if tree.is_leaf(i):
            times.append(tree.get_time(i))
        out_degrees.append(len(tree.children(i)))
    out_degrees.pop(0)

    correct_degrees = all(x == 2 or x == 0 for x in out_degrees)

    return times, len(times), correct_degrees


class ecDNABirthDeathSimulatorTest(unittest.TestCase):

    def test_ecdna_splitting(self):
        
        np.random.seed(41)
        sim = ecDNABirthDeathSimulator(lambda _: 1, 1, num_extant=16)

        tree = nx.DiGraph()
        tree.add_node(0)

        tree.nodes[0]['ecdna_array'] = np.array([4, 5])

        # test new generation of child 
        new_ecdna_array = sim.get_ecdna_array(0, tree)
        expected_array = [0, 0] # manually find out and set
        self.assertTrue(np.all(expected_array == new_ecdna_array))

        # test generation when child already exists
        tree.add_edge(0, 1)
        tree.nodes[1]['ecdna_array'] = np.array([5, 7])
        new_ecdna_array = sim.get_ecdna_array(0, tree)
        expected_array = [3, 3] # manually find out and set
        self.assertTrue(np.all(expected_array == new_ecdna_array))






if __name__ == "__main__":
    unittest.main()
