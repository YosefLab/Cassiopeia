"""
Tests the functionality of cassiopeia.simulator.ecDNABirthDeathSimulator.
"""

from tkinter.tix import Tree
import unittest

import networkx as nx
import numpy as np

from typing import List, Tuple, Generator
from queue import PriorityQueue


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

        tree.nodes[0]["ecdna_array"] = np.array([4, 5])

        # test new generation of child
        new_ecdna_array = sim.get_ecdna_array(0, tree)
        expected_array = [3, 2]  # manually find out and set
        self.assertTrue(np.all(expected_array == new_ecdna_array))

        # test generation when child already exists
        tree.add_edge(0, 1)
        tree.nodes[1]["ecdna_array"] = np.array([5, 7])
        new_ecdna_array = sim.get_ecdna_array(0, tree)
        expected_array = [3, 3]  # calculated manually as [4*2-5, 5*2 - 7]
        self.assertTrue(np.all(expected_array == new_ecdna_array))

    def test_sample_lineage_event(self):
        """Checks that sample_lineage_event behaves as expected under edge cases"""
        np.random.seed(41)
        sim = ecDNABirthDeathSimulator(
            birth_waiting_distribution=lambda _: 1,
            initial_birth_scale=1,
            num_extant=16,
            experiment_time=5,
            fitness_array = np.array([[[0,0.1], [0.1,0.2]],[[0.1,0.5],[0.2,0.6]]]),
        )

        def node_name_generator() -> Generator[str, None, None]:
            """Generates unique node names for the tree."""
            i = 0
            while True:
                yield str(i)
                i += 1

        # first, we'll trigger the time=0 condition in sample_lineage_event.
        names = node_name_generator()

        tree = nx.DiGraph()
        root = next(names)
        tree.add_node(root)
        tree.nodes[root]["birth_scale"] = 1
        tree.nodes[root]["time"] = 0
        tree.nodes[root]["ecdna_array"] = np.array([3, 2, 5])
        current_lineages = PriorityQueue()
        observed_nodes = []
        starting_lineage = {
            "id": root,
            "birth_scale": tree.nodes[root]["birth_scale"],
            "total_time": tree.nodes[root]["time"],
            "active": True,
        }

        sim.sample_lineage_event(
            starting_lineage, current_lineages, tree, names, observed_nodes
        )
        self.assertTrue(current_lineages.qsize() == 1)

        _, _, new_lineage = current_lineages.get()
        self.assertTrue(
            np.all(
                tree.nodes[root]["ecdna_array"]
                == tree.nodes[new_lineage["id"]]["ecdna_array"]
            )
        )
        self.assertTrue(new_lineage["active"])
        self.assertTrue(
            new_lineage["total_time"] > 0
        )  # does this really need to be true? yes

        # now, let's do one normal division
        sim.sample_lineage_event(
            new_lineage, current_lineages, tree, names, observed_nodes
        )
        # checked: the first ecdna_array is [2,0,6], so the other one should be [4,4,4]
        sim.sample_lineage_event(
            new_lineage, current_lineages, tree, names, observed_nodes
        )

        _, _, new_lineage = current_lineages.get()
        total_time_d1 = new_lineage["total_time"]

        _, _, new_lineage = current_lineages.get()
        total_time_d2 = new_lineage["total_time"]

        # This need not be true in general, it's just b/c our birth_waiting_distribution is constant.
        self.assertTrue(total_time_d1 == total_time_d2)

        expected_array = [4, 4, 4]
        self.assertTrue(
            np.all(tree.nodes[new_lineage["id"]]["ecdna_array"] == expected_array)
        )

        # Now, let's edit new_lineage to ensure it triggers stopping condition, then pass it into sample_lineage_event
        new_lineage["total_time"] = 4.5
        sim.sample_lineage_event(
            new_lineage, current_lineages, tree, names, observed_nodes
        )

        _, _, new_lineage = current_lineages.get()
        self.assertTrue(observed_nodes[0] == new_lineage["id"])
        self.assertTrue(new_lineage["total_time"] == 5)
        self.assertTrue(not (new_lineage["active"]))


if __name__ == "__main__":
    unittest.main()
