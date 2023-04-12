"""
Tests the functionality of cassiopeia.simulator.ecDNABirthDeathSimulator.
"""
import heapq
from typing import List, Tuple, Generator
# from queue import PriorityQueue
import unittest

import networkx as nx
import numpy as np
import pandas as pd

from cassiopeia.data.CassiopeiaTree import CassiopeiaTree
from cassiopeia.simulator.ecDNABirthDeathSimulator import (
    ecDNABirthDeathSimulator,
)
import cassiopeia.data.utilities as utilities


def node_name_generator() -> Generator[str, None, None]:
    """Generates unique node names for the tree."""
    i = 0
    while True:
        yield str(i)
        i += 1

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
    
    def test_initial_sample_event(self):

        np.random.seed(41)
        sim = ecDNABirthDeathSimulator(
            birth_waiting_distribution=lambda _: 1,
            initial_birth_scale=1,
            num_extant=16,
            experiment_time=5,
            fitness_array = np.array([[[0,0.1], [0.1,0.2]],[[0.1,0.5],[0.2,0.6]]]),
        )

        # first, we'll trigger the time=0 condition in sample_lineage_event.
        names = node_name_generator()

        tree = nx.DiGraph()
        root = next(names)
        tree.add_node(root)
        tree.nodes[root]["birth_scale"] = 1
        tree.nodes[root]["time"] = 0
        tree.nodes[root]["ecdna_array"] = np.array([3, 2, 5])
        # current_lineages = PriorityQueue()
        current_lineages = []
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
        self.assertTrue(len(current_lineages) == 1)

        _, _, new_lineage = heapq.heappop(current_lineages)
        self.assertTrue(
            np.all(
                tree.nodes[root]["ecdna_array"]
                == tree.nodes[new_lineage["id"]]["ecdna_array"]
            )
        )

    def test_basic_sample_lineage_events(self):
        """Checks that sample_lineage_event behaves as expected under edge cases."""

        np.random.seed(41)
        sim = ecDNABirthDeathSimulator(
            birth_waiting_distribution=lambda _: 1,
            initial_birth_scale=1,
            num_extant=16,
            experiment_time=5,
            fitness_array = np.array([[[0,0.1], [0.1,0.2]],[[0.1,0.5],[0.2,0.6]]]),
        )

        names = node_name_generator()

        tree = nx.DiGraph()
        root = next(names)
        tree.add_node(root)
        tree.nodes[root]["birth_scale"] = 1
        tree.nodes[root]["time"] = 0
        tree.nodes[root]["ecdna_array"] = np.array([3, 2, 5])
        current_lineages = []
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

        _, _, new_lineage = heapq.heappop(current_lineages)
        # now, let's do one normal division
        sim.sample_lineage_event(
            new_lineage, current_lineages, tree, names, observed_nodes
        )

        # checked: the first ecdna_array is [2,0,6], so the other one should be [4,4,4]
        sim.sample_lineage_event(
            new_lineage, current_lineages, tree, names, observed_nodes
        )

        _, _, new_lineage = heapq.heappop(current_lineages)
        total_time_d1 = new_lineage["total_time"]

        _, _, new_lineage = heapq.heappop(current_lineages)
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

        _, _, new_lineage = heapq.heappop(current_lineages)
        self.assertTrue(observed_nodes[0] == new_lineage["id"])
        self.assertTrue(new_lineage["total_time"] == 5)
        self.assertTrue(not (new_lineage["active"]))

    def test_populate_tree_from_simulation(self):

        np.random.seed(41)
        sim = ecDNABirthDeathSimulator(
            birth_waiting_distribution=lambda _: 1,
            initial_birth_scale=1,
            num_extant=16,
            experiment_time=5,
            initial_copy_number=[3,2,5]
        )
        
        names = node_name_generator()

        tree = nx.DiGraph()
        root = next(names)
        tree.add_node(root)

        child_1, child_2 = next(names), next(names)
        tree.add_edges_from([(root, child_1), (root, child_2)])

        tree.nodes[root]["birth_scale"] = 1
        tree.nodes[root]["time"] = 0
        tree.nodes[root]["ecdna_array"] = np.array([3, 2, 5])

        tree.nodes[child_1]['birth_scale'] = 1
        tree.nodes[child_1]["time"] = 2
        tree.nodes[child_1]["ecdna_array"] = np.array([2,0,6])
        
        tree.nodes[child_2]['birth_scale'] = 1
        tree.nodes[child_2]["time"] = 2
        tree.nodes[child_2]["ecdna_array"] = np.array([4, 4, 4])

        cassiopeia_tree = sim.populate_tree_from_simulation(tree, [child_1, child_2])

        expected_meta_data = pd.DataFrame.from_dict({
            child_1: [2, 0, 6, 2, 0, 6],
            child_2: [4, 4, 4, 4, 4, 4]
        }, orient='index', columns = ['ecDNA_0', 'ecDNA_1', 'ecDNA_2', 'Observed_ecDNA_0', 'Observed_ecDNA_1', 'Observed_ecDNA_2'])
        
        pd.testing.assert_frame_equal(expected_meta_data, cassiopeia_tree.cell_meta.astype(int))

    def test_basic_cosegregation(self):

        np.random.seed(41)

        # test basic cosegregation
        sim = ecDNABirthDeathSimulator(
            birth_waiting_distribution=lambda _: 1,
            initial_birth_scale=1,
            num_extant=16,
            experiment_time=5,
            cosegregation_coefficient=0.5,
            fitness_array = np.array([[0, 0.5], [0.5, 0]]),
        )

        names = node_name_generator()

        tree = nx.DiGraph()
        root = next(names)
        tree.add_node(root)
        tree.nodes[root]["birth_scale"] = 1
        tree.nodes[root]["time"] = 0
        tree.nodes[root]["ecdna_array"] = np.array([5, 5])
        current_lineages = []
        observed_nodes = []
        starting_lineage = {
            "id": root,
            "birth_scale": tree.nodes[root]["birth_scale"],
            "total_time": tree.nodes[root]["time"],
            "active": True,
        }

        # simulate time till first cell division 
        sim.sample_lineage_event(
            starting_lineage, current_lineages, tree, names, observed_nodes
        )

        _, _, new_lineage = heapq.heappop(current_lineages)

        # now, let's do one normal division
        sim.sample_lineage_event(
            new_lineage, current_lineages, tree, names, observed_nodes
        )
        sim.sample_lineage_event(
            new_lineage, current_lineages, tree, names, observed_nodes
        )

        # get out the children
        _, _, child_lineage_1 = heapq.heappop(current_lineages)
        _, _, child_lineage_2 = heapq.heappop(current_lineages)
        
        # expected arrays derivation (with random seed 41):
        # ecdna1 segregates as (4, 6):
        # ecdna2 co-segregation compartment - (2, 3)
        # remainder binomial: np.random.binomial(5, p=0.5) = 1 [with random seed 41, second draw]

        expected_child_1_array = [4, 3]
        expected_child_2_array = [6, 7]

        print(tree.nodes[child_lineage_1["id"]]["ecdna_array"])

        self.assertTrue(
            np.all(tree.nodes[child_lineage_1["id"]]["ecdna_array"] == expected_child_1_array)
        )
        self.assertTrue(
            np.all(tree.nodes[child_lineage_2["id"]]["ecdna_array"] == expected_child_2_array)
        )    

    def test_perfect_cosegregation(self):

        np.random.seed(41)

        # test perfect cosegregation
        sim = ecDNABirthDeathSimulator(
            birth_waiting_distribution=lambda _: 1,
            initial_birth_scale=1,
            num_extant=16,
            experiment_time=5,
            cosegregation_coefficient=1.0,
            fitness_array = np.array([[0, 0.5], [0.5, 0]]),
        )

        names = node_name_generator()

        tree = nx.DiGraph()
        root = next(names)
        tree.add_node(root)
        tree.nodes[root]["birth_scale"] = 1
        tree.nodes[root]["time"] = 0
        tree.nodes[root]["ecdna_array"] = np.array([5, 5])
        current_lineages = []
        observed_nodes = []
        starting_lineage = {
            "id": root,
            "birth_scale": tree.nodes[root]["birth_scale"],
            "total_time": tree.nodes[root]["time"],
            "active": True,
        }

        # simulate time till first cell division 
        sim.sample_lineage_event(
            starting_lineage, current_lineages, tree, names, observed_nodes
        )

        _, _, new_lineage = heapq.heappop(current_lineages)

        # now, let's do one normal division
        sim.sample_lineage_event(
            new_lineage, current_lineages, tree, names, observed_nodes
        )

        sim.sample_lineage_event(
            new_lineage, current_lineages, tree, names, observed_nodes
        )

        _, _, child_lineage_1 = heapq.heappop(current_lineages)
        _, _, child_lineage_2 = heapq.heappop(current_lineages)

        self.assertEqual(tree.nodes[child_lineage_1["id"]]["ecdna_array"][0], tree.nodes[child_lineage_1["id"]]["ecdna_array"][1])
        self.assertEqual(tree.nodes[child_lineage_2["id"]]["ecdna_array"][0], tree.nodes[child_lineage_2["id"]]["ecdna_array"][1])

    def test_low_capture_efficiency(self):

        np.random.seed(41)
        sim = ecDNABirthDeathSimulator(
            birth_waiting_distribution=lambda _: 1,
            initial_birth_scale=1,
            num_extant=16,
            experiment_time=5,
            initial_copy_number=[3,2,5],
            capture_efficiency=0.5,
        )
        
        names = node_name_generator()

        tree = nx.DiGraph()
        root = next(names)
        tree.add_node(root)

        child_1, child_2 = next(names), next(names)
        tree.add_edges_from([(root, child_1), (root, child_2)])

        tree.nodes[root]["birth_scale"] = 1
        tree.nodes[root]["time"] = 0
        tree.nodes[root]["ecdna_array"] = np.array([3, 2, 5])

        tree.nodes[child_1]['birth_scale'] = 1
        tree.nodes[child_1]["time"] = 2
        tree.nodes[child_1]["ecdna_array"] = np.array([2,0,6])
        
        tree.nodes[child_2]['birth_scale'] = 1
        tree.nodes[child_2]["time"] = 2
        tree.nodes[child_2]["ecdna_array"] = np.array([4, 4, 4])

        cassiopeia_tree = sim.populate_tree_from_simulation(tree, [child_1, child_2])
        
        # reset seed to create expected observations
        np.random.seed(41)
        c1_0 = np.random.binomial(2, 0.5)
        c2_0 = np.random.binomial(4, 0.5)
        c1_1 = np.random.binomial(0, 0.5)
        c2_1 = np.random.binomial(4, 0.5)
        c1_2 = np.random.binomial(6, 0.5)
        c2_2 = np.random.binomial(4, 0.5)

        expected_meta_data = pd.DataFrame.from_dict({
            child_1: [2, 0, 6, c1_0, c1_1, c1_2],
            child_2: [4, 4, 4, c2_0, c2_1, c2_2]
        }, orient='index', columns = ['ecDNA_0', 'ecDNA_1', 'ecDNA_2', 'Observed_ecDNA_0', 'Observed_ecDNA_1', 'Observed_ecDNA_2'])
 
        pd.testing.assert_frame_equal(expected_meta_data, cassiopeia_tree.cell_meta.astype(int))

    
if __name__ == "__main__":
    unittest.main()
