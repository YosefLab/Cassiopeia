import unittest

import networkx as nx
import numpy as np
import pandas as pd

import cassiopeia as cas
from cassiopeia.simulator import BrownianSpatialDataSimulator, DataSimulatorError


class TestBrownianSpatialDataSimulator(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        topology = nx.DiGraph()
        topology.add_edges_from(
            [
                ("0", "1"),
                ("0", "2"),
                ("1", "3"),
                ("1", "4"),
                ("2", "5"),
                ("2", "6"),
                ("3", "7"),
                ("3", "8"),
                ("4", "9"),
                ("4", "10"),
                ("5", "11"),
                ("5", "12"),
                ("6", "13"),
                ("6", "14"),
            ]
        )

        tree = cas.data.CassiopeiaTree(tree=topology)
        tree.set_times(
            {
                "0": 0,
                "1": 1,
                "2": 1,
                "3": 2,
                "4": 2,
                "5": 2,
                "6": 2,
                "7": 3,
                "8": 3,
                "9": 3,
                "10": 3,
                "11": 3,
                "12": 3,
                "13": 3,
                "14": 3,
            }
        )

        self.basic_tree = tree

    def test_init(self):
        with self.assertRaises(DataSimulatorError):
            BrownianSpatialDataSimulator(dim=0, diffusion_coefficient=1)

        with self.assertRaises(DataSimulatorError):
            BrownianSpatialDataSimulator(dim=2, diffusion_coefficient=-1)

        simulator = BrownianSpatialDataSimulator(dim=2, diffusion_coefficient=1)
        self.assertEqual(simulator.dim, 2)
        self.assertEqual(simulator.diffusion_coefficient, 1)

    def test_overlay_data(self):
        simulator = BrownianSpatialDataSimulator(dim=2, diffusion_coefficient=1)
        simulator.overlay_data(self.basic_tree)

        expected_coordinates = {
            "0": (0.11770352522994826, 0.35650052707603014),
            "1": (0.4552956241457416, 0.4330798373089543),
            "2": (0.2026469911744811, 0.42035680480502396),
            "3": (0.6425997090262963, 0.8619264303784856),
            "4": (0.4355423164254138, 0.5116573298634731),
            "5": (0.4885734782236841, 0.3810950396805744),
            "6": (0.36807679342430616, 0.2783264123659291),
            "7": (1.0, 0.6749017701150868),
            "8": (0.8244210386264144, 0.8329607379759592),
            "9": (0.46310837569496033, 0.7899661038267269),
            "10": (0.5811844361582845, 0.5349426503344454),
            "11": (0.5484862028912141, 0.2176441228384555),
            "12": (0.0, 0.5061800316477849),
            "13": (0.8024466857278588, 0.0),
            "14": (0.3768337409912992, 0.24250446583593732),
        }
        for node in self.basic_tree.nodes:
            np.testing.assert_almost_equal(
                self.basic_tree.get_attribute(node, 'spatial'),
                expected_coordinates[node]
            )
        expected_cell_meta = pd.DataFrame(
            data=[expected_coordinates[leaf] for leaf in self.basic_tree.leaves],
            index=self.basic_tree.leaves,
            columns=['spatial_0', 'spatial_1']
        )
        pd.testing.assert_frame_equal(self.basic_tree.cell_meta, expected_cell_meta)

    def test_overlay_data_without_scale(self):
        simulator = BrownianSpatialDataSimulator(dim=2, diffusion_coefficient=1, scale_unit_area=False)
        simulator.overlay_data(self.basic_tree)

        expected_coordinates = {
            "0": (0.0, 0.0),
            "1": (2.494746752403546, 0.5659077511542838),
            "2": (0.6277174035873467, 0.4718867591884083),
            "3": (3.878891283535585, 3.735009305294619),
            "4": (2.3487732523045177, 1.1465817212856055),
            "5": (2.740664292104657, 0.18174916013789594),
            "6": (1.8502147998207057, -0.577693078189287),
            "7": (6.5200171217239085, 2.352929673366174),
            "8": (5.222519209053866, 3.5209578885570854),
            "9": (2.5524816242128856, 3.2032350382325285),
            "10": (3.4250431246849637, 1.3186561798117336),
            "11": (3.183408881720336, -1.0261246179476786),
            "12": (-0.8698085300482874, 1.1061054424289072),
            "13": (5.060132572323026, -2.6344767398567557),
            "14": (1.9149271155824947, -0.8424110175530323),
        }
        for node in self.basic_tree.nodes:
            np.testing.assert_almost_equal(
                self.basic_tree.get_attribute(node, 'spatial'),
                expected_coordinates[node]
            )
        expected_cell_meta = pd.DataFrame(
            data=[expected_coordinates[leaf] for leaf in self.basic_tree.leaves],
            index=self.basic_tree.leaves,
            columns=['spatial_0', 'spatial_1']
        )
        pd.testing.assert_frame_equal(self.basic_tree.cell_meta, expected_cell_meta)
