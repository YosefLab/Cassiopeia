import unittest

from cassiopeia.mixins import TreeSimulatorError
from cassiopeia.simulator import CompleteBinarySimulator


class TestCompleteBinarySimulator(unittest.TestCase):
    def test_init(self):
        with self.assertRaises(TreeSimulatorError):
            CompleteBinarySimulator()

        with self.assertRaises(TreeSimulatorError):
            CompleteBinarySimulator(num_cells=3)

        with self.assertRaises(TreeSimulatorError):
            CompleteBinarySimulator(depth=0)

        simulator = CompleteBinarySimulator(num_cells=4)
        self.assertEqual(simulator.depth, 2)

    def test_simulate_tree(self):
        tree = CompleteBinarySimulator(depth=2).simulate_tree()

        self.assertEqual(set(tree.nodes), {"0", "1", "2", "3", "4", "5", "6", "7"})
        self.assertEqual(set(tree.leaves), {"4", "5", "6", "7"})
        self.assertEqual(
            set(tree.edges),
            {
                ("0", "1"),
                ("1", "2"),
                ("1", "3"),
                ("2", "4"),
                ("2", "5"),
                ("3", "6"),
                ("3", "7"),
            },
        )

        # Test branch lengths
        self.assertEqual(
            tree.get_times(),
            {
                "0": 0.0,
                "1": 1 / 3,
                "2": 2 / 3,
                "3": 2 / 3,
                "4": 1.0,
                "5": 1.0,
                "6": 1.0,
                "7": 1.0,
            },
        )
