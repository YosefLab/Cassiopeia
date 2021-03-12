import unittest

import networkx as nx
import numpy as np

from cassiopeia.simulator.BirthDeathFitnessSimulator import (
    BirthDeathFitnessSimulator,
)
from cassiopeia.simulator.TreeSimulator import (
    TreeSimulatorError,
)

import cassiopeia.data.utilities as utilities


def get_leaves(tree):
    return [n for n in tree.nodes if tree.out_degree(n) == 0]


def test_tree(tree: nx.DiGraph):
    """A helper function for testing simulated trees.

    Outputs the (independently calculated) total lived time for each extant
    lineage, the number of extant lineages, and whether the tree has the
    expected node degrees (to ensure unifurcation collapsing was done
    correctly).
    """
    tree = tree.copy()
    tree.nodes["0"]["total_time"] = 0
    for i in nx.edge_dfs(tree):
        tree.nodes[i[1]]["total_time"] = (
            tree.nodes[i[0]]["total_time"] + tree.edges[i]["length"]
        )

    leaves = get_leaves(tree)
    times = []
    for i in leaves:
        times.append(tree.nodes[i]["total_time"])

    out_degrees = [tree.out_degree(n) for n in tree.nodes]
    out_degrees.pop(0)

    correct_degrees = all(x == 2 or x == 0 for x in out_degrees)

    return times, len(leaves), correct_degrees


class BirthDeathSimulatorTest(unittest.TestCase):
    def test_bad_waiting_distributions(self):
        """Ensures errors when invalid waiting distributions are given."""
        with self.assertRaises(TreeSimulatorError):
            bd_sim = BirthDeathFitnessSimulator(
                lambda _: -1, 1, experiment_time=1
            )
            tree = bd_sim.simulate_tree()

        with self.assertRaises(TreeSimulatorError):
            bd_sim = BirthDeathFitnessSimulator(lambda _: 0, 1, num_extant=4)
            tree = bd_sim.simulate_tree()

        with self.assertRaises(TreeSimulatorError):
            bd_sim = BirthDeathFitnessSimulator(
                lambda _: 1, 1, lambda: -1, num_extant=1
            )
            tree = bd_sim.simulate_tree()

        with self.assertRaises(TreeSimulatorError):
            bd_sim = BirthDeathFitnessSimulator(
                lambda _: 1, 1, lambda: 0, experiment_time=1
            )
            tree = bd_sim.simulate_tree()

    def test_bad_stopping_conditions(self):
        """Ensures errors when an invalid stopping conditions are given."""
        with self.assertRaises(TreeSimulatorError):
            bd_sim = BirthDeathFitnessSimulator(lambda _: 1, 1, lambda: 2)

        with self.assertRaises(TreeSimulatorError):
            bd_sim = BirthDeathFitnessSimulator(
                lambda _: 1, 1, lambda: 2, num_extant=4, experiment_time=4
            )

        with self.assertRaises(TreeSimulatorError):
            bd_sim = BirthDeathFitnessSimulator(
                lambda _: 1, 1, lambda: 2, num_extant=-1
            )

        with self.assertRaises(TreeSimulatorError):
            bd_sim = BirthDeathFitnessSimulator(
                lambda _: 1, 1, lambda: 2, num_extant=0
            )

        with self.assertRaises(TreeSimulatorError):
            bd_sim = BirthDeathFitnessSimulator(
                lambda _: 1, 1, lambda: 2, experiment_time=-1
            )

        with self.assertRaises(TreeSimulatorError):
            bd_sim = BirthDeathFitnessSimulator(
                lambda _: 1, 1, lambda: 2, experiment_time=0
            )

    def test_dead_at_start(self):
        """Ensures errors in base case where all lineages die on first event."""
        with self.assertRaises(TreeSimulatorError):
            bd_sim = BirthDeathFitnessSimulator(
                lambda _: 2, 1, lambda: 1, num_extant=4
            )
            tree = bd_sim.simulate_tree()

        with self.assertRaises(TreeSimulatorError):
            bd_sim = BirthDeathFitnessSimulator(
                lambda _: 2, 1, lambda: 1, experiment_time=4
            )
            tree = bd_sim.simulate_tree()

    def test_dead_before_end(self):
        """Ensures errors when all lineages die before stopping condition."""
        birth_wd = lambda scale: np.random.exponential(scale)
        death_wd = lambda: np.random.exponential(0.6)

        with self.assertRaises(TreeSimulatorError):
            np.random.seed(5)
            bd_sim = BirthDeathFitnessSimulator(
                birth_wd, 0.5, death_wd, num_extant=8
            )
            tree = bd_sim.simulate_tree()

        with self.assertRaises(TreeSimulatorError):
            np.random.seed(5)
            bd_sim = BirthDeathFitnessSimulator(
                birth_wd, 0.5, death_wd, experiment_time=2
            )
            tree = bd_sim.simulate_tree()

    def test_single_lineage(self):
        """Tests base case that stopping conditions work before divisions."""
        bd_sim = BirthDeathFitnessSimulator(lambda _: 1, 1, num_extant=1)
        tree = bd_sim.simulate_tree()
        results = test_tree(tree.get_tree_topology())
        self.assertEqual(results[1], 1)
        self.assertEqual(
            tree.get_tree_topology().get_edge_data("0", "1")["length"], 1.0
        )
        self.assertEqual(results[0], [1])

        bd_sim = BirthDeathFitnessSimulator(lambda _: 1, 1, experiment_time=1)
        tree = bd_sim.simulate_tree()
        self.assertEqual(results[1], 1)
        self.assertEqual(
            tree.get_tree_topology().get_edge_data("0", "1")["length"], 1.0
        )
        self.assertEqual(results[0], [1])

    def test_constant_yule(self):
        """Tests small case without death with constant waiting times."""
        bd_sim = BirthDeathFitnessSimulator(lambda _: 1, 1, num_extant=32)
        tree = bd_sim.simulate_tree()
        results = test_tree(tree.get_tree_topology())
        for i in results[0]:
            self.assertEqual(i, 6)
        self.assertEqual(results[1], 32)
        self.assertTrue(results[2])

        bd_sim = BirthDeathFitnessSimulator(lambda _: 1, 1, experiment_time=6)
        tree = bd_sim.simulate_tree()
        results = test_tree(tree.get_tree_topology())
        for i in results[0]:
            self.assertEqual(i, 6)
        self.assertEqual(results[1], 32)
        self.assertTrue(results[2])

    def test_nonconstant_yule(self):
        """Tests case without death with variable waiting times."""
        birth_wd = lambda scale: np.random.exponential(scale)
        bd_sim = BirthDeathFitnessSimulator(birth_wd, 1, num_extant=16)
        tree = bd_sim.simulate_tree()

        tree_top = tree.get_tree_topology()
        results = test_tree(tree_top)
        self.assertTrue(all(np.isclose(x, results[0][0]) for x in results[0]))
        self.assertEqual(results[1], 16)
        self.assertTrue(results[2])
        self.assertEqual(max([int(i) for i in tree.nodes]), 31)

        bd_sim = BirthDeathFitnessSimulator(birth_wd, 1, experiment_time=2)
        tree = bd_sim.simulate_tree()

        tree_top = tree.get_tree_topology()
        results = test_tree(tree_top)
        for i in results[0]:
            self.assertEqual(i, 2)
        self.assertTrue(results[2])

    def test_nonconstant_birth_death(self):
        """Tests case with with variable birth and death waiting times.
        Also, tests pruning dead lineages and unifurcation collapsing."""
        birth_wd = lambda scale: np.random.exponential(scale)
        death_wd = lambda: np.random.exponential(1.5)

        np.random.seed(1234)
        bd_sim = BirthDeathFitnessSimulator(
            birth_wd, 0.5, death_wd, num_extant=8
        )
        tree = bd_sim.simulate_tree()

        tree_top = tree.get_tree_topology()
        results = test_tree(tree_top)
        self.assertTrue(all(np.isclose(x, results[0][0]) for x in results[0]))
        self.assertEqual(results[1], 8)
        self.assertTrue(results[2])
        self.assertNotIn(9, tree_top.nodes)
        self.assertNotIn(2, tree_top.nodes)

        np.random.seed(1234)
        bd_sim = BirthDeathFitnessSimulator(
            birth_wd, 0.5, death_wd, experiment_time=2
        )
        tree = bd_sim.simulate_tree()

        tree_top = tree.get_tree_topology()
        results = test_tree(tree_top)
        for i in results[0]:
            self.assertTrue(np.isclose(i, 2))
        self.assertTrue(results[2])
        self.assertNotIn(9, tree_top.nodes)
        self.assertNotIn(2, tree_top.nodes)

    def test_nonconstant_yule_with_predictable_fitness(self):
        """Tests case with birth and death with constant fitness."""

        def check_fitness_values_as_expected(tree):
            """Checks if the fitness value stored at each node is what we
            expect given deterministic fitness evolution"""
            for u, v in tree.edges:
                tree[u][v]["val"] = 1
            tree.nodes["0"]["depth"] = 0
            for u, v in nx.dfs_edges(tree, source="0"):
                tree.nodes[v]["depth"] = (
                    tree.nodes[u]["depth"] + tree[u][v]["val"]
                )
            leaves = get_leaves(tree)
            for i in tree.nodes:
                if i in leaves:
                    self.assertTrue(
                        np.isclose(
                            tree.nodes[i]["birth_scale"],
                            0.5 * 0.98 ** (2 * (tree.nodes[i]["depth"] - 1)),
                        )
                    )
                else:
                    self.assertTrue(
                        np.isclose(
                            tree.nodes[i]["birth_scale"],
                            0.5 * 0.98 ** (2 * tree.nodes[i]["depth"]),
                        )
                    )

        birth_wd = lambda scale: np.random.exponential(scale)

        np.random.seed(1234)
        bd_sim = BirthDeathFitnessSimulator(
            birth_wd,
            0.5,
            mutation_distribution=lambda: 2,
            fitness_distribution=lambda: 1,
            fitness_base=0.98,
            num_extant=8,
        )
        tree = bd_sim.simulate_tree()

        tree_top = tree.get_tree_topology()
        results = test_tree(tree_top)
        self.assertTrue(all(np.isclose(x, results[0][0]) for x in results[0]))
        self.assertEqual(results[1], 8)
        self.assertTrue(results[2])
        check_fitness_values_as_expected(tree_top)

        np.random.seed(1234)
        bd_sim = BirthDeathFitnessSimulator(
            birth_wd,
            0.5,
            mutation_distribution=lambda: 2,
            fitness_distribution=lambda: 1,
            fitness_base=0.98,
            experiment_time=0.6,
        )
        tree = bd_sim.simulate_tree()

        tree_top = tree.get_tree_topology()
        results = test_tree(tree_top)
        for i in results[0]:
            self.assertTrue(np.isclose(i, 0.6))
        self.assertTrue(results[2])
        check_fitness_values_as_expected(tree_top)

    def test_nonconstant_birth_death_with_variable_fitness(self):
        """Tests a case with variable birth and death waiting times, as well
        as variable fitness evolution. Also tests pruning and collapsing."""

        birth_wd = lambda scale: np.random.exponential(scale)
        death_wd = lambda: np.random.exponential(0.6)
        mut_dist = lambda: 1 if np.random.uniform() < 0.2 else 0
        fit_dist = lambda: np.random.uniform(-1, 1)

        np.random.seed(12364)
        bd_sim = BirthDeathFitnessSimulator(
            birth_wd, 0.5, death_wd, mut_dist, fit_dist, 1.5, num_extant=8
        )
        tree = bd_sim.simulate_tree()

        tree_top = tree.get_tree_topology()
        results = test_tree(tree_top)
        self.assertTrue(all(np.isclose(x, results[0][0]) for x in results[0]))
        self.assertEqual(results[1], 8)
        self.assertTrue(results[2])
        self.assertNotIn(2, tree_top.nodes)
        self.assertNotIn(3, tree_top.nodes)

        np.random.seed(12364)
        bd_sim = BirthDeathFitnessSimulator(
            birth_wd, 0.5, death_wd, mut_dist, fit_dist, 1.5, experiment_time=3
        )
        tree = bd_sim.simulate_tree()

        tree_top = tree.get_tree_topology()
        results = test_tree(tree_top)
        for i in results[0]:
            self.assertTrue(np.isclose(i, 3))
        self.assertTrue(results[2])
        self.assertNotIn(2, tree_top.nodes)
        self.assertNotIn(3, tree_top.nodes)


if __name__ == "__main__":
    unittest.main()
