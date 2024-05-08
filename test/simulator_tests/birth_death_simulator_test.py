"""
Tests the functionality of cassiopeia.simulator.BirthDeathFitnessSimulator.
"""

import unittest

import networkx as nx
import numpy as np

from typing import List, Tuple


from cassiopeia.data.CassiopeiaTree import CassiopeiaTree
from cassiopeia.mixins import TreeSimulatorError
from cassiopeia.simulator.BirthDeathFitnessSimulator import (
    BirthDeathFitnessSimulator,
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


class BirthDeathSimulatorTest(unittest.TestCase):
    def test_bad_waiting_distributions(self):
        """Ensures errors when invalid distributions are given."""
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

        with self.assertRaises(TreeSimulatorError):
            bd_sim = BirthDeathFitnessSimulator(
                lambda _: 1,
                1,
                lambda: 0,
                mutation_distribution=lambda: -1,
                fitness_distribution=lambda: 1,
                experiment_time=1,
            )
            tree = bd_sim.simulate_tree()

    def test_bad_stopping_conditions(self):
        """Ensures errors when an invalid stopping conditions are given."""
        with self.assertRaises(TreeSimulatorError):
            bd_sim = BirthDeathFitnessSimulator(lambda _: 1, 1, lambda: 2)

        with self.assertRaises(TreeSimulatorError):
            bd_sim = BirthDeathFitnessSimulator(
                lambda _: 1, 1, lambda: 2, num_extant=0.5
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
            bd_sim = BirthDeathFitnessSimulator(
                birth_wd, 0.5, death_wd, num_extant=8, random_seed=5
            )
            tree = bd_sim.simulate_tree()

        with self.assertRaises(TreeSimulatorError):
            bd_sim = BirthDeathFitnessSimulator(
                birth_wd, 0.5, death_wd, experiment_time=2, random_seed=5
            )
            tree = bd_sim.simulate_tree()

    def test_single_lineage(self):
        """Tests base case that stopping conditions work before divisions."""
        bd_sim = BirthDeathFitnessSimulator(lambda _: 1, 1, num_extant=1)
        tree = bd_sim.simulate_tree()
        results = extract_tree_statistics(tree)
        self.assertEqual(results[1], 1)
        self.assertEqual(tree.get_branch_length("0", "1"), 1.0)
        self.assertEqual(results[0], [1])

        bd_sim = BirthDeathFitnessSimulator(lambda _: 1, 1, experiment_time=1)
        tree = bd_sim.simulate_tree()
        results = extract_tree_statistics(tree)
        self.assertEqual(results[1], 1)
        self.assertEqual(tree.get_branch_length("0", "1"), 1.0)
        self.assertEqual(results[0], [1])

    def test_constant_yule(self):
        """Tests small case without death with constant waiting times."""
        bd_sim = BirthDeathFitnessSimulator(lambda _: 1, 1, num_extant=32)
        tree = bd_sim.simulate_tree()
        results = extract_tree_statistics(tree)
        for i in results[0]:
            self.assertEqual(i, 6)
        self.assertEqual(results[1], 32)
        self.assertTrue(results[2])

        bd_sim = BirthDeathFitnessSimulator(lambda _: 1, 1, experiment_time=6)
        tree = bd_sim.simulate_tree()
        results = extract_tree_statistics(tree)
        for i in results[0]:
            self.assertEqual(i, 6)
        self.assertEqual(results[1], 32)
        self.assertTrue(results[2])

    def test_nonconstant_yule(self):
        """Tests case without death with variable waiting times."""
        birth_wd = lambda scale: np.random.exponential(scale)

        bd_sim = BirthDeathFitnessSimulator(
            birth_wd, 1, num_extant=16, random_seed=54
        )
        tree = bd_sim.simulate_tree()
        results = extract_tree_statistics(tree)
        self.assertTrue(all(np.isclose(x, results[0][0]) for x in results[0]))
        self.assertEqual(results[1], 16)
        self.assertTrue(results[2])
        self.assertEqual(max([int(i) for i in tree.nodes]), 31)

        bd_sim = BirthDeathFitnessSimulator(
            birth_wd, 1, experiment_time=2, random_seed=54
        )
        tree = bd_sim.simulate_tree()
        results = extract_tree_statistics(tree)
        for i in results[0]:
            self.assertEqual(i, 2)
        self.assertTrue(results[2])

    def test_nonconstant_birth_death(self):
        """Tests case with with variable birth and death waiting times.
        Also, tests pruning dead lineages and unifurcation collapsing."""
        birth_wd = lambda scale: np.random.exponential(scale)
        death_wd = lambda: np.random.exponential(1.5)

        bd_sim = BirthDeathFitnessSimulator(
            birth_wd, 0.5, death_wd, num_extant=8, random_seed=1234
        )
        tree = bd_sim.simulate_tree()
        results = extract_tree_statistics(tree)
        self.assertTrue(all(np.isclose(x, results[0][0]) for x in results[0]))
        self.assertEqual(results[1], 8)
        self.assertTrue(results[2])
        self.assertNotIn("9", tree.nodes)
        self.assertNotIn("2", tree.nodes)

        bd_sim = BirthDeathFitnessSimulator(
            birth_wd, 0.5, death_wd, experiment_time=2, random_seed=1234
        )
        tree = bd_sim.simulate_tree()
        results = extract_tree_statistics(tree)
        for i in results[0]:
            self.assertTrue(np.isclose(i, 2))
        self.assertTrue(results[2])
        self.assertNotIn("9", tree.nodes)
        self.assertNotIn("2", tree.nodes)

    def test_nonconstant_birth_death_no_unifurcation_collapsing(self):
        """Tests case with with variable birth and death waiting times.
        Checks that unifurcations are not collapsed."""
        birth_wd = lambda scale: np.random.exponential(scale)
        death_wd = lambda: np.random.exponential(1.5)

        bd_sim = BirthDeathFitnessSimulator(
            birth_wd,
            0.5,
            death_wd,
            num_extant=8,
            collapse_unifurcations=False,
            random_seed=12,
        )
        tree = bd_sim.simulate_tree()
        results = extract_tree_statistics(tree)
        self.assertTrue(all(np.isclose(x, results[0][0]) for x in results[0]))
        self.assertEqual(results[1], 8)
        self.assertFalse(results[2])
        self.assertNotIn("3", tree.nodes)
        self.assertIn("2", tree.nodes)
        self.assertIn("6", tree.nodes)

        bd_sim = BirthDeathFitnessSimulator(
            birth_wd,
            0.5,
            death_wd,
            experiment_time=1.3,
            collapse_unifurcations=False,
            random_seed=12,
        )
        tree = bd_sim.simulate_tree()
        results = extract_tree_statistics(tree)
        for i in results[0]:
            self.assertTrue(np.isclose(i, 1.3))
        self.assertFalse(results[2])
        self.assertNotIn("3", tree.nodes)
        self.assertIn("2", tree.nodes)
        self.assertIn("6", tree.nodes)

    def test_nonconstant_birth_death_both_stopping_conditions(self):
        """Tests case with with variable birth and death waiting times.
        Checks that using both stopping conditions works fine."""
        birth_wd = lambda scale: np.random.exponential(scale)
        death_wd = lambda: np.random.exponential(1.5)

        bd_sim = BirthDeathFitnessSimulator(
            birth_wd,
            0.5,
            death_wd,
            num_extant=8,
            experiment_time=2,
            random_seed=17,
        )
        tree = bd_sim.simulate_tree()
        results = extract_tree_statistics(tree)
        self.assertTrue(all(np.isclose(x, results[0][0]) for x in results[0]))
        self.assertTrue(all(x > 1 for x in results[0]))
        self.assertEqual(results[1], 8)
        self.assertTrue(results[2])

        bd_sim = BirthDeathFitnessSimulator(
            birth_wd,
            0.5,
            death_wd,
            num_extant=8,
            experiment_time=1,
            random_seed=17,
        )
        tree = bd_sim.simulate_tree()
        results = extract_tree_statistics(tree)
        for i in results[0]:
            self.assertTrue(np.isclose(i, 1))
        self.assertEqual(results[1], 3)
        self.assertTrue(results[2])

    def test_nonconstant_yule_with_predictable_fitness(self):
        """Tests case with birth and death with constant fitness."""

        def check_fitness_values_as_expected(tree: nx.DiGraph):
            """Checks if the fitness value stored at each node is what we
            expect given deterministic fitness evolution"""
            tree = tree.copy()
            for u, v in tree.edges:
                tree[u][v]["val"] = 1
            tree.nodes["0"]["depth"] = 0
            for u, v in nx.dfs_edges(tree, source="0"):
                tree.nodes[v]["depth"] = (
                    tree.nodes[u]["depth"] + tree[u][v]["val"]
                )
            leaves = [n for n in tree if tree.out_degree(n) == 0]
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

        bd_sim = BirthDeathFitnessSimulator(
            birth_wd,
            0.5,
            mutation_distribution=lambda: 2,
            fitness_distribution=lambda: 1,
            fitness_base=0.98,
            num_extant=8,
            random_seed=1234,
        )
        tree = bd_sim.simulate_tree()
        results = extract_tree_statistics(tree)
        self.assertTrue(all(np.isclose(x, results[0][0]) for x in results[0]))
        self.assertEqual(results[1], 8)
        self.assertTrue(results[2])
        check_fitness_values_as_expected(tree.get_tree_topology())

        bd_sim = BirthDeathFitnessSimulator(
            birth_wd,
            0.5,
            mutation_distribution=lambda: 2,
            fitness_distribution=lambda: 1,
            fitness_base=0.98,
            experiment_time=0.6,
            random_seed=1234,
        )
        tree = bd_sim.simulate_tree()
        results = extract_tree_statistics(tree)
        for i in results[0]:
            self.assertTrue(np.isclose(i, 0.6))
        self.assertTrue(results[2])
        check_fitness_values_as_expected(tree.get_tree_topology())

    def test_nonconstant_birth_death_with_variable_fitness(self):
        """Tests a case with variable birth and death waiting times, as well
        as variable fitness evolution. Also tests pruning and collapsing."""

        birth_wd = lambda scale: np.random.exponential(scale)
        death_wd = lambda: np.random.exponential(0.6)
        mut_dist = lambda: 1 if np.random.uniform() < 0.2 else 0
        fit_dist = lambda: np.random.uniform(-1, 1)

        bd_sim = BirthDeathFitnessSimulator(
            birth_wd,
            0.5,
            death_wd,
            mut_dist,
            fit_dist,
            1.5,
            num_extant=8,
            random_seed=12364,
        )
        tree = bd_sim.simulate_tree()
        results = extract_tree_statistics(tree)
        self.assertTrue(all(np.isclose(x, results[0][0]) for x in results[0]))
        self.assertEqual(results[1], 8)
        self.assertTrue(results[2])
        self.assertNotIn(2, tree.nodes)
        self.assertNotIn(3, tree.nodes)

        bd_sim = BirthDeathFitnessSimulator(
            birth_wd,
            0.5,
            death_wd,
            mut_dist,
            fit_dist,
            1.5,
            experiment_time=3,
            random_seed=12364,
        )
        tree = bd_sim.simulate_tree()
        results = extract_tree_statistics(tree)
        for i in results[0]:
            self.assertTrue(np.isclose(i, 3))
        self.assertTrue(results[2])
        self.assertNotIn(2, tree.nodes)
        self.assertNotIn(3, tree.nodes)

    def test_no_initial_birth_scale(self):
        
        topology = nx.DiGraph()
        topology.add_edges_from(
            [
                ("0", "1"),
                ("0", "2"),
                ("1", "3"),
                ("1", "4"),
                ("2", "5"),
                ("2", "6"),
            ]
        )
        initial_tree = CassiopeiaTree(tree=topology)
        
        # initialize simulator with tree without default initial birth scales
        birth_wd = lambda scale: np.random.exponential(scale)

        bd_sim = BirthDeathFitnessSimulator(
            birth_wd, 1, num_extant=16, random_seed=54, initial_tree=initial_tree
        )
        final_tree = bd_sim.simulate_tree()

        self.assertEqual(16, len(final_tree.leaves))
        for l in initial_tree.leaves:
            birth_scale = final_tree.get_attribute(l, 'birth_scale')
            self.assertEquals(1, birth_scale)

    def test_birth_scale(self):

        birth_wd = lambda scale: np.random.exponential(scale)
        bd_sim_1 = BirthDeathFitnessSimulator(
            birth_wd, 1, num_extant=16, random_seed=54
        )

        initial_tree = bd_sim_1.simulate_tree()

        bd_sim_2 = BirthDeathFitnessSimulator(
            birth_wd, 1, num_extant=100, random_seed=54, initial_tree=initial_tree
        )
        final_tree = bd_sim_2.simulate_tree()

        self.assertEqual(100, len(final_tree.leaves))
        
        for l in initial_tree.leaves:
            birth_scale = initial_tree.get_attribute(l, 'birth_scale')
            final_birth_scale = final_tree.get_attribute(l, 'birth_scale')
            self.assertEquals(birth_scale, final_birth_scale)



if __name__ == "__main__":
    unittest.main()
