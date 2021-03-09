import unittest

import networkx as nx
import numpy as np

from cassiopeia.simulator.BirthDeathFitnessSimulator import (
    BirthDeathFitnessSimulator,
)
from cassiopeia.simulator.BirthDeathFitnessSimulator import (
    BirthDeathFitnessError,
)


def get_leaves(tree):
    return [n for n in tree.nodes if tree.out_degree(n) == 0]


def test_tree(tree):
    tree = tree.copy()
    tree.nodes[0]["total_time"] = 0
    for i in nx.edge_dfs(tree):
        tree.nodes[i[1]]["total_time"] = (
            tree.nodes[i[0]]["total_time"] + tree.edges[i]["weight"]
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
    def test_remove_and_prune_lineage_all(self):
        bd_sim = BirthDeathFitnessSimulator()
        tree = nx.balanced_tree(2, 3, create_using=nx.DiGraph())
        leaves = get_leaves(tree)
        for i in leaves:
            bd_sim.remove_and_prune_lineage(i, tree)

        self.assertEqual(list(tree.edges), [])

    def test_remove_and_prune_lineage_1(self):
        bd_sim = BirthDeathFitnessSimulator()
        tree = nx.balanced_tree(2, 3, create_using=nx.DiGraph())
        bd_sim.remove_and_prune_lineage(11, tree)
        bd_sim.remove_and_prune_lineage(13, tree)
        bd_sim.remove_and_prune_lineage(14, tree)

        expected_edges = [
            (0, 1),
            (0, 2),
            (1, 3),
            (1, 4),
            (2, 5),
            (3, 7),
            (3, 8),
            (4, 9),
            (4, 10),
            (5, 12),
        ]
        self.assertEqual(list(tree.edges), expected_edges)

    def test_remove_and_prune_lineage_2(self):
        bd_sim = BirthDeathFitnessSimulator()
        tree = nx.balanced_tree(2, 3, create_using=nx.DiGraph())
        for i in range(7, 11):
            bd_sim.remove_and_prune_lineage(i, tree)

        expected_edges = [
            (0, 2),
            (2, 5),
            (2, 6),
            (5, 11),
            (5, 12),
            (6, 13),
            (6, 14),
        ]
        self.assertEqual(list(tree.edges), expected_edges)

    def test_collapse_unifurcations_source(self):
        bd_sim = BirthDeathFitnessSimulator()
        tree = nx.DiGraph()
        tree.add_nodes_from(list(range(6)))
        tree.add_edges_from([(0, 1), (1, 2), (2, 3), (2, 4), (3, 5)])
        for i in tree.edges:
            tree.edges[i]["weight"] = 1.5

        bd_sim.collapse_unifurcations(tree, source=1)

        expected_edges = [
            (0, 1, {"weight": 1.5}),
            (1, 4, {"weight": 3.0}),
            (1, 5, {"weight": 4.5}),
        ]
        self.assertEqual(list(tree.edges(data=True)), expected_edges)

    def test_collapse_unifurcations_1(self):
        bd_sim = BirthDeathFitnessSimulator()
        tree = nx.DiGraph()
        tree.add_nodes_from(list(range(10)))
        tree.add_edges_from(
            [
                (0, 1),
                (0, 2),
                (2, 3),
                (3, 4),
                (2, 5),
                (5, 6),
                (6, 7),
                (6, 8),
                (2, 9),
            ]
        )
        for i in tree.edges:
            tree.edges[i]["weight"] = 1.5

        bd_sim.collapse_unifurcations(tree)
        expected_edges = [
            (0, 1, {"weight": 1.5}),
            (0, 2, {"weight": 1.5}),
            (2, 9, {"weight": 1.5}),
            (2, 4, {"weight": 3.0}),
            (2, 6, {"weight": 3.0}),
            (6, 7, {"weight": 1.5}),
            (6, 8, {"weight": 1.5}),
        ]
        self.assertEqual(list(tree.edges(data=True)), expected_edges)

    def test_collapse_unifurcations_2(self):
        bd_sim = BirthDeathFitnessSimulator()
        tree = nx.DiGraph()
        tree.add_nodes_from(list(range(15)))
        tree.add_edges_from(
            [
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4),
                (3, 5),
                (4, 6),
                (6, 7),
                (6, 8),
                (5, 9),
                (5, 10),
                (5, 11),
                (10, 12),
                (12, 13),
                (13, 14),
            ]
        )
        for i in tree.edges:
            tree.edges[i]["weight"] = 1.5

        bd_sim.collapse_unifurcations(tree)

        expected_edges = [
            (0, 5, {"weight": 6.0}),
            (0, 6, {"weight": 7.5}),
            (5, 9, {"weight": 1.5}),
            (5, 11, {"weight": 1.5}),
            (5, 14, {"weight": 6.0}),
            (6, 7, {"weight": 1.5}),
            (6, 8, {"weight": 1.5}),
        ]
        self.assertEqual(list(tree.edges(data=True)), expected_edges)

    def test_bad_waiting_distributions(self):
        bd_sim = BirthDeathFitnessSimulator()
        with self.assertRaises(BirthDeathFitnessError):
            birth_waiting_dist = lambda _: -1
            birth_scale_param = 1

            tree = bd_sim.simulate_tree(
                birth_waiting_dist, birth_scale_param, experiment_time=1
            )

        with self.assertRaises(BirthDeathFitnessError):
            birth_waiting_dist = lambda _: 0
            birth_scale_param = 1

            tree = bd_sim.simulate_tree(
                birth_waiting_dist, birth_scale_param, num_extant=4
            )

        with self.assertRaises(BirthDeathFitnessError):
            birth_waiting_dist = lambda _: 1
            death_waiting_dist = lambda: -1
            birth_scale_param = 1

            tree = bd_sim.simulate_tree(
                birth_waiting_dist,
                birth_scale_param,
                death_waiting_dist=death_waiting_dist,
                num_extant=1,
            )

        with self.assertRaises(BirthDeathFitnessError):
            birth_waiting_dist = lambda _: 1
            death_waiting_dist = lambda: 0
            birth_scale_param = 1

            tree = bd_sim.simulate_tree(
                birth_waiting_dist,
                birth_scale_param,
                death_waiting_dist=death_waiting_dist,
                experiment_time=1,
            )

    def test_bad_stopping_conditions(self):
        bd_sim = BirthDeathFitnessSimulator()
        birth_waiting_dist = lambda _: 1
        death_waiting_dist = lambda: 2
        birth_scale_param = 1

        with self.assertRaises(BirthDeathFitnessError):
            tree = bd_sim.simulate_tree(
                birth_waiting_dist,
                birth_scale_param,
                death_waiting_dist=death_waiting_dist,
            )

        with self.assertRaises(BirthDeathFitnessError):
            tree = bd_sim.simulate_tree(
                birth_waiting_dist,
                birth_scale_param,
                death_waiting_dist=death_waiting_dist,
                num_extant=4,
                experiment_time=4,
            )

        with self.assertRaises(BirthDeathFitnessError):
            tree = bd_sim.simulate_tree(
                birth_waiting_dist,
                birth_scale_param,
                death_waiting_dist=death_waiting_dist,
                experiment_time=-1,
            )

        with self.assertRaises(BirthDeathFitnessError):
            tree = bd_sim.simulate_tree(
                birth_waiting_dist,
                birth_scale_param,
                death_waiting_dist=death_waiting_dist,
                num_extant=0,
            )

        with self.assertRaises(BirthDeathFitnessError):
            tree = bd_sim.simulate_tree(
                birth_waiting_dist,
                birth_scale_param,
                death_waiting_dist=death_waiting_dist,
                num_extant=-1,
            )

    def test_dead_at_start(self):
        bd_sim = BirthDeathFitnessSimulator()
        birth_waiting_dist = lambda _: 2
        death_waiting_dist = lambda: 1
        birth_scale_param = 1

        with self.assertRaises(BirthDeathFitnessError):
            tree = bd_sim.simulate_tree(
                birth_waiting_dist,
                birth_scale_param,
                death_waiting_dist=death_waiting_dist,
                num_extant=4,
            )

        with self.assertRaises(BirthDeathFitnessError):
            tree = bd_sim.simulate_tree(
                birth_waiting_dist,
                birth_scale_param,
                death_waiting_dist=death_waiting_dist,
                experiment_time=4,
            )

    def test_single_lineage(self):
        bd_sim = BirthDeathFitnessSimulator()
        birth_waiting_dist = lambda _: 1
        birth_scale_param = 1

        tree = bd_sim.simulate_tree(
            birth_waiting_dist, birth_scale_param, num_extant=1
        )
        results = test_tree(tree.get_tree_topology())
        self.assertEqual(results[1], 1)
        self.assertEqual(
            tree.get_tree_topology().get_edge_data(0, 1)["weight"], 1.0
        )
        self.assertEqual(results[0], [1])

        tree = bd_sim.simulate_tree(
            birth_waiting_dist, birth_scale_param, experiment_time=1
        )
        self.assertEqual(results[1], 1)
        self.assertEqual(
            tree.get_tree_topology().get_edge_data(0, 1)["weight"], 1.0
        )
        self.assertEqual(results[0], [1])

    def test_constant_yule(self):
        bd_sim = BirthDeathFitnessSimulator()
        birth_waiting_dist = lambda _: 1
        birth_scale_param = 1

        tree = bd_sim.simulate_tree(
            birth_waiting_dist, birth_scale_param, num_extant=32
        )

        results = test_tree(tree.get_tree_topology())
        for i in results[0]:
            self.assertEqual(i, 6)
        self.assertEqual(results[1], 32)
        self.assertTrue(results[2])

        tree = bd_sim.simulate_tree(
            birth_waiting_dist, birth_scale_param, experiment_time=6
        )

        results = test_tree(tree.get_tree_topology())
        for i in results[0]:
            self.assertEqual(i, 6)
        self.assertEqual(results[1], 32)
        self.assertTrue(results[2])

    def test_nonconstant_yule(self):
        bd_sim = BirthDeathFitnessSimulator()
        birth_waiting_dist = lambda scale: np.random.exponential(scale)
        birth_scale_param = 1

        tree = bd_sim.simulate_tree(
            birth_waiting_dist, birth_scale_param, num_extant=16
        )

        tree_top = tree.get_tree_topology()
        results = test_tree(tree_top)
        self.assertTrue(all(np.isclose(x, results[0][0]) for x in results[0]))
        self.assertEqual(results[1], 16)
        self.assertTrue(results[2])
        self.assertEqual(max(tree.nodes), 31)

        tree = bd_sim.simulate_tree(
            birth_waiting_dist, birth_scale_param, experiment_time=2
        )

        tree_top = tree.get_tree_topology()
        results = test_tree(tree_top)
        for i in results[0]:
            self.assertEqual(i, 2)
        self.assertTrue(results[2])

    def test_nonconstant_birth_death(self):
        bd_sim = BirthDeathFitnessSimulator()
        np.random.seed(1234)
        birth_waiting_dist = lambda scale: np.random.exponential(scale)
        death_waiting_dist = lambda: np.random.exponential(1.5)
        birth_scale_param = 0.5

        tree = bd_sim.simulate_tree(
            birth_waiting_dist,
            birth_scale_param,
            death_waiting_dist,
            num_extant=8,
        )

        tree_top = tree.get_tree_topology()
        results = test_tree(tree_top)
        self.assertTrue(all(np.isclose(x, results[0][0]) for x in results[0]))
        self.assertEqual(results[1], 8)
        self.assertTrue(results[2])
        self.assertNotIn(9, tree_top.nodes)
        self.assertNotIn(2, tree_top.nodes)

        np.random.seed(1234)
        tree = bd_sim.simulate_tree(
            birth_waiting_dist,
            birth_scale_param,
            death_waiting_dist,
            experiment_time=2,
        )

        tree_top = tree.get_tree_topology()
        results = test_tree(tree_top)
        for i in results[0]:
            self.assertTrue(np.isclose(i, 2))
        self.assertTrue(results[2])
        self.assertNotIn(10, tree_top.nodes)
        self.assertNotIn(2, tree_top.nodes)

    def test_nonconstant_yule_with_predictable_fitness(self):
        bd_sim = BirthDeathFitnessSimulator()
        np.random.seed(1234)
        birth_waiting_dist = lambda scale: np.random.exponential(scale)
        birth_scale_param = 0.5
        fitness_num_dist = lambda: 2
        fitness_strength_dist = lambda: 0.98

        tree = bd_sim.simulate_tree(
            birth_waiting_dist,
            birth_scale_param,
            fitness_num_dist=fitness_num_dist,
            fitness_strength_dist=fitness_strength_dist,
            num_extant=8,
        )

        tree_top = tree.get_tree_topology()
        results = test_tree(tree_top)
        self.assertTrue(all(np.isclose(x, results[0][0]) for x in results[0]))
        self.assertEqual(results[1], 8)
        self.assertTrue(results[2])

        for u, v in tree_top.edges:
            tree_top[u][v]["val"] = 1
        tree_top.nodes[0]["depth"] = 0
        for u, v in nx.dfs_edges(tree_top, source=0):
            tree_top.nodes[v]["depth"] = (
                tree_top.nodes[u]["depth"] + tree_top[u][v]["val"]
            )
        leaves = get_leaves(tree_top)
        for i in tree_top.nodes:
            if i in leaves:
                self.assertTrue(
                    np.isclose(
                        tree_top.nodes[i]["birth_scale"],
                        0.5 * 0.98 ** (2 * (tree_top.nodes[i]["depth"] - 1)),
                    )
                )
            else:
                self.assertTrue(
                    np.isclose(
                        tree_top.nodes[i]["birth_scale"],
                        0.5 * 0.98 ** (2 * tree_top.nodes[i]["depth"]),
                    )
                )

        np.random.seed(1234)

        tree = bd_sim.simulate_tree(
            birth_waiting_dist,
            birth_scale_param,
            fitness_num_dist=fitness_num_dist,
            fitness_strength_dist=fitness_strength_dist,
            experiment_time=0.6,
        )

        tree_top = tree.get_tree_topology()
        results = test_tree(tree_top)
        for i in results[0]:
            self.assertTrue(np.isclose(i, 0.6))
        self.assertTrue(results[2])

        for u, v in tree_top.edges:
            tree_top[u][v]["val"] = 1
        tree_top.nodes[0]["depth"] = 0
        for u, v in nx.dfs_edges(tree_top, source=0):
            tree_top.nodes[v]["depth"] = (
                tree_top.nodes[u]["depth"] + tree_top[u][v]["val"]
            )
        leaves = get_leaves(tree_top)
        for i in tree_top.nodes:
            if i in leaves:
                self.assertTrue(
                    np.isclose(
                        tree_top.nodes[i]["birth_scale"],
                        0.5 * 0.98 ** (2 * (tree_top.nodes[i]["depth"] - 1)),
                    )
                )
            else:
                self.assertTrue(
                    np.isclose(
                        tree_top.nodes[i]["birth_scale"],
                        0.5 * 0.98 ** (2 * tree_top.nodes[i]["depth"]),
                    )
                )

    def test_nonconstant_birth_death_with_variable_fitness(self):
        bd_sim = BirthDeathFitnessSimulator()
        np.random.seed(12364)
        birth_waiting_dist = lambda scale: np.random.exponential(scale)
        death_waiting_dist = lambda: np.random.exponential(0.6)
        birth_scale_param = 0.5
        fitness_num_dist = lambda: 1 if np.random.uniform() < 0.2 else 0
        fitness_strength_dist = lambda: 1.5 ** np.random.uniform(-1, 1)

        tree = bd_sim.simulate_tree(
            birth_waiting_dist,
            birth_scale_param,
            death_waiting_dist,
            fitness_num_dist=fitness_num_dist,
            fitness_strength_dist=fitness_strength_dist,
            num_extant=8,
        )

        np.random.seed(12364)

        tree_top = tree.get_tree_topology()
        results = test_tree(tree_top)
        self.assertTrue(all(np.isclose(x, results[0][0]) for x in results[0]))
        self.assertEqual(results[1], 8)
        self.assertTrue(results[2])
        self.assertNotIn(2, tree_top.nodes)
        self.assertNotIn(3, tree_top.nodes)

        tree = bd_sim.simulate_tree(
            birth_waiting_dist,
            birth_scale_param,
            death_waiting_dist,
            fitness_num_dist=fitness_num_dist,
            fitness_strength_dist=fitness_strength_dist,
            experiment_time=3,
        )

        tree_top = tree.get_tree_topology()
        results = test_tree(tree_top)
        for i in results[0]:
            self.assertTrue(np.isclose(i, 3))
        self.assertTrue(results[2])
        self.assertNotIn(2, tree_top.nodes)
        self.assertNotIn(3, tree_top.nodes)


if __name__ == "__main__":
    unittest.main()
