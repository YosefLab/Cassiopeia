"""
Test IIDExponentialMLE in Cassiopeia.tools.
"""
import unittest

import networkx as nx
import numpy as np
from parameterized import parameterized

from cassiopeia.data import CassiopeiaTree
from cassiopeia.mixins import IIDExponentialMLEError
from cassiopeia.simulator import Cas9LineageTracingDataSimulator
from cassiopeia.tools import IIDExponentialMLE


class TestIIDExponentialMLE(unittest.TestCase):
    @parameterized.expand([("ECOS", "ECOS"), ("SCS", "SCS")])
    def test_no_mutations(self, name, solver):
        """
        Tree topology is just a branch 0->1.
        There is one unmutated character i.e.:
            root [state = '0']
            |
            v
            child [state = '0']
        This is thus the simplest possible example of no mutations, and the MLE
        branch length should be 0, so it should error out.
        """
        tree = nx.DiGraph()
        tree.add_node("0"), tree.add_node("1")
        tree.add_edge("0", "1")
        tree = CassiopeiaTree(tree=tree)
        tree.set_all_character_states({"0": [0], "1": [0]})
        model = IIDExponentialMLE(minimum_branch_length=0.0, solver=solver)
        with self.assertRaises(IIDExponentialMLEError):
            model.estimate_branch_lengths(tree)

    @parameterized.expand([("ECOS", "ECOS"), ("SCS", "SCS")])
    def test_saturation(self, name, solver):
        """
        Tree topology is just a branch 0->1.
        There is one mutated character i.e.:
            root [state = '0']
            |
            v
            child [state = '1']
        This is thus the simplest possible example of saturation, and the MLE
        mutation rate should be infinity, so should error out.
        """
        tree = nx.DiGraph()
        tree.add_nodes_from(["0", "1"])
        tree.add_edge("0", "1")
        tree = CassiopeiaTree(tree=tree)
        tree.set_all_character_states({"0": [0], "1": [1]})
        model = IIDExponentialMLE(minimum_branch_length=0.0, solver=solver)
        with self.assertRaises(IIDExponentialMLEError):
            model.estimate_branch_lengths(tree)

    @parameterized.expand([("ECOS", "ECOS"), ("SCS", "SCS")])
    def test_hand_solvable_problem_1(self, name, solver):
        """
        Tree topology is just a branch 0->1.
        There is one mutated character and one unmutated character, i.e.:
            root [state = '00']
            |
            v
            child [state = '01']
        The solution can be verified by hand. The optimization problem is:
            min_{r * t0} log(exp(-r * t0)) + log(1 - exp(-r * t0))
        The solution is r * t0 = ln(2) ~ 0.693
        (Note that because the depth of the tree is fixed to 1, r * t0 = r * 1
        is the mutation rate.)
        """
        tree = nx.DiGraph()
        tree.add_nodes_from(["0", "1"])
        tree.add_edge("0", "1")
        tree = CassiopeiaTree(tree=tree)
        tree.set_all_character_states({"0": [0, 0], "1": [0, 1]})
        model = IIDExponentialMLE(minimum_branch_length=0.0, solver=solver)
        model.estimate_branch_lengths(tree)
        log_likelihood = model.log_likelihood
        self.assertAlmostEqual(model.mutation_rate, np.log(2), places=3)
        self.assertAlmostEqual(tree.get_branch_length("0", "1"), 1.0, places=3)
        self.assertAlmostEqual(tree.get_time("1"), 1, places=3)
        self.assertAlmostEqual(tree.get_time("0"), 0.0, places=3)
        self.assertAlmostEqual(log_likelihood, -1.386, places=3)

    @parameterized.expand([("ECOS", "ECOS"), ("SCS", "SCS")])
    def test_hand_solvable_problem_2(self, name, solver):
        """
        Tree topology is just a branch 0->1.
        There are two mutated characters and one unmutated character, i.e.:
            root [state = '000']
            |
            v
            child [state = '011']
        The solution can be verified by hand. The optimization problem is:
            min_{r * t0} log(exp(-r * t0)) + 2 * log(1 - exp(-r * t0))
        The solution is r * t0 = ln(3) ~ 1.098
        (Note that because the depth of the tree is fixed to 1, r * t0 = r * 1
        is the mutation rate.)
        """
        tree = nx.DiGraph()
        tree.add_nodes_from(["0", "1"])
        tree.add_edge("0", "1")
        tree = CassiopeiaTree(tree=tree)
        tree.set_all_character_states({"0": [0, 0, 0], "1": [0, 1, 1]})
        model = IIDExponentialMLE(minimum_branch_length=0.0, solver=solver)
        model.estimate_branch_lengths(tree)
        log_likelihood = model.log_likelihood
        self.assertAlmostEqual(tree.get_branch_length("0", "1"), 1.0, places=3)
        self.assertAlmostEqual(tree.get_time("1"), 1.0, places=3)
        self.assertAlmostEqual(tree.get_time("0"), 0.0, places=3)
        self.assertAlmostEqual(model.mutation_rate, np.log(3), places=3)
        self.assertAlmostEqual(model.log_likelihood, -1.910, places=3)

    @parameterized.expand([("ECOS", "ECOS"), ("SCS", "SCS")])
    def test_hand_solvable_problem_3(self, name, solver):
        """
        Tree topology is just a branch 0->1.
        There are two unmutated characters and one mutated character, i.e.:
            root [state = '000']
            |
            v
            child [state = '001']
        The solution can be verified by hand. The optimization problem is:
            min_{r * t0} 2 * log(exp(-r * t0)) + log(1 - exp(-r * t0))
        The solution is r * t0 = ln(1.5) ~ 0.405
        (Note that because the depth of the tree is fixed to 1, r * t0 = r * 1
        is the mutation rate.)
        """
        tree = nx.DiGraph()
        tree.add_nodes_from(["0", "1"])
        tree.add_edge("0", "1")
        tree = CassiopeiaTree(tree=tree)
        tree.set_all_character_states({"0": [0, 0, 0], "1": [0, 0, 1]})
        model = IIDExponentialMLE(minimum_branch_length=0.0, solver=solver)
        model.estimate_branch_lengths(tree)
        self.assertAlmostEqual(tree.get_branch_length("0", "1"), 1.0, places=3)
        self.assertAlmostEqual(model.mutation_rate, np.log(1.5), places=3)
        self.assertAlmostEqual(model.log_likelihood, -1.910, places=3)
        self.assertAlmostEqual(tree.get_time("1"), 1.0, places=3)
        self.assertAlmostEqual(tree.get_time("0"), 0.0, places=3)

    @parameterized.expand([("ECOS", "ECOS"), ("SCS", "SCS")])
    def test_small_tree_with_one_mutation(self, name, solver):
        """
        Perfect binary tree with one mutation at a node 6: Should give very short
        edges 1->3,1->4,0->2.
        The problem can be solved by hand: it trivially reduces to a 1-dimensional
        problem:
            min_{r * t0} 2 * log(exp(-r * t0)) + log(1 - exp(-r * t0))
        The solution is r * t0 = ln(1.5) ~ 0.405
        (Note that because the depth of the tree is fixed to 1, r * t0 = r * 1
        is the mutation rate.)
        """
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
        tree.set_all_character_states(
            {
                "0": [0],
                "1": [0],
                "2": [0],
                "3": [0],
                "4": [0],
                "5": [0],
                "6": [1],
            }
        )
        model = IIDExponentialMLE(minimum_branch_length=0.0, solver=solver)
        model.estimate_branch_lengths(tree)
        self.assertAlmostEqual(tree.get_branch_length("0", "1"), 1.0, places=3)
        self.assertAlmostEqual(tree.get_branch_length("0", "2"), 0.0, places=3)
        self.assertAlmostEqual(tree.get_branch_length("1", "3"), 0.0, places=3)
        self.assertAlmostEqual(tree.get_branch_length("1", "4"), 0.0, places=3)
        self.assertAlmostEqual(tree.get_branch_length("2", "5"), 1.0, places=3)
        self.assertAlmostEqual(tree.get_branch_length("2", "6"), 1.0, places=3)
        self.assertAlmostEqual(model.log_likelihood, -1.910, places=3)
        self.assertAlmostEqual(model.mutation_rate, np.log(1.5), places=3)

    @parameterized.expand([("ECOS", "ECOS"), ("SCS", "SCS")])
    def test_small_tree_regression(self, name, solver):
        """
        Perfect binary tree with "normal" amount of mutations on each edge.

        Regression test. Cannot be solved by hand. We just check that this solution
        never changes.
        """
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
        tree.set_all_character_states(
            {
                "0": [0, 0, 0, 0, 0, 0, 0, 0, 0],
                "1": [1, 0, 0, 0, 0, 0, 0, 0, 0],
                "2": [0, 0, 0, 0, 0, 6, 0, 0, 0],
                "3": [1, 2, 0, 0, 0, 0, 0, 0, 0],
                "4": [1, 0, 3, 0, 0, 0, 0, 0, 0],
                "5": [0, 0, 0, 0, 5, 6, 7, 0, 0],
                "6": [0, 0, 0, 4, 0, 6, 0, 8, 9],
            }
        )
        model = IIDExponentialMLE(minimum_branch_length=0.0, solver=solver)
        model.estimate_branch_lengths(tree)
        self.assertAlmostEqual(model.mutation_rate, 0.378, places=3)
        self.assertAlmostEqual(
            tree.get_branch_length("0", "1"), 0.537, places=3
        )
        self.assertAlmostEqual(
            tree.get_branch_length("0", "2"), 0.219, places=3
        )
        self.assertAlmostEqual(
            tree.get_branch_length("1", "3"), 0.463, places=3
        )
        self.assertAlmostEqual(
            tree.get_branch_length("1", "4"), 0.463, places=3
        )
        self.assertAlmostEqual(
            tree.get_branch_length("2", "5"), 0.781, places=3
        )
        self.assertAlmostEqual(
            tree.get_branch_length("2", "6"), 0.781, places=3
        )
        self.assertAlmostEqual(model.log_likelihood, -22.689, places=3)

    @parameterized.expand([("ECOS", "ECOS"), ("SCS", "SCS")])
    def test_on_simulated_data(self, name, solver):
        """
        We run the estimator on data simulated under the correct model.
        The estimator should be close to the ground truth.
        """
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
        Cas9LineageTracingDataSimulator(
            number_of_cassettes=200,
            size_of_cassette=1,
            mutation_rate=1.5,
        ).overlay_data(tree)
        model = IIDExponentialMLE(minimum_branch_length=0.0, solver=solver)
        model.estimate_branch_lengths(tree)
        self.assertTrue(0.05 < tree.get_time("1") < 0.15)
        self.assertTrue(0.8 < tree.get_time("2") < 1.0)
        self.assertTrue(0.9 < tree.get_time("3") < 1.1)
        self.assertTrue(0.9 < tree.get_time("4") < 1.1)
        self.assertTrue(0.9 < tree.get_time("5") < 1.1)
        self.assertTrue(0.9 < tree.get_time("6") < 1.1)
        self.assertTrue(1.4 < model.mutation_rate < 1.6)
        self.assertAlmostEqual(tree.get_time("0"), 0.0, places=3)

    @parameterized.expand([("ECOS", "ECOS"), ("SCS", "SCS")])
    def test_subtree_collapses_when_no_mutations(self, name, solver):
        """
        A subtree with no mutations should collapse to 0. It reduces the problem to
        the same as in 'test_hand_solvable_problem_1'
        """
        tree = nx.DiGraph()
        tree.add_nodes_from(["0", "1", "2", "3", "4"]),
        tree.add_edges_from([("0", "1"), ("1", "2"), ("1", "3"), ("0", "4")])
        tree = CassiopeiaTree(tree=tree)
        tree.set_all_character_states(
            {"0": [0], "1": [1], "2": [1], "3": [1], "4": [0]}
        )
        model = IIDExponentialMLE(minimum_branch_length=0.0, solver=solver)
        model.estimate_branch_lengths(tree)
        self.assertAlmostEqual(model.log_likelihood, -1.386, places=3)
        self.assertAlmostEqual(tree.get_branch_length("0", "1"), 1.0, places=3)
        self.assertAlmostEqual(tree.get_branch_length("1", "2"), 0.0, places=3)
        self.assertAlmostEqual(tree.get_branch_length("1", "3"), 0.0, places=3)
        self.assertAlmostEqual(tree.get_branch_length("0", "4"), 1.0, places=3)
        self.assertAlmostEqual(model.mutation_rate, np.log(2), places=3)

    @parameterized.expand([("ECOS", "ECOS"), ("SCS", "SCS")])
    def test_minimum_branch_length(self, name, solver):
        """
        Test that the minimum branch length feature works.

        Same as test_small_tree_with_one_mutation but now we constrain the
        minimum branch length.Should give very short edges 1->3,1->4,0->2
        and edges 0->1,2->5,2->6 close to 1.
        """
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
        tree.set_all_character_states(
            {
                "0": [0],
                "1": [0],
                "2": [0],
                "3": [0],
                "4": [0],
                "5": [0],
                "6": [1],
            }
        )
        model = IIDExponentialMLE(minimum_branch_length=0.01, solver=solver)
        model.estimate_branch_lengths(tree)
        self.assertAlmostEqual(
            tree.get_branch_length("0", "1"), 0.990, places=3
        )
        self.assertAlmostEqual(
            tree.get_branch_length("0", "2"), 0.010, places=3
        )
        self.assertAlmostEqual(
            tree.get_branch_length("1", "3"), 0.010, places=3
        )
        self.assertAlmostEqual(
            tree.get_branch_length("1", "4"), 0.010, places=3
        )
        self.assertAlmostEqual(
            tree.get_branch_length("2", "5"), 0.990, places=3
        )
        self.assertAlmostEqual(
            tree.get_branch_length("2", "6"), 0.990, places=3
        )
        self.assertAlmostEqual(model.log_likelihood, -1.922, places=3)
        self.assertAlmostEqual(model.mutation_rate, 0.405, places=3)
