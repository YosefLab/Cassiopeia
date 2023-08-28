"""
Test IIDExponentialMLE in cassiopeia.tools.
"""
import math
import unittest

import networkx as nx
import numpy as np
from parameterized import parameterized

from cassiopeia.data import CassiopeiaTree
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
        Since the character matrix is degenerate (it has no mutations),
        an error should be raised.
        """
        tree = nx.DiGraph()
        tree.add_node("0"), tree.add_node("1")
        tree.add_edge("0", "1")
        tree = CassiopeiaTree(tree=tree)
        tree.set_all_character_states({"0": [0], "1": [0]})
        model = IIDExponentialMLE(minimum_branch_length=1e-4, solver=solver)
        with self.assertRaises(ValueError):
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
        Since the character matrix is degenerate (it is saturated),
        an error should be raised.
        """
        tree = nx.DiGraph()
        tree.add_nodes_from(["0", "1"])
        tree.add_edge("0", "1")
        tree = CassiopeiaTree(tree=tree)
        tree.set_all_character_states({"0": [0], "1": [1]})
        model = IIDExponentialMLE(minimum_branch_length=1e-4, solver=solver)
        with self.assertRaises(ValueError):
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
        model = IIDExponentialMLE(minimum_branch_length=1e-4, solver=solver)
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
        model = IIDExponentialMLE(minimum_branch_length=1e-4, solver=solver)
        model.estimate_branch_lengths(tree)
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
        model = IIDExponentialMLE(minimum_branch_length=1e-4, solver=solver)
        model.estimate_branch_lengths(tree)
        self.assertAlmostEqual(tree.get_branch_length("0", "1"), 1.0, places=3)
        self.assertAlmostEqual(model.mutation_rate, np.log(1.5), places=3)
        self.assertAlmostEqual(model.log_likelihood, -1.910, places=3)
        self.assertAlmostEqual(tree.get_time("1"), 1.0, places=3)
        self.assertAlmostEqual(tree.get_time("0"), 0.0, places=3)

    @parameterized.expand([("ECOS", "ECOS"), ("SCS", "SCS")])
    def test_small_tree_with_one_mutation(self, name, solver):
        """
        Perfect binary tree with one mutation at a node 6: Should give very
        short edges 1->3,1->4,0->2.
        The problem can be solved by hand: it trivially reduces to a
        1-dimensional problem:
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
        # Need to make minimum_branch_length be epsilon or else SCS fails...
        model = IIDExponentialMLE(minimum_branch_length=1e-4, solver=solver)
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

        Regression test. Cannot be solved by hand. We just check that this
        solution never changes.
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
                "0": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                "1": [1, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                "2": [0, 0, 0, 0, 0, 6, 0, 0, 0, -1],
                "3": [1, 2, 0, 0, 0, 0, 0, 0, 0, -1],
                "4": [1, 0, 3, 0, 0, 0, 0, 0, 0, -1],
                "5": [0, 0, 0, 0, 5, 6, 7, 0, 0, -1],
                "6": [0, 0, 0, 4, 0, 6, 0, 8, 9, -1],
            }
        )
        model = IIDExponentialMLE(minimum_branch_length=1e-4, solver=solver)
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
        model = IIDExponentialMLE(minimum_branch_length=1e-4, solver=solver)
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
        A subtree with no mutations should collapse to 0. It reduces the
        problem to the same as in 'test_hand_solvable_problem_1'
        """
        tree = nx.DiGraph()
        tree.add_nodes_from(["0", "1", "2", "3", "4"]),
        tree.add_edges_from([("0", "1"), ("1", "2"), ("1", "3"), ("0", "4")])
        tree = CassiopeiaTree(tree=tree)
        tree.set_all_character_states(
            {"0": [0], "1": [1], "2": [1], "3": [1], "4": [0]}
        )
        model = IIDExponentialMLE(minimum_branch_length=1e-4, solver=solver)
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

    @parameterized.expand(
        [
            ("should_pass", [1.5, 2, 2.5], [1.5, 2, 2.5]),
            ("should_pass", [0.05, 0.08, 0.09], [0.05, 0.08, 0.09]),
            ("should_pass", [150, 200, 250], [150, 200, 250]),
            ("should_not_pass", [1.5, 2, 2.5], [1.47, 2, 2.5]),
            ("should_not_pass", [1.5, 2, 2.5], [1.5, 1.97, 2.5]),
            ("should_not_pass", [1.5, 2, 2.5], [1.5, 2, 2.52]),
        ]
    )
    def test_hand_solvable_problem_with_site_rates(
        self, name, solver_rates, math_rates
    ):
        """
        Tree topology is 0->1->2.
        The structure:
            root [state = '000']
            |
            x
            child [state = '100']
            |
            y
            child [state = '110']
        Given the site rates as rate_1, rate_2, and rate_3 respectively
        we find the two branch lengths by solving the MLE expression by hand.
        Prior to rescaling, the first branch is of length
        Ln[(rate_1+rate_2+rate_3)/(rate_2+rate_3)]/rate_1 and the other is
        of length equal to Ln[(rate_2+rate_3)/rate_3]/rate_2.
        """
        rate_1, rate_2, rate_3 = solver_rates
        math_rate_1, math_rate_2, math_rate_3 = math_rates

        tree = nx.DiGraph()
        tree.add_nodes_from(["0", "1", "2"])
        tree.add_edge("0", "1")
        tree.add_edge("1", "2")
        tree = CassiopeiaTree(tree=tree)
        tree.set_all_character_states(
            {"0": [0, 0, 0], "1": [1, 0, 0], "2": [1, 1, 0]}
        )
        relative_rates = [rate_1, rate_2, rate_3]
        model = IIDExponentialMLE(
            minimum_branch_length=1e-4, relative_mutation_rates=relative_rates
        )
        model.estimate_branch_lengths(tree)

        branch1 = (
            math.log(
                (math_rate_1 + math_rate_2 + math_rate_3)
                / (math_rate_2 + math_rate_3)
            )
            / math_rate_1
        )
        branch2 = (
            math.log((math_rate_2 + math_rate_3) / math_rate_3) / math_rate_2
        )
        total = branch1 + branch2
        branch1, branch2 = branch1 / total, branch2 / total
        mutation_rates = [x * total for x in relative_rates]

        should_be_equal = True
        for r1, r2 in zip(solver_rates, math_rates):
            if r1 != r2:
                should_be_equal = False
                break

        if should_be_equal:
            self.assertAlmostEqual(
                tree.get_branch_length("0", "1"), branch1, places=3
            )
            self.assertAlmostEqual(
                tree.get_branch_length("1", "2"), branch2, places=3
            )
            self.assertAlmostEqual(tree.get_time("0"), 0.0, places=3)
            self.assertAlmostEqual(tree.get_time("1"), branch1, places=3)
            self.assertAlmostEqual(tree.get_time("2"), 1.0, places=3)
            for x, y in zip(model.mutation_rate, mutation_rates):
                self.assertAlmostEqual(x, y, places=3)
        else:
            with self.assertRaises(AssertionError):
                self.assertAlmostEqual(
                    tree.get_branch_length("0", "1"), branch1, places=3
                )
            with self.assertRaises(AssertionError):
                self.assertAlmostEqual(
                    tree.get_branch_length("1", "2"), branch2, places=3
                )
            self.assertAlmostEqual(tree.get_time("0"), 0.0, places=3)
            with self.assertRaises(AssertionError):
                self.assertAlmostEqual(tree.get_time("1"), branch1, places=3)
            self.assertAlmostEqual(tree.get_time("2"), 1.0, places=3)
            for x, y in zip(model.mutation_rate, mutation_rates):
                with self.assertRaises(AssertionError):
                    self.assertAlmostEqual(x, y, places=3)

    @parameterized.expand(
        [
            ("negative_rate", [1.5, -1, 2.5]),
            ("zero_rate", [1, 3, 0]),
            ("too_many_rates", [1, 1, 1, 1]),
            ("too_few_rates", [2, 2]),
            ("empty_list", []),
        ]
    )
    def test_invalid_site_rates(self, name, rates):
        """
        Tree topology is the same as test_hand_solvable_problem_with_site_rate
        but rates are misspecified so we should error out.
        """

        tree = nx.DiGraph()
        tree.add_nodes_from(["0", "1", "2"])
        tree.add_edge("0", "1")
        tree.add_edge("1", "2")
        tree = CassiopeiaTree(tree=tree)
        tree.set_all_character_states(
            {"0": [0, 0, 0], "1": [1, 0, 0], "2": [1, 1, 0]}
        )
        relative_rates = rates
        model = IIDExponentialMLE(
            minimum_branch_length=1e-4,
            relative_mutation_rates=relative_rates,
        )
        with self.assertRaises(ValueError):
            model.estimate_branch_lengths(tree)

    @parameterized.expand(
        [
            (
                "should_pass",
                [1.5, 2, 2.5, 1.5, 2, 2.5],
                [1.5, 2, 2.5, 1.5, 2, 2.5],
            ),
            (
                "should_not_pass",
                [1.5, 2, 2.5, 1.5, 2, 2.5],
                [1.52, 1.98, 2.48, 1.52, 2.01, 2.49],
            ),
        ]
    )
    def test_larger_hand_solvable_problem_with_site_rates(
        self, name, solver_rates, math_rates
    ):
        """
        Tree topology is a duplicated version of
        test_hand_solvable_problem_with_site_rates. That is, we double the
        number of characters (while using the same site rates for each pair)
        and decouple each using missing characters as shown below. The expected
        result is the same as the aforementioned test.

        The structure: ('X' indicates missing data)
                   root [state = '0000000']
                    |
                    x
                  child [state = '100100']
                    |
           |------------------|
           y                  z
        [state=            [state=
         XXX110]            110XXX]
        """
        rate_1, rate_2, rate_3, rate_4, rate_5, rate_6 = solver_rates
        (
            math_rate_1,
            math_rate_2,
            math_rate_3,
            _,
            _,
            _,
        ) = math_rates

        tree = nx.DiGraph()
        tree.add_nodes_from(["0", "1", "2", "3"])
        tree.add_edge("0", "1")
        tree.add_edge("1", "2")
        tree.add_edge("1", "3")
        tree = CassiopeiaTree(tree=tree)
        tree.set_all_character_states(
            {
                "0": [0, 0, 0, 0, 0, 0],
                "1": [1, 0, 0, 1, 0, 0],
                "2": [-1, -1, -1, 1, 1, 0],
                "3": [1, 1, 0, -1, -1, -1],
            }
        )
        relative_rates = [rate_1, rate_2, rate_3, rate_4, rate_5, rate_6]
        model = IIDExponentialMLE(
            minimum_branch_length=1e-4, relative_mutation_rates=relative_rates
        )
        model.estimate_branch_lengths(tree)

        branch1 = (
            math.log(
                (math_rate_1 + math_rate_2 + math_rate_3)
                / (math_rate_2 + math_rate_3)
            )
            / math_rate_1
        )
        branch2 = (
            math.log((math_rate_2 + math_rate_3) / math_rate_3) / math_rate_2
        )
        total = branch1 + branch2
        branch1, branch2 = branch1 / total, branch2 / total
        mutation_rates = [x * total for x in relative_rates]

        should_be_equal = True
        for r1, r2 in zip(solver_rates, math_rates):
            if r1 != r2:
                should_be_equal = False
                break

        if should_be_equal:
            self.assertAlmostEqual(
                tree.get_branch_length("0", "1"), branch1, places=3
            )
            self.assertAlmostEqual(
                tree.get_branch_length("1", "2"), branch2, places=3
            )
            self.assertAlmostEqual(
                tree.get_branch_length("1", "3"), branch2, places=3
            )
            self.assertAlmostEqual(tree.get_time("0"), 0.0, places=3)
            self.assertAlmostEqual(tree.get_time("1"), branch1, places=3)
            self.assertAlmostEqual(tree.get_time("2"), 1.0, places=3)

            for x, y in zip(model.mutation_rate, mutation_rates):
                self.assertAlmostEqual(x, y, places=3)
        else:
            with self.assertRaises(AssertionError):
                self.assertAlmostEqual(
                    tree.get_branch_length("0", "1"), branch1, places=3
                )
            with self.assertRaises(AssertionError):
                self.assertAlmostEqual(
                    tree.get_branch_length("1", "2"), branch2, places=3
                )
            with self.assertRaises(AssertionError):
                self.assertAlmostEqual(
                    tree.get_branch_length("1", "3"), branch2, places=3
                )
            self.assertAlmostEqual(tree.get_time("0"), 0.0, places=3)
            with self.assertRaises(AssertionError):
                self.assertAlmostEqual(tree.get_time("1"), branch1, places=3)
            self.assertAlmostEqual(tree.get_time("2"), 1.0, places=3)

            for x, y in zip(model.mutation_rate, mutation_rates):
                with self.assertRaises(AssertionError):
                    self.assertAlmostEqual(x, y, places=3)
