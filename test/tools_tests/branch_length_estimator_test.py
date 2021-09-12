import itertools
import multiprocessing
import unittest
from copy import deepcopy
from typing import List

import networkx as nx
import numpy as np
import pytest
from parameterized import parameterized

from cassiopeia.data import CassiopeiaTree
from cassiopeia.tools import (BirthProcess, BLEMultifurcationWrapper,
                              IgnoreCharactersWrapper,
                              IIDExponentialBLE,
                              IIDExponentialBLEGridSearchCV,
                              IIDExponentialLineageTracer,
                              IIDExponentialPosteriorMeanBLEGridSearchCV,
                              NumberOfMutationsBLE,
                              UniformCellSubsampler,
                              EmptySubtreeError,
                              BLEEnsemble)


from branch_length_estimator_tests.iid_exponential_bayesian_test import calc_numerical_log_likelihood

class TestBLEEnsemble(unittest.TestCase):
    def test_basic(self):
        r"""
        Just tests that BLEEnsemble runs.
        """
        tree = nx.DiGraph()
        tree.add_nodes_from(["0", "1", "2", "3", "4", "5", "6", "7"]),
        tree.add_edges_from([("0", "1"), ("0", "2"), ("1", "3"), ("1", "4"),
                            ("2", "5"), ("2", "6"), ("3", "7")])
        tree = CassiopeiaTree(tree=tree)
        tree.set_all_character_states(
            {"0": [0, 0, 0],
            "1": [0, 0, -1],
            "2": [1, 0, 0],
            "3": [0, 1, -1],
            "4": [1, 1, -1],
            "5": [1, -1, -1],
            "6": [1, -1, 0],
            "7": [1, 1, -1],}
        )
        model = BLEEnsemble(
            branch_length_estimators=
                [
                    NumberOfMutationsBLE(
                        length_of_mutationless_edges=0.5,
                        treat_missing_states_as_mutations=False,
                    ),
                    NumberOfMutationsBLE(
                        length_of_mutationless_edges=0.5,
                        treat_missing_states_as_mutations=True,
                    ),
                ]
        )
        model.estimate_branch_lengths(tree)


class TestIgnoreCharactersWrapper(unittest.TestCase):
    def test_basic(self):
        tree = nx.DiGraph()
        tree.add_nodes_from(["0", "1", "2", "3", "4", "5", "6", "7"]),
        tree.add_edges_from([("0", "1"), ("0", "2"), ("1", "3"), ("1", "4"),
                            ("2", "5"), ("2", "6"), ("3", "7")])
        tree = CassiopeiaTree(tree=tree)
        tree.set_all_character_states(
            {"0": [0, 0, 0],
            "1": [0, 0, -1],
            "2": [1, 0, 0],
            "3": [0, 1, -1],
            "4": [1, 1, -1],
            "5": [1, -1, -1],
            "6": [1, -1, 0],
            "7": [1, 1, -1],}
        )
        model = IgnoreCharactersWrapper(
            NumberOfMutationsBLE(
                length_of_mutationless_edges=0.5,
                treat_missing_states_as_mutations=False,
            )
        )
        model.estimate_branch_lengths(tree)
        np.testing.assert_almost_equal(tree.get_branch_length("0", "1"), 0.5, decimal=3)
        np.testing.assert_almost_equal(tree.get_branch_length("0", "2"), 0.5, decimal=3)
        np.testing.assert_almost_equal(tree.get_branch_length("1", "3"), 0.5, decimal=3)
        np.testing.assert_almost_equal(tree.get_branch_length("1", "4"), 1.0, decimal=3)
        np.testing.assert_almost_equal(tree.get_branch_length("2", "5"), 1.0, decimal=3)
        np.testing.assert_almost_equal(tree.get_branch_length("2", "6"), 1.0, decimal=3)
        np.testing.assert_almost_equal(tree.get_branch_length("3", "7"), 0.5, decimal=3)



class TestNumberOfMutationsBLE(unittest.TestCase):
    def test_basic_missing_ignored(self):
        tree = nx.DiGraph()
        tree.add_nodes_from(["0", "1", "2", "3", "4", "5", "6"]),
        tree.add_edges_from([("0", "1"), ("0", "2"), ("1", "3"), ("1", "4"),
                            ("2", "5"), ("2", "6")])
        tree = CassiopeiaTree(tree=tree)
        tree.set_all_character_states(
            {"0": [0, 0, 0],
            "1": [0, 0, -1],
            "2": [1, 0, 0],
            "3": [0, 1, -1],
            "4": [1, 1, -1],
            "5": [1, -1, -1],
            "6": [1, -1, 0]}
        )
        model = NumberOfMutationsBLE(length_of_mutationless_edges=0.5, treat_missing_states_as_mutations=False)
        model.estimate_branch_lengths(tree)
        np.testing.assert_almost_equal(tree.get_branch_length("0", "1"), 0.5, decimal=3)
        np.testing.assert_almost_equal(tree.get_branch_length("0", "2"), 1.0, decimal=3)
        np.testing.assert_almost_equal(tree.get_branch_length("1", "3"), 2.0, decimal=3)
        np.testing.assert_almost_equal(tree.get_branch_length("1", "4"), 2.0, decimal=3)
        np.testing.assert_almost_equal(tree.get_branch_length("2", "5"), 1.5, decimal=3)
        np.testing.assert_almost_equal(tree.get_branch_length("2", "6"), 1.5, decimal=3)

    def test_basic_missing_as_cut(self):
        tree = nx.DiGraph()
        tree.add_nodes_from(["0", "1", "2", "3", "4", "5", "6"]),
        tree.add_edges_from([("0", "1"), ("0", "2"), ("1", "3"), ("1", "4"),
                            ("2", "5"), ("2", "6")])
        tree = CassiopeiaTree(tree=tree)
        tree.set_all_character_states(
            {"0": [0, 0, 0],
            "1": [0, 0, -1],
            "2": [1, 0, 0],
            "3": [0, 1, -1],
            "4": [1, 1, -1],
            "5": [1, -1, -1],
            "6": [1, -1, 0]}
        )
        model = NumberOfMutationsBLE(length_of_mutationless_edges=0.5, treat_missing_states_as_mutations=True)
        model.estimate_branch_lengths(tree)
        np.testing.assert_almost_equal(tree.get_branch_length("0", "1"), 1.0, decimal=3)
        np.testing.assert_almost_equal(tree.get_branch_length("0", "2"), 1.0, decimal=3)
        np.testing.assert_almost_equal(tree.get_branch_length("1", "3"), 2.0, decimal=3)
        np.testing.assert_almost_equal(tree.get_branch_length("1", "4"), 2.0, decimal=3)
        np.testing.assert_almost_equal(tree.get_branch_length("2", "5"), 2.0, decimal=3)
        np.testing.assert_almost_equal(tree.get_branch_length("2", "6"), 2.0, decimal=3)

    def test_basic_depth_3_missing_ignored(self):
        tree = nx.DiGraph()
        tree.add_nodes_from(["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14"]),
        tree.add_edges_from([("0", "1"), ("0", "2"), ("1", "3"), ("1", "4"),
                            ("2", "5"), ("2", "6"), ("3", "7"), ("3", "8"),
                            ("4", "9"), ("4", "10"), ("5", "11"), ("5", "12"),
                            ("6", "13"), ("6", "14")])
        tree = CassiopeiaTree(tree=tree)
        tree.set_all_character_states(
            {"0": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "1": [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "2": [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "3": [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            "4": [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            "5": [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            "6": [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            "7": [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            "8": [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            "9": [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
            "10": [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            "11": [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
            "12": [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
            "13": [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
            "14": [1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 0, 0],},
        )
        model = NumberOfMutationsBLE(length_of_mutationless_edges=0.5, treat_missing_states_as_mutations=False)
        model.estimate_branch_lengths(tree)
        np.testing.assert_almost_equal(tree.get_branch_length("0", "1"), 3.0, decimal=3)
        np.testing.assert_almost_equal(tree.get_branch_length("0", "2"), 2.0, decimal=3)
        np.testing.assert_almost_equal(tree.get_branch_length("1", "3"), 1.0, decimal=3)
        np.testing.assert_almost_equal(tree.get_branch_length("1", "4"), 2.0, decimal=3)
        np.testing.assert_almost_equal(tree.get_branch_length("2", "5"), 3.0, decimal=3)
        np.testing.assert_almost_equal(tree.get_branch_length("2", "6"), 4.0, decimal=3)
        np.testing.assert_almost_equal(tree.get_branch_length("3", "7"), 5.0, decimal=3)
        np.testing.assert_almost_equal(tree.get_branch_length("3", "8"), 5.0, decimal=3)
        np.testing.assert_almost_equal(tree.get_branch_length("4", "9"), 4.0, decimal=3)
        np.testing.assert_almost_equal(tree.get_branch_length("4", "10"), 4.0, decimal=3)
        np.testing.assert_almost_equal(tree.get_branch_length("5", "11"), 4.0, decimal=3)
        np.testing.assert_almost_equal(tree.get_branch_length("5", "12"), 4.0, decimal=3)
        np.testing.assert_almost_equal(tree.get_branch_length("6", "13"), 3.0, decimal=3)
        np.testing.assert_almost_equal(tree.get_branch_length("6", "14"), 3.0, decimal=3)

    def test_basic_depth_3_missing_as_cuts(self):
        tree = nx.DiGraph()
        tree.add_nodes_from(["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14"]),
        tree.add_edges_from([("0", "1"), ("0", "2"), ("1", "3"), ("1", "4"),
                            ("2", "5"), ("2", "6"), ("3", "7"), ("3", "8"),
                            ("4", "9"), ("4", "10"), ("5", "11"), ("5", "12"),
                            ("6", "13"), ("6", "14")])
        tree = CassiopeiaTree(tree=tree)
        tree.set_all_character_states(
            {"0": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "1": [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "2": [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "3": [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            "4": [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            "5": [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            "6": [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            "7": [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            "8": [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            "9": [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
            "10": [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            "11": [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
            "12": [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
            "13": [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
            "14": [1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 0, 0],},
        )
        model = NumberOfMutationsBLE(length_of_mutationless_edges=0.5, treat_missing_states_as_mutations=True)
        model.estimate_branch_lengths(tree)
        np.testing.assert_almost_equal(tree.get_branch_length("0", "1"), 3.0, decimal=3)
        np.testing.assert_almost_equal(tree.get_branch_length("0", "2"), 2.0, decimal=3)
        np.testing.assert_almost_equal(tree.get_branch_length("1", "3"), 1.0, decimal=3)
        np.testing.assert_almost_equal(tree.get_branch_length("1", "4"), 2.0, decimal=3)
        np.testing.assert_almost_equal(tree.get_branch_length("2", "5"), 3.0, decimal=3)
        np.testing.assert_almost_equal(tree.get_branch_length("2", "6"), 4.0, decimal=3)
        np.testing.assert_almost_equal(tree.get_branch_length("3", "7"), 6.0, decimal=3)
        np.testing.assert_almost_equal(tree.get_branch_length("3", "8"), 6.0, decimal=3)
        np.testing.assert_almost_equal(tree.get_branch_length("4", "9"), 5.0, decimal=3)
        np.testing.assert_almost_equal(tree.get_branch_length("4", "10"), 5.0, decimal=3)
        np.testing.assert_almost_equal(tree.get_branch_length("5", "11"), 5.0, decimal=3)
        np.testing.assert_almost_equal(tree.get_branch_length("5", "12"), 5.0, decimal=3)
        np.testing.assert_almost_equal(tree.get_branch_length("6", "13"), 4.0, decimal=3)
        np.testing.assert_almost_equal(tree.get_branch_length("6", "14"), 4.0, decimal=3)


class TestIIDExponentialBLE(unittest.TestCase):
    def test_no_mutations(self):
        r"""
        Tree topology is just a branch 0->1.
        There is one unmutated character i.e.:
            root [state = '0']
            |
            v
            child [state = '0']
        This is thus the simplest possible example of no mutations, and the MLE
        branch length should be 0
        """
        tree = nx.DiGraph()
        tree.add_node("0"), tree.add_node("1")
        tree.add_edge("0", "1")
        tree = CassiopeiaTree(tree=tree)
        tree.set_all_character_states(
            {"0": [0],
            "1": [0]}
        )
        model = IIDExponentialBLE()
        model.estimate_branch_lengths(tree)
        log_likelihood = model.log_likelihood
        np.testing.assert_almost_equal(tree.get_branch_length("0", "1"), 0.0)
        np.testing.assert_almost_equal(tree.get_time("0"), 0.0)
        np.testing.assert_almost_equal(tree.get_time("1"), 0.0)
        np.testing.assert_almost_equal(log_likelihood, 0.0)
        log_likelihood_2 = IIDExponentialBLE.log_likelihood(tree)
        np.testing.assert_almost_equal(log_likelihood, log_likelihood_2, decimal=3)

    def test_saturation(self):
        r"""
        Tree topology is just a branch 0->1.
        There is one mutated character i.e.:
            root [state = '0']
            |
            v
            child [state = '1']
        This is thus the simplest possible example of saturation, and the MLE
        branch length should be infinity (>15 for all practical purposes)
        """
        tree = nx.DiGraph()
        tree.add_nodes_from(["0", "1"])
        tree.add_edge("0", "1")
        tree = CassiopeiaTree(tree=tree)
        tree.set_all_character_states(
            {"0": [0],
            "1": [1]}
        )
        model = IIDExponentialBLE()
        model.estimate_branch_lengths(tree)
        log_likelihood = model.log_likelihood
        assert tree.get_branch_length("0", "1") > 15.0
        assert tree.get_time("1") > 15.0
        np.testing.assert_almost_equal(tree.get_time("0"), 0.0)
        np.testing.assert_almost_equal(log_likelihood, 0.0, decimal=5)
        log_likelihood_2 = IIDExponentialBLE.log_likelihood(tree)
        np.testing.assert_almost_equal(log_likelihood, log_likelihood_2, decimal=3)

    def test_hand_solvable_problem_1(self):
        r"""
        Tree topology is just a branch 0->1.
        There is one mutated character and one unmutated character, i.e.:
            root [state = '00']
            |
            v
            child [state = '01']
        The solution can be verified by hand. The optimization problem is:
            min_{r * t0} log(exp(-r * t0)) + log(1 - exp(-r * t0))
        The solution is r * t0 = ln(2) ~ 0.693
        """
        tree = nx.DiGraph()
        tree.add_nodes_from(["0", "1"])
        tree.add_edge("0", "1")
        tree = CassiopeiaTree(tree=tree)
        tree.set_all_character_states(
            {"0": [0, 0],
            "1": [0, 1]}
        )
        model = IIDExponentialBLE()
        model.estimate_branch_lengths(tree)
        log_likelihood = model.log_likelihood
        np.testing.assert_almost_equal(
            tree.get_branch_length("0", "1"), np.log(2), decimal=3
        )
        np.testing.assert_almost_equal(tree.get_time("1"), np.log(2), decimal=3)
        np.testing.assert_almost_equal(tree.get_time("0"), 0.0)
        np.testing.assert_almost_equal(log_likelihood, -1.386, decimal=3)
        log_likelihood_2 = IIDExponentialBLE.log_likelihood(tree)
        np.testing.assert_almost_equal(log_likelihood, log_likelihood_2, decimal=3)

    def test_hand_solvable_problem_2(self):
        r"""
        Tree topology is just a branch 0->1.
        There are two mutated characters and one unmutated character, i.e.:
            root [state = '000']
            |
            v
            child [state = '011']
        The solution can be verified by hand. The optimization problem is:
            min_{r * t0} log(exp(-r * t0)) + 2 * log(1 - exp(-r * t0))
        The solution is r * t0 = ln(3) ~ 1.098
        """
        tree = nx.DiGraph()
        tree.add_nodes_from(["0", "1"])
        tree.add_edge("0", "1")
        tree = CassiopeiaTree(tree=tree)
        tree.set_all_character_states(
            {"0": [0, 0, 0],
            "1": [0, 1, 1]}
        )
        model = IIDExponentialBLE()
        model.estimate_branch_lengths(tree)
        log_likelihood = model.log_likelihood
        np.testing.assert_almost_equal(
            tree.get_branch_length("0", "1"), np.log(3), decimal=3
        )
        np.testing.assert_almost_equal(tree.get_time("1"), np.log(3), decimal=3)
        np.testing.assert_almost_equal(tree.get_time("0"), 0.0)
        np.testing.assert_almost_equal(log_likelihood, -1.909, decimal=3)
        log_likelihood_2 = IIDExponentialBLE.log_likelihood(tree)
        np.testing.assert_almost_equal(log_likelihood, log_likelihood_2, decimal=3)

    def test_hand_solvable_problem_3(self):
        r"""
        Tree topology is just a branch 0->1.
        There are two unmutated characters and one mutated character, i.e.:
            root [state = '000']
            |
            v
            child [state = '001']
        The solution can be verified by hand. The optimization problem is:
            min_{r * t0} 2 * log(exp(-r * t0)) + log(1 - exp(-r * t0))
        The solution is r * t0 = ln(1.5) ~ 0.405
        """
        tree = nx.DiGraph()
        tree.add_nodes_from(["0", "1"])
        tree.add_edge("0", "1")
        tree = CassiopeiaTree(tree=tree)
        tree.set_all_character_states(
            {"0": [0, 0, 0],
            "1": [0, 0, 1]}
        )
        model = IIDExponentialBLE()
        model.estimate_branch_lengths(tree)
        log_likelihood = model.log_likelihood
        np.testing.assert_almost_equal(
            tree.get_branch_length("0", "1"), np.log(1.5), decimal=3
        )
        np.testing.assert_almost_equal(tree.get_time("1"), np.log(1.5), decimal=3)
        np.testing.assert_almost_equal(tree.get_time("0"), 0.0)
        np.testing.assert_almost_equal(log_likelihood, -1.909, decimal=3)
        log_likelihood_2 = IIDExponentialBLE.log_likelihood(tree)
        np.testing.assert_almost_equal(log_likelihood, log_likelihood_2, decimal=3)

    def test_small_tree_with_no_mutations(self):
        r"""
        Perfect binary tree with no mutations: Should give edges of length 0
        """
        tree = nx.DiGraph()
        tree.add_nodes_from(["0", "1", "2", "3", "4", "5", "6"])
        tree.add_edges_from([("0", "1"), ("0", "2"), ("1", "3"), ("1", "4"),
                            ("2", "5"), ("2", "6")])
        tree = CassiopeiaTree(tree=tree)
        tree.set_all_character_states(
            {"0": [0, 0, 0, 0],
            "1": [0, 0, 0, 0],
            "2": [0, 0, 0, 0],
            "3": [0, 0, 0, 0],
            "4": [0, 0, 0, 0],
            "5": [0, 0, 0, 0],
            "6": [0, 0, 0, 0]}
        )
        model = IIDExponentialBLE()
        model.estimate_branch_lengths(tree)
        log_likelihood = model.log_likelihood
        for edge in tree.edges:
            np.testing.assert_almost_equal(
                tree.get_branch_length(*edge), 0, decimal=3
            )
        np.testing.assert_almost_equal(log_likelihood, 0.0, decimal=3)
        log_likelihood_2 = IIDExponentialBLE.log_likelihood(tree)
        np.testing.assert_almost_equal(log_likelihood, log_likelihood_2, decimal=3)

    def test_small_tree_with_one_mutation(self):
        r"""
        Perfect binary tree with one mutation at a node 6: Should give very short
        edges 1->3,1->4,0->2 and very long edges 0->1,2->5,2->6.
        The problem can be solved by hand: it trivially reduces to a 1-dimensional
        problem:
            min_{r * t0} 2 * log(exp(-r * t0)) + log(1 - exp(-r * t0))
        The solution is r * t0 = ln(1.5) ~ 0.405
        """
        tree = nx.DiGraph()
        tree.add_nodes_from(["0", "1", "2", "3", "4", "5", "6"]),
        tree.add_edges_from([("0", "1"), ("0", "2"), ("1", "3"), ("1", "4"),
                            ("2", "5"), ("2", "6")])
        tree = CassiopeiaTree(tree=tree)
        tree.set_all_character_states(
            {"0": [0],
            "1": [0],
            "2": [0],
            "3": [0],
            "4": [0],
            "5": [0],
            "6": [1]}
        )
        model = IIDExponentialBLE()
        model.estimate_branch_lengths(tree)
        log_likelihood = model.log_likelihood
        np.testing.assert_almost_equal(tree.get_branch_length("0", "1"), 0.405, decimal=3)
        np.testing.assert_almost_equal(tree.get_branch_length("0", "2"), 0.0, decimal=3)
        np.testing.assert_almost_equal(tree.get_branch_length("1", "3"), 0.0, decimal=3)
        np.testing.assert_almost_equal(tree.get_branch_length("1", "4"), 0.0, decimal=3)
        np.testing.assert_almost_equal(tree.get_branch_length("2", "5"), 0.405, decimal=3)
        np.testing.assert_almost_equal(tree.get_branch_length("2", "6"), 0.405, decimal=3)
        np.testing.assert_almost_equal(log_likelihood, -1.909, decimal=3)
        log_likelihood_2 = IIDExponentialBLE.log_likelihood(tree)
        np.testing.assert_almost_equal(log_likelihood, log_likelihood_2, decimal=3)

    def test_small_tree_with_saturation(self):
        r"""
        Perfect binary tree with saturation. The edges which saturate should thus
        have length infinity (>15 for all practical purposes)
        """
        tree = nx.DiGraph()
        tree.add_nodes_from(["0", "1", "2", "3", "4", "5", "6"]),
        tree.add_edges_from([("0", "1"), ("0", "2"), ("1", "3"), ("1", "4"),
                            ("2", "5"), ("2", "6")])
        tree = CassiopeiaTree(tree=tree)
        tree.set_all_character_states(
            {"0": [0],
            "1": [0],
            "2": [1],
            "3": [1],
            "4": [1],
            "5": [1],
            "6": [1]}
        )
        model = IIDExponentialBLE()
        model.estimate_branch_lengths(tree)
        log_likelihood = model.log_likelihood
        assert tree.get_branch_length("0", "2") > 15.0
        assert tree.get_branch_length("1", "3") > 15.0
        assert tree.get_branch_length("1", "4") > 15.0
        log_likelihood_2 = IIDExponentialBLE.log_likelihood(tree)
        np.testing.assert_almost_equal(log_likelihood, log_likelihood_2, decimal=3)

    def test_small_tree_regression(self):
        r"""
        Regression test. Cannot be solved by hand. We just check that this solution
        never changes.
        """
        # Perfect binary tree with normal amount of mutations on each edge
        tree = nx.DiGraph()
        tree.add_nodes_from(["0", "1", "2", "3", "4", "5", "6"]),
        tree.add_edges_from([("0", "1"), ("0", "2"), ("1", "3"), ("1", "4"),
                            ("2", "5"), ("2", "6")])
        tree = CassiopeiaTree(tree=tree)
        tree.set_all_character_states(
            {"0": [0, 0, 0, 0, 0, 0, 0, 0, 0],
            "1": [1, 0, 0, 0, 0, 0, 0, 0, 0],
            "2": [0, 0, 0, 0, 0, 6, 0, 0, 0],
            "3": [1, 2, 0, 0, 0, 0, 0, 0, 0],
            "4": [1, 0, 3, 0, 0, 0, 0, 0, 0],
            "5": [0, 0, 0, 0, 5, 6, 7, 0, 0],
            "6": [0, 0, 0, 4, 0, 6, 0, 8, 9]}
        )
        model = IIDExponentialBLE()
        model.estimate_branch_lengths(tree)
        log_likelihood = model.log_likelihood
        np.testing.assert_almost_equal(tree.get_branch_length("0", "1"), 0.203, decimal=3)
        np.testing.assert_almost_equal(tree.get_branch_length("0", "2"), 0.082, decimal=3)
        np.testing.assert_almost_equal(tree.get_branch_length("1", "3"), 0.175, decimal=3)
        np.testing.assert_almost_equal(tree.get_branch_length("1", "4"), 0.175, decimal=3)
        np.testing.assert_almost_equal(tree.get_branch_length("2", "5"), 0.295, decimal=3)
        np.testing.assert_almost_equal(tree.get_branch_length("2", "6"), 0.295, decimal=3)
        np.testing.assert_almost_equal(log_likelihood, -22.689, decimal=3)
        log_likelihood_2 = IIDExponentialBLE.log_likelihood(tree)
        np.testing.assert_almost_equal(log_likelihood, log_likelihood_2, decimal=3)

    def test_small_symmetric_tree(self):
        r"""
        Symmetric tree should have equal length edges.
        """
        tree = nx.DiGraph()
        tree.add_nodes_from(["0", "1", "2", "3", "4", "5", "6"]),
        tree.add_edges_from([("0", "1"), ("0", "2"), ("1", "3"), ("1", "4"),
                            ("2", "5"), ("2", "6")])
        tree = CassiopeiaTree(tree=tree)
        tree.set_all_character_states(
            {"0": [0, 0, 0],
            "1": [1, 0, 0],
            "2": [1, 0, 0],
            "3": [1, 1, 0],
            "4": [1, 1, 0],
            "5": [1, 1, 0],
            "6": [1, 1, 0]}
        )
        model = IIDExponentialBLE()
        model.estimate_branch_lengths(tree)
        log_likelihood = model.log_likelihood
        np.testing.assert_almost_equal(
            tree.get_branch_length("0", "1"), tree.get_branch_length("0", "2")
        )
        np.testing.assert_almost_equal(
            tree.get_branch_length("1", "3"), tree.get_branch_length("1", "4")
        )
        np.testing.assert_almost_equal(
            tree.get_branch_length("1", "4"), tree.get_branch_length("2", "5")
        )
        np.testing.assert_almost_equal(
            tree.get_branch_length("2", "5"), tree.get_branch_length("2", "6")
        )
        log_likelihood_2 = IIDExponentialBLE.log_likelihood(tree)
        np.testing.assert_almost_equal(log_likelihood, log_likelihood_2, decimal=3)

    def test_small_tree_with_infinite_legs(self):
        r"""
        Perfect binary tree with saturated leaves. The first level of the tree
        should be normal (can be solved by hand, solution is log(2)),
        the branches for the leaves should be infinity (>15 for all practical
        purposes)
        """
        tree = nx.DiGraph()
        tree.add_nodes_from(["0", "1", "2", "3", "4", "5", "6"]),
        tree.add_edges_from([("0", "1"), ("0", "2"), ("1", "3"), ("1", "4"),
                            ("2", "5"), ("2", "6")])
        tree = CassiopeiaTree(tree=tree)
        tree.set_all_character_states(
            {"0": [0, 0],
            "1": [1, 0],
            "2": [1, 0],
            "3": [1, 1],
            "4": [1, 1],
            "5": [1, 1],
            "6": [1, 1]}
        )
        model = IIDExponentialBLE()
        model.estimate_branch_lengths(tree)
        log_likelihood = model.log_likelihood
        np.testing.assert_almost_equal(tree.get_branch_length("0", "1"), 0.693, decimal=3)
        np.testing.assert_almost_equal(tree.get_branch_length("0", "2"), 0.693, decimal=3)
        assert tree.get_branch_length("1", "3") > 15
        assert tree.get_branch_length("1", "4") > 15
        assert tree.get_branch_length("2", "5") > 15
        assert tree.get_branch_length("2", "6") > 15
        log_likelihood_2 = IIDExponentialBLE.log_likelihood(tree)
        np.testing.assert_almost_equal(log_likelihood, log_likelihood_2, decimal=3)

    def test_on_simulated_data(self):
        tree = nx.DiGraph()
        tree.add_nodes_from(["0", "1", "2", "3", "4", "5", "6"]),
        tree.add_edges_from([("0", "1"), ("0", "2"), ("1", "3"), ("1", "4"),
                            ("2", "5"), ("2", "6")])
        tree = CassiopeiaTree(tree=tree)
        tree.set_times(
            {"0": 0,
            "1": 0.1,
            "2": 0.9,
            "3": 1.0,
            "4": 1.0,
            "5": 1.0,
            "6": 1.0}
        )
        np.random.seed(1)
        IIDExponentialLineageTracer(
            mutation_rate=1.0, num_characters=100
        ).overlay_lineage_tracing_data(tree)
        model = IIDExponentialBLE()
        model.estimate_branch_lengths(tree)
        log_likelihood = model.log_likelihood
        assert 0.05 < tree.get_time("1") < 0.15
        assert 0.8 < tree.get_time("2") < 1.0
        assert 0.9 < tree.get_time("3") < 1.1
        assert 0.9 < tree.get_time("4") < 1.1
        assert 0.9 < tree.get_time("5") < 1.1
        assert 0.9 < tree.get_time("6") < 1.1
        np.testing.assert_almost_equal(tree.get_time("0"), 0)
        log_likelihood_2 = IIDExponentialBLE.log_likelihood(tree)
        np.testing.assert_almost_equal(log_likelihood, log_likelihood_2, decimal=2)

    def test_subtree_collapses_when_no_mutations(self):
        r"""
        A subtree with no mutations should collapse to 0. It reduces the problem to
        the same as in 'test_hand_solvable_problem_1'
        """
        tree = nx.DiGraph()
        tree.add_nodes_from(["0", "1", "2", "3", "4"]),
        tree.add_edges_from([("0", "1"), ("1", "2"), ("1", "3"), ("0", "4")])
        tree = CassiopeiaTree(tree=tree)
        tree.set_all_character_states(
            {"0": [0],
            "1": [1],
            "2": [1],
            "3": [1],
            "4": [0]}
        )
        model = IIDExponentialBLE()
        model.estimate_branch_lengths(tree)
        log_likelihood = model.log_likelihood
        np.testing.assert_almost_equal(
            tree.get_branch_length("0", "1"), np.log(2), decimal=3
        )
        np.testing.assert_almost_equal(tree.get_branch_length("1", "2"), 0.0, decimal=3)
        np.testing.assert_almost_equal(tree.get_branch_length("1", "3"), 0.0, decimal=3)
        np.testing.assert_almost_equal(
            tree.get_branch_length("0", "4"), np.log(2), decimal=3
        )
        np.testing.assert_almost_equal(log_likelihood, -1.386, decimal=3)
        log_likelihood_2 = IIDExponentialBLE.log_likelihood(tree)
        np.testing.assert_almost_equal(log_likelihood, log_likelihood_2, decimal=3)

    def test_minimum_branch_length(self):
        tree = nx.DiGraph()
        tree.add_nodes_from(["0", "1", "2", "3", "4"])
        tree.add_edges_from([("0", "1"), ("0", "2"), ("0", "3"), ("2", "4")])
        tree = CassiopeiaTree(tree=tree)
        tree.set_character_states_at_leaves(
            {"1": [1],
            "3": [1],
            "4": [1]}
        )
        tree.reconstruct_ancestral_characters(zero_the_root=True)
        # Too large a minimum_branch_length
        model = IIDExponentialBLE(minimum_branch_length=0.6)
        model.estimate_branch_lengths(tree)
        for node in tree.nodes:
            print(f"{node} = {tree.get_time(node)}")
        assert model.log_likelihood < -20.0  # Should be really negative (but not inf due to numerically stable computation of LL)
        # An okay minimum_branch_length
        model = IIDExponentialBLE(minimum_branch_length=0.4)
        model.estimate_branch_lengths(tree)
        assert model.log_likelihood != -np.inf


class TestIIDExponentialBLEGridSearchCV(unittest.TestCase):
    def test_IIDExponentialBLEGridSearchCV_smoke(self):
        r"""
        Just want to see that it runs in both single and multiprocessor mode
        """
        tree = nx.DiGraph()
        tree.add_nodes_from(["0", "1"]),
        tree.add_edges_from([("0", "1")])
        tree = CassiopeiaTree(tree=tree)
        tree.set_all_character_states(
            {"0": [0],
            "1": [1]},
        )
        for processes in [1, 2]:
            model = IIDExponentialBLEGridSearchCV(
                minimum_branch_lengths=(1.0,),
                l2_regularizations=(1.0,),
                verbose=True,
                processes=processes,
            )
            model.estimate_branch_lengths(tree)

    def test_IIDExponentialBLEGridSearchCV(self):
        r"""
        We make sure to test a tree for which no regularization produces
        a likelihood of 0.
        """
        tree = nx.DiGraph()
        tree.add_nodes_from(["0", "1", "2", "3", "4", "5", "6", "7"]),
        tree.add_edges_from(
            [("0", "1"), ("1", "2"), ("1", "3"), ("2", "4"), ("2", "5"), ("3", "6"),
            ("3", "7")]
        )
        tree = CassiopeiaTree(tree=tree)
        tree.set_character_states_at_leaves(
            {"4": [1, 1, 0],
            "5": [1, 1, 0],
            "6": [1, 0, 0],
            "7": [1, 0, 0]},
        )
        tree.reconstruct_ancestral_characters(zero_the_root=True)
        model = IIDExponentialBLEGridSearchCV(
            minimum_branch_lengths=(0, 0.2, 4.0),
            l2_regularizations=(0.0, 2.0, 4.0),
            verbose=True,
            processes=6,
        )
        model.estimate_branch_lengths(tree)
        print(model.grid)
        assert model.grid[0, 0] == -np.inf

        # import seaborn as sns
        # import matplotlib.pyplot as plt
        # sns.heatmap(
        #     model.grid,
        #     yticklabels=model.minimum_branch_lengths,
        #     xticklabels=model.l2_regularizations,
        #     mask=np.isneginf(model.grid),
        # )
        # plt.ylabel("minimum_branch_length")
        # plt.xlabel("l2_regularization")
        # plt.show()

        np.testing.assert_almost_equal(model.minimum_branch_length, 0.2)
        np.testing.assert_almost_equal(model.l2_regularization, 2.0)


class TestIIDExponentialPosteriorMeanBLEGridSeachCV(unittest.TestCase):
    def test_IIDExponentialPosteriorMeanBLEGridSeachCV_smoke(self):
        r"""
        Just want to see that it runs in both single and multiprocessor mode
        """
        tree = nx.DiGraph()
        tree.add_nodes_from(["0", "1"]),
        tree.add_edges_from([("0", "1")])
        tree = CassiopeiaTree(tree=tree)
        tree.set_all_character_states(
            {"0": [0],
            "1": [1]}
        )
        for processes in [1, 2]:
            model = IIDExponentialPosteriorMeanBLEGridSearchCV(
                mutation_rates=(0.5,),
                birth_rates=(1.5,),
                discretization_level=5,
                verbose=True,
            )
            model.estimate_branch_lengths(tree)


class TestBLEMultifurcationWrapper(unittest.TestCase):
    def test_smoke(self):
        tree = nx.DiGraph()
        tree.add_nodes_from(["0", "1", "2", "3"])
        tree.add_edges_from([("0", "1"), ("0", "2"), ("0", "3")])
        tree = CassiopeiaTree(tree=tree)
        tree.set_all_character_states(
            {"0": [0, 0],
             "1": [0, 1],
             "2": [0, 1],
             "3": [0, 1]}
        )
        model = BLEMultifurcationWrapper(IIDExponentialBLE())
        model.estimate_branch_lengths(tree)
        log_likelihood = model.log_likelihood
        np.testing.assert_almost_equal(
            tree.get_branch_length("0", "1"), np.log(2), decimal=3
        )
        np.testing.assert_almost_equal(
            tree.get_branch_length("0", "2"), np.log(2), decimal=3
        )
        np.testing.assert_almost_equal(
            tree.get_branch_length("0", "3"), np.log(2), decimal=3
        )
        np.testing.assert_almost_equal(tree.get_time("1"), np.log(2), decimal=3)
        np.testing.assert_almost_equal(tree.get_time("2"), np.log(2), decimal=3)
        np.testing.assert_almost_equal(tree.get_time("3"), np.log(2), decimal=3)
        np.testing.assert_almost_equal(tree.get_time("0"), 0.0)
        np.testing.assert_almost_equal(log_likelihood, -1.386, decimal=3)
        log_likelihood_2 = IIDExponentialBLE.log_likelihood(tree)
        # The tree topology that the estimator sees is different from the
        # one in the final phylogeny, thus the lik will be different!
        np.testing.assert_almost_equal(log_likelihood * 3, log_likelihood_2, decimal=3)
