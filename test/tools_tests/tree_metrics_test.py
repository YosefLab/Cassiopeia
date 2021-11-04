"""
Tests for cassiopeia/tools/tree_metrics.py
"""
import unittest
from typing import Dict, Optional

import itertools
import networkx as nx
import numpy as np
import pandas as pd

import cassiopeia as cas
import cassiopeia.tools.tree_metrics as tm
from cassiopeia.data.CassiopeiaTree import (
    CassiopeiaTreeError,
    CassiopeiaTreeWarning,
)

class TestCassiopeiaTree(unittest.TestCase):
    def setUp(self):
        small_net = nx.DiGraph()
        small_net.add_edges_from(
            [
                ("node5", "node0"),
                ("node5", "node1"),
                ("node6", "node2"),
                ("node6", "node3"),
                ("node6", "node4"),
                ("node7", "node5"),
                ("node7", "node6"),
            ]
        )
        self.small_net = small_net


    def test_parsimony_bad_cases(self):
        small_tree = cas.data.CassiopeiaTree(tree=self.small_net)
        with self.assertRaises(CassiopeiaTreeError):
            tm.calculate_parsimony(small_tree, infer_ancestral_characters=False)

        small_cm = pd.DataFrame.from_dict(
            {
                "node0": [1, -1, -1],
                "node1": [2, 1, -1],
                "node2": [2, -1, -1],
                "node3": [1, 2, 2],
                "node4": [1, 1, 2],
            },
            orient="index",
        )
        small_tree = cas.data.CassiopeiaTree(
            tree=self.small_net, character_matrix=small_cm
        )
        with self.assertRaises(CassiopeiaTreeError):
            tm.calculate_parsimony(small_tree, infer_ancestral_characters=False)

        small_tree.set_character_states("node7", [0, 0, 0])
        small_tree.set_character_states("node5", [0, 0, 0])
        with self.assertRaises(CassiopeiaTreeError):
            tm.calculate_parsimony(small_tree, infer_ancestral_characters=False)

    def test_parsimony_reconstruct_internal_states(self):
        small_cm = pd.DataFrame.from_dict(
            {
                "node0": [1, -1, -1],
                "node1": [2, 1, -1],
                "node2": [2, -1, -1],
                "node3": [1, 2, 2],
                "node4": [1, 1, 2],
            },
            orient="index",
        )
        small_tree = cas.data.CassiopeiaTree(
            tree=self.small_net, character_matrix=small_cm
        )
        p = tm.calculate_parsimony(small_tree, infer_ancestral_characters=True)
        self.assertEqual(p, 8)
        p = tm.calculate_parsimony(small_tree, 
            infer_ancestral_characters=True, treat_missing_as_mutation=True
        )
        self.assertEqual(p, 12)

    def test_parsimony_specify_internal_states(self):
        small_cm = pd.DataFrame.from_dict(
            {
                "node0": [1, -1, -1],
                "node1": [2, 1, -1],
                "node2": [2, -1, -1],
                "node3": [1, 2, 2],
                "node4": [1, 1, 2],
            },
            orient="index",
        )
        small_tree = cas.data.CassiopeiaTree(
            tree=self.small_net, character_matrix=small_cm
        )
        small_tree.set_character_states("node7", [0, 0, 0])
        small_tree.set_character_states("node5", [0, 0, 0])
        small_tree.set_character_states("node6", [0, 0, 2])

        p = tm.calculate_parsimony(small_tree, infer_ancestral_characters=False)
        self.assertEqual(p, 9)
        p = tm.calculate_parsimony(small_tree, 
            infer_ancestral_characters=False, treat_missing_as_mutation=True
        )
        self.assertEqual(p, 14)

    def test_likelihood_bad_cases(self):
        small_tree = cas.data.CassiopeiaTree(tree=self.small_net)
        with self.assertRaises(CassiopeiaTreeError):
            tm.calculate_likelihood(small_tree)

        small_cm = pd.DataFrame.from_dict(
            {
                "node0": [1, -1, -1],
                "node1": [2, 1, -1],
                "node2": [2, -1, -1],
                "node3": [1, 2, 2],
                "node4": [1, 1, 2],
            },
            orient="index",
        )
        small_tree = cas.data.CassiopeiaTree(
            tree=self.small_net, character_matrix=small_cm
        )
        with self.assertRaises(CassiopeiaTreeError):
            tm.calculate_likelihood(small_tree)

        priors = {
            0: {1: 0.3, 2: 0.7},
            1: {1: 0.3, 2: 0.7},
            2: {1: 0.3, 2: 0.7},
            3: {1: 0.3, 2: 0.7},
        }
        small_tree = cas.data.CassiopeiaTree(
            tree=self.small_net, character_matrix=small_cm, priors=priors
        )

        with self.assertRaises(CassiopeiaTreeError):
            tm.calculate_likelihood(small_tree,
                use_branch_lengths=False,
                use_internal_character_states=True
            )

        small_tree.set_character_states("node7", [0, 0, 0])
        small_tree.set_character_states("node5", [0, 0, 0])

        with self.assertRaises(CassiopeiaTreeError):
            tm.calculate_likelihood(small_tree,
                use_branch_lengths=False,
                use_internal_character_states=True,
            )

        small_tree.set_character_states("node6", [0, 0, 1])
        L = tm.calculate_likelihood(small_tree,
            use_branch_lengths=False,
            use_internal_character_states=True,
        )
        self.assertEqual(-np.inf, L)

        with self.assertRaises(CassiopeiaTreeError):
            small_tree.parameters["mutation_rate"] = -1
            tm.calculate_likelihood(small_tree)
        with self.assertRaises(CassiopeiaTreeError):
            small_tree.parameters["mutation_rate"] = -1
            tm.calculate_likelihood(small_tree, use_branch_lengths=False)
        with self.assertRaises(CassiopeiaTreeError):
            small_tree.parameters["heritable_missing_rate"] = -1
            tm.calculate_likelihood(small_tree)
        with self.assertRaises(CassiopeiaTreeError):
            small_tree.parameters["heritable_missing_rate"] = 1.5
            tm.calculate_likelihood(small_tree, use_branch_lengths=False)
        with self.assertRaises(CassiopeiaTreeError):
            small_tree.parameters["stochastic_missing_probability"] = -1
            tm.calculate_likelihood(small_tree)
        with self.assertRaises(CassiopeiaTreeError):
            small_tree.parameters["stochastic_missing_probability"] = 1.5
            tm.calculate_likelihood(small_tree)
        with self.assertRaises(CassiopeiaTreeError):
            tm.calculate_likelihood(small_tree,
                proportion_of_missing_as_stochastic=-1
            )
        with self.assertRaises(CassiopeiaTreeError):
            tm.calculate_likelihood(small_tree,
                proportion_of_missing_as_stochastic=1.5
            )

        small_tree.parameters = {}
        no_missing_cm = pd.DataFrame.from_dict(
            {
                "node0": [1, 1, 1],
                "node1": [2, 1, 1],
                "node2": [2, 1, 1],
                "node3": [1, 2, 2],
                "node4": [1, 1, 2],
            },
            orient="index",
        )
        small_tree = cas.data.CassiopeiaTree(
            tree=self.small_net, character_matrix=no_missing_cm, priors=priors
        )
        with self.assertRaises(CassiopeiaTreeWarning):
            small_tree.parameters["heritable_missing_rate"] = 0.5
            tm.calculate_likelihood(small_tree)
        with self.assertRaises(CassiopeiaTreeWarning):
            small_tree.parameters.pop("heritable_missing_rate")
            small_tree.parameters["stochastic_missing_probability"] = 0.5
            tm.calculate_likelihood(small_tree)

    def test_likelihood_simple_mostly_missing(self):
        small_cm = pd.DataFrame.from_dict(
            {
                "node0": [0, -1, -1],
                "node1": [1, 1, -1],
                "node2": [1, -1, -1],
                "node3": [1, -1, -1],
                "node4": [1, -1, -1],
            },
            orient="index",
        )
        priors = {0: {1: 1}, 1: {1: 1}, 2: {1: 1}}
        small_tree = cas.data.CassiopeiaTree(
            tree=self.small_net, character_matrix=small_cm, priors=priors
        )
        L = tm.calculate_likelihood(small_tree,
            use_branch_lengths=False,
        )
        self.assertTrue(np.isclose(L, -11.458928604116634))

        small_tree.parameters["mutation_rate"] = 0.5
        L = tm.calculate_likelihood(small_tree,
            use_branch_lengths=False,
            proportion_of_missing_as_stochastic = 0.2,
        )
        self.assertTrue(np.isclose(L, -10.855197443145142))

        small_tree.parameters["stochastic_missing_probability"] = 0.2
        L = tm.calculate_likelihood(small_tree,
            use_branch_lengths=False,
        )
        self.assertTrue(np.isclose(L, -11.09716890609409))

        small_tree.parameters.pop("stochastic_missing_probability")
        small_tree.parameters["heritable_missing_rate"] = 0.25
        L = tm.calculate_likelihood(small_tree,
            use_branch_lengths=False,
        )
        self.assertTrue(np.isclose(L, -10.685658651089808))

        small_tree.parameters["stochastic_missing_probability"] = 0
        L = tm.calculate_likelihood(small_tree,
            use_branch_lengths=False,
        )
        self.assertTrue(np.isclose(L, -10.549534744691526))


    def test_likelihood_more_complex_case(self):
        small_cm = pd.DataFrame.from_dict(
            {
                "node0": [1, -1, -1, 1],
                "node1": [2, 1, -1, 1],
                "node2": [2, -1, -1, -1],
                "node3": [1, 2, 2, -1],
                "node4": [1, 1, 2, 1],
            },
            orient="index",
        )
        priors = {
            0: {1: 0.3, 2: 0.7},
            1: {1: 0.3, 2: 0.7},
            2: {1: 0.3, 2: 0.7},
            3: {1: 0.3, 2: 0.7},
        }
        small_tree = cas.data.CassiopeiaTree(
            tree=self.small_net, character_matrix=small_cm, priors=priors
        )

        small_tree.parameters["mutation_rate"] = 0.5
        small_tree.parameters["heritable_missing_rate"] = 0.25
        small_tree.parameters["stochastic_missing_probability"] = 0
        L = tm.calculate_likelihood(small_tree,
            use_branch_lengths=False,
        )
        self.assertTrue(np.isclose(L, -33.11623901010781))

    def test_likelihood_set_internal_states(self):
        small_cm = pd.DataFrame.from_dict(
            {
                "node0": [1, -1, -1],
                "node1": [2, 1, -1],
                "node2": [2, -1, -1],
                "node3": [1, 2, 2],
                "node4": [1, 1, 2],
            },
            orient="index",
        )
        priors = {
            0: {1: 0.3, 2: 0.7},
            1: {1: 0.3, 2: 0.7},
            2: {1: 0.3, 2: 0.7},
            3: {1: 0.3, 2: 0.7},
        }
        small_tree = cas.data.CassiopeiaTree(
            tree=self.small_net, character_matrix=small_cm, priors=priors
        )
        small_tree.parameters["mutation_rate"] = 0.5
        small_tree.parameters["heritable_missing_rate"] = 0.25
        small_tree.parameters["stochastic_missing_probability"] = 0
        small_tree.reconstruct_ancestral_characters()
        L = tm.calculate_likelihood(small_tree,
            use_branch_lengths=False,
            use_internal_character_states=True,
        )
        self.assertTrue(np.isclose(L, -24.57491637086155))

        small_tree.set_character_states("node7", [0, 0, 0])
        small_tree.set_character_states("node5", [0, 0, 0])
        small_tree.set_character_states("node6", [0, 0, 2])
        L = tm.calculate_likelihood(small_tree,
            use_branch_lengths=False,
            use_internal_character_states=True,
        )
        self.assertTrue(np.isclose(L, -28.68500929005179))

    def test_likelihood_time(self):
        small_cm = pd.DataFrame.from_dict(
            {
                "node0": [1, 0],
                "node1": [1, 1],
                "node2": [2, 3],
                "node3": [-1, 2],
                "node4": [-1, 1],
            },
            orient="index",
        )
        priors = {
            0: {1: 0.2, 2: 0.7, 3: 0.1},
            1: {1: 0.2, 2: 0.7, 3: 0.1},
            2: {1: 0.2, 2: 0.7, 3: 0.1},
        }
        small_tree = cas.data.CassiopeiaTree(
            tree=self.small_net, character_matrix=small_cm, priors=priors
        )
        small_tree.set_branch_length("node5", "node0", 1.5)
        small_tree.set_branch_length("node6", "node3", 2)

        L = tm.calculate_likelihood(small_tree)
        self.assertTrue(np.isclose(L, -20.5238276768878))

        small_tree.parameters["mutation_rate"] = 0.5
        L = tm.calculate_likelihood(small_tree,
            proportion_of_missing_as_stochastic=0.2
        )
        self.assertTrue(np.isclose(L, -21.003429330744467))

        small_tree.parameters["stochastic_missing_probability"] = 0.1
        L = tm.calculate_likelihood(small_tree)
        self.assertTrue(np.isclose(L, -20.67410206503938))

        small_tree.parameters.pop("stochastic_missing_probability")
        small_tree.parameters["heritable_missing_rate"] = 0.05
        L = tm.calculate_likelihood(small_tree)
        self.assertTrue(np.isclose(L, -20.959879404598198))

        small_tree.parameters["heritable_missing_rate"] = 0.25
        small_tree.parameters["stochastic_missing_probability"] = 0
        L = tm.calculate_likelihood(small_tree)
        self.assertTrue(np.isclose(L, -21.943439525312456))

        small_tree.parameters["stochastic_missing_probability"] = 0.2
        L = tm.calculate_likelihood(small_tree)
        self.assertTrue(np.isclose(L, -22.926786566275887))

    def test_likelihood_sum_to_one(self):
        priors = {0: {1: 0.2, 2: 0.8}, 1: {1: 0.2, 2: 0.8}, 2: {1: 0.2, 2: 0.8}}
        ls_branch = []
        ls_no_branch = []
        for (
            a,
            b,
        ) in itertools.product([0, 1, -1, 2], repeat=2):
            for a_, b_ in itertools.product([0, 1, -1, 2], repeat=2):
                small_net = nx.DiGraph()
                small_net.add_edges_from(
                    [("node2", "node0"), ("node2", "node1"), ("node3", "node2")]
                )
                small_cm = pd.DataFrame.from_dict(
                    {
                        "node0": [a, a_],
                        "node1": [b, b_],
                    },
                    orient="index",
                )
                small_tree = cas.data.CassiopeiaTree(
                    tree=small_net, character_matrix=small_cm, priors=priors
                )
                small_tree.parameters["mutation_rate"] = 0.5
                small_tree.parameters["heritable_missing_rate"] = 0.25
                small_tree.parameters["stochastic_missing_probability"] = 0.25
                L_no_branch = tm.calculate_likelihood(small_tree,
                    use_branch_lengths=False,
                    use_internal_character_states=False,
                )
                L_branch = tm.calculate_likelihood(small_tree,
                    use_branch_lengths=True,
                    use_internal_character_states=False,
                )
                ls_no_branch.append(np.exp(L_no_branch))
                ls_branch.append(np.exp(L_branch))
        self.assertTrue(np.isclose(sum(ls_branch), 1.0))
        self.assertTrue(np.isclose(sum(ls_no_branch), 1.0))

if __name__ == "__main__":
    unittest.main()