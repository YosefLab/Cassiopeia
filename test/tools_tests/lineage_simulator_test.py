import pytest
import unittest

import numpy as np

from cassiopeia.tools import (
    PerfectBinaryTree,
    PerfectBinaryTreeWithRootBranch,
    BirthProcess,
    TumorWithAFitSubclone,
)


class TestPerfectBinaryTree(unittest.TestCase):
    def test_PerfectBinaryTree(self):
        tree = PerfectBinaryTree(
            generation_branch_lengths=[2, 3]
        ).simulate_lineage()
        newick = tree.get_newick()
        assert newick == "((3,4),(5,6));"
        self.assertDictEqual(
            tree.get_times(), {'0': 0, '1': 2, '2': 2, '3': 5, '4': 5, '5': 5, '6': 5}
        )


class TestPerfectBinaryTreeWithRootBranch(unittest.TestCase):
    def test_PerfectBinaryTreeWithRootBranch(self):
        tree = PerfectBinaryTreeWithRootBranch(
            generation_branch_lengths=[2, 3, 4]
        ).simulate_lineage()
        newick = tree.get_newick()
        assert newick == "(((4,5),(6,7)));"
        self.assertDictEqual(
            tree.get_times(), {'0': 0, '1': 2, '2': 5, '3': 5, '4': 9, '5': 9, '6': 9, '7': 9}
        )


class TestBirthProcess(unittest.TestCase):
    @pytest.mark.slow
    def test_BirthProcess(self):
        r"""
        Generate tree, then choose a random lineage can count how many nodes
        are on the lineage. This is the number of times the process triggered
        on that lineage.

        Also, the probability that a tree with only one internal node is
        obtained is e^-lam * (1 - e^-lam) where lam is the birth rate, so we
        also check this.
        """
        np.random.seed(1)
        birth_rate = 0.6
        intensities = []
        repetitions = 10000
        topology_hits = 0

        def num_ancestors(tree, node: int) -> int:
            r"""
            Number of ancestors of a node. Terribly inefficient implementation.
            """
            res = 0
            root = tree.root
            while node != root:
                node = tree.parent(node)
                res += 1
            return res

        for _ in range(repetitions):
            tree_true = BirthProcess(
                birth_rate=birth_rate, tree_depth=1.0
            ).simulate_lineage()
            if len(tree_true.nodes) == 4:
                topology_hits += 1
            leaf = np.random.choice(tree_true.leaves)
            n_leaves = len(tree_true.leaves)
            n_hits = num_ancestors(tree_true, leaf) - 1
            intensity = n_leaves / 2 ** n_hits * n_hits
            intensities.append(intensity)
        # Check that the probability of the topology matches
        empirical_topology_prob = topology_hits / repetitions
        theoretical_topology_prob = np.exp(-birth_rate) * (
            1.0 - np.exp(-birth_rate)
        )
        assert (
            np.abs(empirical_topology_prob - theoretical_topology_prob) < 0.02
        )
        inferred_birth_rate = np.array(intensities).mean()
        print(f"{birth_rate} == {inferred_birth_rate}")
        assert np.abs(birth_rate - inferred_birth_rate) < 0.05


class TestTumorWithAFitSubclone(unittest.TestCase):
    def test_TumorWithAFitSubclone(self):
        r"""
        Small test that can be drawn by hand.
        Checks that the generated phylogeny is correct.
        """
        tree = TumorWithAFitSubclone(
            branch_length=1,
            branch_length_fit=0.5,
            experiment_duration=2,
            generations_until_fit_subclone=1,
        ).simulate_lineage()
        self.assertListEqual(
            tree.nodes,
            ["0_unfit", "1_unfit", "2_fit", "3_unfit", "4_fit", "5_fit"],
        )
        self.assertListEqual(
            tree.edges,
            [
                ("0_unfit", "1_unfit"),
                ("1_unfit", "2_fit"),
                ("1_unfit", "3_unfit"),
                ("2_fit", "4_fit"),
                ("2_fit", "5_fit"),
            ],
        )
        self.assertDictEqual(
            tree.get_times(),
            {
                "0_unfit": 0.0,
                "1_unfit": 1.0,
                "2_fit": 1.5,
                "3_unfit": 2.0,
                "4_fit": 2.0,
                "5_fit": 2.0,
            },
        )
