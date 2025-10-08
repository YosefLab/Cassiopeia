import unittest

import numpy as np

from cassiopeia.simulator import SimpleFitSubcloneSimulator


class TestSimpleFitSubcloneSimulator(unittest.TestCase):
    def test_deterministic(self):
        r"""
        Small test that can be drawn by hand.
        Checks that the generated phylogeny is correct.
        """
        tree = SimpleFitSubcloneSimulator(
            branch_length_neutral=1,
            branch_length_fit=0.5,
            experiment_duration=1.9,
            generations_until_fit_subclone=1,
        ).simulate_tree()
        self.assertListEqual(
            tree.nodes,
            ["0_neutral", "1_neutral", "2_fit", "3_neutral", "4_fit", "5_fit"],
        )
        self.assertListEqual(
            tree.edges,
            [
                ("0_neutral", "1_neutral"),
                ("1_neutral", "2_fit"),
                ("1_neutral", "3_neutral"),
                ("2_fit", "4_fit"),
                ("2_fit", "5_fit"),
            ],
        )
        self.assertDictEqual(
            tree.get_times(),
            {
                "0_neutral": 0.0,
                "1_neutral": 1.0,
                "2_fit": 1.5,
                "3_neutral": 1.9,
                "4_fit": 1.9,
                "5_fit": 1.9,
            },
        )

    def test_stochastic(self):
        r"""
        We test the functionality that allows providing a callable for branch
        lengths. Because the test is stochastic, we don't assert anything
        besides the branch lengths being all different.
        """
        np.random.seed(1)

        def branch_length_neutral() -> float:
            return np.random.exponential(1.0)

        def branch_length_fit() -> float:
            return np.random.exponential(0.5)

        tree = SimpleFitSubcloneSimulator(
            branch_length_neutral=branch_length_neutral,
            branch_length_fit=branch_length_fit,
            experiment_duration=4.9,
            generations_until_fit_subclone=2,
        ).simulate_tree()
        # Just check that all branch lengths are distinct to confirm
        # non-determinism. We exclude the leaves because sister leaves have the
        # same branch length.
        branch_lengths = [tree.get_branch_length(p, c) for (p, c) in tree.edges if not tree.is_leaf(c)]
        assert len(branch_lengths) == len(set(branch_lengths))
