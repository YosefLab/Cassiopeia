import unittest
from cassiopeia.simulator import SimpleFitSubcloneSimulator


class TestSimpleFitSubcloneSimulator(unittest.TestCase):
    def test_SimpleFitSubcloneSimulator(self):
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
