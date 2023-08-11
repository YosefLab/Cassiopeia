import unittest

import pandas as pd

from cassiopeia.solver.distance_correction import (
    crispr_cas9_default_collision_probability_estimator,
)


class Test_crispr_cas9_default_collision_probability_estimator(
    unittest.TestCase
):
    """
    Tests for crispr_cas9_default_collision_probability_estimator.
    """
    def test_1(self):
        """
        Compares result against manual computation.
        """
        cm = pd.DataFrame(
            [
                [-1, 1, 0, 2, 2],
                [1, 0, -1, 100000000, 100000000],
                [-1, -1, 1, 3, 100000000],
            ]
        )
        collision_probability = (
            crispr_cas9_default_collision_probability_estimator(cm)
        )
        expected_collision_probability = (
            (3 / 9) ** 2 + (2 / 9) ** 2 + (1 / 9) ** 2 + (3 / 9) ** 2
        )
        assert (
            abs(collision_probability - expected_collision_probability) < 1e-8
        )

    def test_2(self):
        """
        Border case.
        """
        cm = pd.DataFrame(
            [
                [1000000000],
            ]
        )
        collision_probability = (
            crispr_cas9_default_collision_probability_estimator(cm)
        )
        expected_collision_probability = 1.0
        assert (
            abs(collision_probability - expected_collision_probability) < 1e-8
        )

    def test_3(self):
        """
        Border case.
        """
        cm = pd.DataFrame(
            [
                [0],
            ]
        )
        collision_probability = (
            crispr_cas9_default_collision_probability_estimator(cm)
        )
        expected_collision_probability = 0.0
        assert (
            abs(collision_probability - expected_collision_probability) < 1e-8
        )

    def test_4(self):
        """
        Border case.
        """
        cm = pd.DataFrame(
            [
                [-1],
            ]
        )
        collision_probability = (
            crispr_cas9_default_collision_probability_estimator(cm)
        )
        expected_collision_probability = 0.0
        assert (
            abs(collision_probability - expected_collision_probability) < 1e-8
        )
