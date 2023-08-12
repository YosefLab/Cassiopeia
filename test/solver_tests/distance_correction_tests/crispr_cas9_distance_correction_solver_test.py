import unittest

import networkx as nx
import numpy as np
import pandas as pd
import pytest
from parameterized import parameterized

from cassiopeia.data import CassiopeiaTree
from cassiopeia.simulator.Cas9LineageTracingDataSimulator import (
    Cas9LineageTracingDataSimulator,
)
from cassiopeia.solver.distance_correction._crispr_cas9_distance_correction_solver import (
    CRISPRCas9DistanceCorrectionSolver,
    crispr_cas9_corrected_hamming_distance,
    crispr_cas9_corrected_ternary_hamming_distance,
    crispr_cas9_default_collision_probability_estimator,
    crispr_cas9_default_mutation_proportion_estimator,
    crispr_cas9_expected_hamming_distance,
    crispr_cas9_expected_ternary_hamming_distance,
    hamming_distance,
    inverse,
    ternary_hamming_distance,
)


class Test_crispr_cas9_default_mutation_proportion_estimator(unittest.TestCase):
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
        mutation_proportion = crispr_cas9_default_mutation_proportion_estimator(
            cm
        )
        expected_mutation_proportion = 9 / 11
        self.assertAlmostEqual(
            mutation_proportion, expected_mutation_proportion
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
        mutation_proportion = crispr_cas9_default_mutation_proportion_estimator(
            cm
        )
        expected_mutation_proportion = 1 / 1
        self.assertAlmostEqual(
            mutation_proportion, expected_mutation_proportion
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
        mutation_proportion = crispr_cas9_default_mutation_proportion_estimator(
            cm
        )
        expected_mutation_proportion = 0 / 1
        self.assertAlmostEqual(
            mutation_proportion, expected_mutation_proportion
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
        mutation_proportion = crispr_cas9_default_mutation_proportion_estimator(
            cm
        )
        expected_mutation_proportion = 0
        self.assertAlmostEqual(
            mutation_proportion, expected_mutation_proportion
        )


class Test_crispr_cas9_default_collision_probability_estimator(
    unittest.TestCase
):
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
        self.assertAlmostEqual(
            collision_probability, expected_collision_probability
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
        self.assertAlmostEqual(
            collision_probability, expected_collision_probability
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
        self.assertAlmostEqual(
            collision_probability, expected_collision_probability
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
        self.assertAlmostEqual(
            collision_probability, expected_collision_probability
        )


class Test_inverse(unittest.TestCase):
    def test_within_interval(self):
        def f(x):
            return x**2

        res = inverse(
            f=f,
            y=4,
            lower=0,
            upper=10,
        )
        self.assertAlmostEqual(res, 2)

    def test_below_interval(self):
        def f(x):
            return x**2

        res = inverse(
            f=f,
            y=4,
            lower=2.5,
            upper=10,
        )
        self.assertAlmostEqual(res, 2.5)

    def test_above_interval(self):
        def f(x):
            return x**2

        res = inverse(
            f=f,
            y=4,
            lower=0,
            upper=1.5,
        )
        self.assertAlmostEqual(res, 1.5)


class Test_hamming_distance(unittest.TestCase):
    def test_1(self):
        res = hamming_distance(
            s1=[-1, 0, 1, 2, 3],
            s2=[0, 1, 1, 1, -1],
            missing_state_indicator=-1,
        )
        self.assertAlmostEqual(res, 2 / 3)

    def test_2(self):
        res = hamming_distance(
            s1=[-1, 0, 1, 2, 3],
            s2=[0, 1, 1, 1, -1],
            missing_state_indicator=-2,
        )
        self.assertAlmostEqual(res, 4 / 5)


class Test_ternary_hamming_distance(unittest.TestCase):
    def test_1(self):
        res = ternary_hamming_distance(
            s1=[-1, 0, 1, 2, 3],
            s2=[0, 1, 1, 1, -1],
            missing_state_indicator=-1,
        )
        self.assertAlmostEqual(res, (1 + 2) / 3)

    def test_2(self):
        res = ternary_hamming_distance(
            s1=[-1, 0, 1, 2, 3],
            s2=[0, 1, 1, 1, -1],
            missing_state_indicator=-2,
        )
        self.assertAlmostEqual(res, 5 / 5)


class Test_hamming_distance_correction(unittest.TestCase):
    """
    Tests both `crispr_cas9_expected_hamming_distance` and
    `crispr_cas9_corrected_hamming_distance` using simulated data.
    """

    @parameterized.expand(
        [
            (0.8, 0.8, 0.7, 0.7, 0.5, 0.5, True),
            (0.4, 0.4, 0.3, 0.3, 0.5, 0.5, True),
            (0.4, 0.4, 0.3, 0.3, 1.0, 1.0, True),
            (0.4, 0.4, 0.7, 0.7, 0.5, 0.5, True),
            # Next, we create a mismatch in each parameter in turn.
            (0.4, 0.4, 0.7, 0.7, 0.5, 0.75, False),
            (0.4, 0.4, 0.7, 0.7, 0.75, 0.5, False),
            (0.4, 0.4, 0.7, 0.3, 0.5, 0.5, False),
            (0.4, 0.4, 0.3, 0.7, 0.5, 0.5, False),
            (0.4, 0.7, 0.7, 0.7, 0.5, 0.5, False),
            (0.7, 0.4, 0.7, 0.7, 0.5, 0.5, False),
        ]
    )
    @pytest.mark.slow
    def test_1(
        self,
        ground_truth_height,
        simulation_height,
        ground_truth_mutation_proportion,
        simulation_mutation_proportion,
        ground_truth_collision_probability,
        simulation_collision_probability,
        should_pass,
    ):
        theoretical_expected_hamming_distance = (
            crispr_cas9_expected_hamming_distance(
                height=ground_truth_height,
                mutation_proportion=ground_truth_mutation_proportion,
                collision_probability=ground_truth_collision_probability,
            )
        )

        def get_empirical_expected_hamming_distance(
            height: float,
            mutation_proportion: float,
            collision_probability: float,
        ) -> float:
            # Compute the empirical value with simulation
            q_1 = (1 + np.sqrt(1 - 2 * (1 - collision_probability))) / 2
            sim = Cas9LineageTracingDataSimulator(
                number_of_cassettes=10000,
                size_of_cassette=1,
                mutation_rate=-np.log(1.0 - mutation_proportion),
                number_of_states=2,
                state_priors={
                    1: q_1,
                    2: 1.0 - q_1,
                },
                heritable_silencing_rate=0.1,
                stochastic_silencing_rate=0.1,
                random_seed=42,
                heritable_missing_data_state=-1,
                stochastic_missing_data_state=-1,
            )
            tree = nx.DiGraph()
            tree.add_nodes_from(["0", "1", "2", "3"])
            tree.add_edges_from([("0", "1"), ("1", "2"), ("1", "3")])
            tree = CassiopeiaTree(tree=tree)
            tree.set_times({"0": 0.0, "1": 1.0 - height, "2": 1.0, "3": 1.0})
            sim.overlay_data(tree)
            empirical_expected_hamming_distance = hamming_distance(
                tree.get_character_states("2"),
                tree.get_character_states("3"),
                missing_state_indicator=-1,
            )

            # Check tht inference of distance works
            inferred_distance = crispr_cas9_corrected_hamming_distance(
                mutation_proportion=simulation_mutation_proportion,
                collision_probability=simulation_collision_probability,
                s1=tree.get_character_states("2"),
                s2=tree.get_character_states("3"),
                missing_state_indicator=-1,
            )
            assert abs(inferred_distance - 2 * height) < 0.03

            return empirical_expected_hamming_distance

        empirical_expected_hamming_distance = (
            get_empirical_expected_hamming_distance(
                height=simulation_height,
                mutation_proportion=simulation_mutation_proportion,
                collision_probability=simulation_collision_probability,
            )
        )

        if should_pass:
            assert (
                abs(
                    theoretical_expected_hamming_distance
                    - empirical_expected_hamming_distance
                )
                < 0.01
            )
        else:
            assert (
                abs(
                    theoretical_expected_hamming_distance
                    - empirical_expected_hamming_distance
                )
                > 0.01
            )


class Test_crispr_cas9_expected_ternary_hamming_distance(unittest.TestCase):
    """
    Tests both `crispr_cas9_expected_ternary_hamming_distance` and
    `crispr_cas9_corrected_ternary_hamming_distance` using simulated data.
    """

    @parameterized.expand(
        [
            (0.8, 0.8, 0.7, 0.7, 0.5, 0.5, True),
            (0.4, 0.4, 0.3, 0.3, 0.5, 0.5, True),
            (0.4, 0.4, 0.3, 0.3, 1.0, 1.0, True),
            (0.4, 0.4, 0.7, 0.7, 0.5, 0.5, True),
            # Next, we create a mismatch in each parameter in turn.
            (0.4, 0.4, 0.7, 0.7, 0.5, 0.75, False),
            (0.4, 0.4, 0.7, 0.7, 0.75, 0.5, False),
            (0.4, 0.4, 0.7, 0.3, 0.5, 0.5, False),
            (0.4, 0.4, 0.3, 0.7, 0.5, 0.5, False),
            (0.4, 0.7, 0.7, 0.7, 0.5, 0.5, False),
            (0.7, 0.4, 0.7, 0.7, 0.5, 0.5, False),
        ]
    )
    @pytest.mark.slow
    def test_1(
        self,
        ground_truth_height,
        simulation_height,
        ground_truth_mutation_proportion,
        simulation_mutation_proportion,
        ground_truth_collision_probability,
        simulation_collision_probability,
        should_pass,
    ):
        theoretical_expected_hamming_distance = (
            crispr_cas9_expected_ternary_hamming_distance(
                height=ground_truth_height,
                mutation_proportion=ground_truth_mutation_proportion,
                collision_probability=ground_truth_collision_probability,
            )
        )

        def get_empirical_expected_hamming_distance(
            height: float,
            mutation_proportion: float,
            collision_probability: float,
        ) -> float:
            # Compute the empirical value with simulation
            q_1 = (1 + np.sqrt(1 - 2 * (1 - collision_probability))) / 2
            sim = Cas9LineageTracingDataSimulator(
                number_of_cassettes=10000,
                size_of_cassette=1,
                mutation_rate=-np.log(1.0 - mutation_proportion),
                number_of_states=2,
                state_priors={
                    1: q_1,
                    2: 1.0 - q_1,
                },
                heritable_silencing_rate=0.1,
                stochastic_silencing_rate=0.1,
                random_seed=42,
                heritable_missing_data_state=-1,
                stochastic_missing_data_state=-1,
            )
            tree = nx.DiGraph()
            tree.add_nodes_from(["0", "1", "2", "3"])
            tree.add_edges_from([("0", "1"), ("1", "2"), ("1", "3")])
            tree = CassiopeiaTree(tree=tree)
            tree.set_times({"0": 0.0, "1": 1.0 - height, "2": 1.0, "3": 1.0})
            sim.overlay_data(tree)
            empirical_expected_hamming_distance = ternary_hamming_distance(
                tree.get_character_states("2"),
                tree.get_character_states("3"),
                missing_state_indicator=-1,
            )

            # Check tht inference of distance works
            inferred_distance = crispr_cas9_corrected_ternary_hamming_distance(
                mutation_proportion=simulation_mutation_proportion,
                collision_probability=simulation_collision_probability,
                s1=tree.get_character_states("2"),
                s2=tree.get_character_states("3"),
                missing_state_indicator=-1,
            )
            assert abs(inferred_distance - 2 * height) < 0.03

            return empirical_expected_hamming_distance

        empirical_expected_hamming_distance = (
            get_empirical_expected_hamming_distance(
                height=simulation_height,
                mutation_proportion=simulation_mutation_proportion,
                collision_probability=simulation_collision_probability,
            )
        )
        if should_pass:
            assert (
                abs(
                    theoretical_expected_hamming_distance
                    - empirical_expected_hamming_distance
                )
                < 0.02  # The ternary Hamming distance has larger range
            )
        else:
            assert (
                abs(
                    theoretical_expected_hamming_distance
                    - empirical_expected_hamming_distance
                )
                > 0.02
            )


class Test_CRISPRCas9DistanceCorrectionSolver(unittest.TestCase):
    @pytest.mark.slow
    @parameterized.expand(
        [
            ("crispr_cas9_corrected_hamming_distance",),
            ("crispr_cas9_corrected_ternary_hamming_distance",),
        ]
    )
    @pytest.mark.slow
    def test_smoke(self, distance_corrector_name):
        """
        Just tests that `CRISPRCas9DistanceCorrectionSolver` works on a simple
        tree with just 3 leaves.
        """
        tree = nx.DiGraph()
        tree.add_nodes_from(["0", "1", "2", "3", "4", "5"])
        tree.add_edges_from(
            [("0", "1"), ("1", "2"), ("1", "3"), ("2", "4"), ("2", "5")]
        )
        tree = CassiopeiaTree(tree=tree)
        tree.set_times(
            {"0": 0.0, "1": 0.3, "2": 0.6, "3": 1.0, "4": 1.0, "5": 1.0}
        )
        sim = Cas9LineageTracingDataSimulator(
            number_of_cassettes=10000,
            size_of_cassette=1,
            mutation_rate=-np.log(0.3),  # 70% mutation proportion.
            number_of_states=2,
            state_priors={
                1: 0.5,
                2: 0.5,
            },
            heritable_silencing_rate=0.1,
            stochastic_silencing_rate=0.1,
            random_seed=42,
            heritable_missing_data_state=-1,
            stochastic_missing_data_state=-1,
        )
        solver = CRISPRCas9DistanceCorrectionSolver(
            distance_corrector_name=distance_corrector_name
        )
        sim.overlay_data(tree)
        solver.solve(tree)
        self.assertEqual(tree.get_newick(), "(3,(4,5));")
