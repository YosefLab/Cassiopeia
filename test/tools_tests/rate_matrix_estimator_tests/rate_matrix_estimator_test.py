import unittest

import numpy as np

from cassiopeia.tools.rate_matrix_estimator.rate_matrix_estimator import (
    GeometricQuantizationScheme, MarkovModel, NoQuantizationScheme,
    QuantizationScheme, QuantizedTransitionModel, Statistics, Tree,
    TreeStatistic)


class TestGeometricQuantizationScheme(unittest.TestCase):
    def test_basic(self):
        quantization_scheme = GeometricQuantizationScheme(
            center=0.06,
            step_size=0.01,
            n_steps=50,
        )
        quantization_scheme.construct_grid()
        grid = quantization_scheme.get_grid()
        t_quantized = quantization_scheme.quantize(0.0)
        self.assertEqual(t_quantized, grid[0])
        t_quantized = quantization_scheme.quantize(1000.0)
        self.assertEqual(t_quantized, grid[-1])
        t_quantized = quantization_scheme.quantize(0.06 - 1e-5)
        self.assertEqual(t_quantized, grid[50])
        t_quantized = quantization_scheme.quantize(0.06 + 1e-5)
        self.assertEqual(t_quantized, grid[50])
        t_quantized = quantization_scheme.quantize(0.06 / 1.01)
        self.assertEqual(t_quantized, grid[49])
        t_quantized = quantization_scheme.quantize(
            np.sqrt(0.06 / 1.01 * 0.06) + 1e-5
        )
        self.assertEqual(t_quantized, grid[50])
        t_quantized = quantization_scheme.quantize(
            np.sqrt(0.06 / 1.01 * 0.06) - 1e-5
        )
        self.assertEqual(t_quantized, grid[49])
        t_quantized = quantization_scheme.quantize(-1000.0)
        self.assertEqual(t_quantized, grid[0])
        t_quantized = quantization_scheme.quantize(0.0)
        self.assertEqual(t_quantized, grid[0])
        t_quantized = quantization_scheme.quantize(1e-8)
        self.assertEqual(t_quantized, grid[0])


class TestMarkovModel(unittest.TestCase):
    def test_basic(self):
        markov_model = MarkovModel(
            np.array([[-1.0, 1.0], [1.0, -1.0]]), root_prior=np.ones(2) / 2.0
        )
        p_1 = markov_model.transition_probability_matrix(t=10)
        np.testing.assert_almost_equal(p_1, np.ones(shape=(2, 2)) * 0.5)
        np.testing.assert_almost_equal(
            markov_model.root_prior(), np.ones(2) / 2.0
        )


class TestQuantizedTransitionModel(unittest.TestCase):
    def test_basic(self):
        wrapped_transition_model = MarkovModel(
            np.array([[-1.0, 1.0], [0.5, -0.5]]), root_prior=np.ones(2) / 2.0
        )
        p_1 = wrapped_transition_model.transition_probability_matrix(t=1.0)
        quantization_scheme = GeometricQuantizationScheme(
            center=1.0, n_steps=0, step_size=1.0
        )
        quantized_transition_model = QuantizedTransitionModel(
            wrapped_transition_model, quantization_scheme
        )
        p_1_q = quantized_transition_model.transition_probability_matrix(t=10)
        np.testing.assert_almost_equal(p_1, p_1_q)
        p_1_q = quantized_transition_model.transition_probability_matrix(t=0.1)
        np.testing.assert_almost_equal(p_1, p_1_q)
        self.assertEqual(quantized_transition_model.cache_hits, 1)
        self.assertEqual(quantized_transition_model.cache_misses, 1)
        np.testing.assert_almost_equal(
            quantized_transition_model.root_prior(), np.ones(2) / 2.0
        )


class TestStatistics(unittest.TestCase):
    def test_basic(self):
        tree_statistic_1 = TreeStatistic(
            root_frequencies=np.array([0.2, 0.8]),
            transition_frequencies=[
                (
                    3.14,
                    np.array([[0.5, 1.1], [3.1, 4.4]]),
                ),
                (
                    2.71,
                    np.array([[0.1, 2.1], [4.1, 3.4]]),
                ),
                (
                    0.06,
                    np.array([[4.1, 5.1], [7.1, 6.4]]),
                ),
            ],
        )

        tree_statistic_2 = TreeStatistic(
            root_frequencies=np.array([0.1, 0.9]),
            transition_frequencies=[
                (
                    3.14,
                    np.array([[0.6, 1.2], [3.2, 4.5]]),
                ),
            ],
        )

        self.assertNotEqual(tree_statistic_1, tree_statistic_2)

        quantization_scheme = GeometricQuantizationScheme(
            center=3.0,
            step_size=0.5,
            n_steps=1,
        )

        statistics_1 = Statistics(
            per_tree_statistics=True,
            quantization_scheme=quantization_scheme,
            num_states=2,
        ).add_tree_statistic(
            tree_id=1,
            tree_statistic=tree_statistic_1,
        )

        statistics_2 = Statistics(
            per_tree_statistics=True,
            quantization_scheme=quantization_scheme,
            num_states=2,
        ).add_tree_statistic(
            tree_id=2,
            tree_statistic=tree_statistic_2,
        )

        expected_statistics_for_tree_1 = TreeStatistic(
            root_frequencies=np.array([0.2, 0.8]),
            transition_frequencies=[
                (
                    2.0,
                    np.array([[4.1, 5.1], [7.1, 6.4]]),
                ),
                (
                    3.0,
                    np.array([[0.6, 3.2], [7.2, 7.8]]),
                ),
            ],
        )

        expected_statistics_for_tree_2 = TreeStatistic(
            root_frequencies=np.array([0.1, 0.9]),
            transition_frequencies=[
                (
                    3.0,
                    np.array([[0.6, 1.2], [3.2, 4.5]]),
                ),
            ],
        )

        self.assertNotEqual(
            expected_statistics_for_tree_1, expected_statistics_for_tree_2
        )

        expected_statistics = [
            (1, expected_statistics_for_tree_1),
            (2, expected_statistics_for_tree_2),
        ]

        for statistics_1_plus_2 in [
            statistics_1 + statistics_2,
            statistics_2 + statistics_1,
        ]:
            statistics = statistics_1_plus_2.get_statistics()
            self.assertEqual(statistics, expected_statistics)

            statistics_for_tree_1 = statistics_1_plus_2.get_statistics_for_tree(
                tree_id=1
            )
            self.assertEqual(
                statistics_for_tree_1, expected_statistics_for_tree_1
            )

            statistics_for_tree_2 = statistics_1_plus_2.get_statistics_for_tree(
                tree_id=2
            )
            self.assertEqual(
                statistics_for_tree_2, expected_statistics_for_tree_2
            )

        statistics_1 = Statistics(
            per_tree_statistics=False,
            quantization_scheme=quantization_scheme,
            num_states=2,
        ).add_tree_statistic(
            tree_id=1,
            tree_statistic=tree_statistic_1,
        )

        statistics_2 = Statistics(
            per_tree_statistics=False,
            quantization_scheme=quantization_scheme,
            num_states=2,
        ).add_tree_statistic(
            tree_id=2,
            tree_statistic=tree_statistic_2,
        )

        expected_statistics_for_tree_0 = TreeStatistic(
            root_frequencies=np.array([0.3, 1.7]),
            transition_frequencies=[
                (
                    2.0,
                    np.array([[4.1, 5.1], [7.1, 6.4]]),
                ),
                (
                    3.0,
                    np.array([[1.2, 4.4], [10.4, 12.3]]),
                ),
            ],
        )

        expected_statistics = [
            (0, expected_statistics_for_tree_0),
        ]

        for statistics_1_plus_2 in [
            statistics_1 + statistics_2,
            statistics_2 + statistics_1,
        ]:
            statistics = statistics_1_plus_2.get_statistics()
            self.assertEqual(statistics, expected_statistics)

            statistics_for_tree_0 = statistics_1_plus_2.get_statistics_for_tree(
                tree_id=0
            )
            self.assertEqual(
                statistics_for_tree_0, expected_statistics_for_tree_0
            )


class TestRateMatrixEstimator(unittest.TestCase):
    def test_basic(self):
        pass
