"""
Tests the Cas9-based lineage tracing data simulator in
cassiopeia.simulator.Cas9LineageTracingDataSimulator.
"""
import unittest

import networkx as nx
import numpy as np

import cassiopeia as cas
from cassiopeia.simulator.DataSimulator import DataSimulatorError


class TestCas9LineageTracingDataSimulator(unittest.TestCase):
    def setUp(self):

        topology = nx.DiGraph()
        topology.add_edges_from(
            [
                ("0", "1"),
                ("0", "2"),
                ("1", "3"),
                ("1", "4"),
                ("2", "5"),
                ("2", "6"),
                ("3", "7"),
                ("3", "8"),
                ("4", "9"),
                ("4", "10"),
                ("5", "11"),
                ("5", "12"),
                ("6", "13"),
                ("6", "14"),
            ]
        )

        tree = cas.data.CassiopeiaTree(tree=topology)
        tree.set_times(
            {
                "0": 0,
                "1": 1,
                "2": 1,
                "3": 2,
                "4": 2,
                "5": 2,
                "6": 2,
                "7": 3,
                "8": 3,
                "9": 3,
                "10": 3,
                "11": 3,
                "12": 3,
                "13": 3,
                "14": 3,
            }
        )

        self.basic_tree = tree

        self.basic_lineage_tracing_data_simulator = cas.sim.Cas9LineageTracingDataSimulator(
            number_of_cassettes=3,
            size_of_cassette=3,
            mutation_rate=0.3,
            mutation_priors={1: 0.1, 2: 0.1, 3: 0.75, 4: 0.05},
            heritable_silencing_rate=1e-5,
            stochastic_silencing_rate=1e-2,
            random_seed=123412232,
        )

    def test_basic_setup(self):

        number_of_characters = (
            self.basic_lineage_tracing_data_simulator.number_of_cassettes
            * self.basic_lineage_tracing_data_simulator.size_of_cassette
        )
        self.assertEqual(9, number_of_characters)

        self.assertEqual(
            1e-5,
            self.basic_lineage_tracing_data_simulator.heritable_silencing_rate,
        )

        self.assertEqual(
            1e-2,
            self.basic_lineage_tracing_data_simulator.stochastic_silencing_rate,
        )

        self.assertEqual(
            4, len(self.basic_lineage_tracing_data_simulator.mutation_priors)
        )

        self.assertEqual(
            [0.3] * number_of_characters,
            self.basic_lineage_tracing_data_simulator.mutation_rate_per_character,
        )

        self.assertAlmostEqual(
            1.0,
            np.sum(
                [
                    v
                    for v in self.basic_lineage_tracing_data_simulator.mutation_priors.values()
                ]
            ),
        )

    def test_setup_errors(self):

        # test number of cassettes is not a positive integer
        with self.assertRaises(DataSimulatorError):
            data_sim = cas.sim.Cas9LineageTracingDataSimulator(
                number_of_cassettes=0, size_of_cassette=2
            )

        with self.assertRaises(DataSimulatorError):
            data_sim = cas.sim.Cas9LineageTracingDataSimulator(
                number_of_cassettes=0.1, size_of_cassette=2
            )

        # test size of cassette is not a positive integer
        with self.assertRaises(DataSimulatorError):
            data_sim = cas.sim.Cas9LineageTracingDataSimulator(
                number_of_cassettes=2, size_of_cassette=0
            )

        with self.assertRaises(DataSimulatorError):
            data_sim = cas.sim.Cas9LineageTracingDataSimulator(
                number_of_cassettes=2, size_of_cassette=0.1
            )

        # test for positive mutation rates
        with self.assertRaises(DataSimulatorError):
            data_sim = cas.sim.Cas9LineageTracingDataSimulator(
                number_of_cassettes=2, size_of_cassette=2, mutation_rate=-0.2
            )

        # test for correct number of mutation rates (one per character)
        with self.assertRaises(DataSimulatorError):
            data_sim = cas.sim.Cas9LineageTracingDataSimulator(
                number_of_cassettes=2,
                size_of_cassette=2,
                mutation_rate=[0.1, 0.1, 0.2],
            )

        # check for postive mutation rates in a specified array
        with self.assertRaises(DataSimulatorError):
            data_sim = cas.sim.Cas9LineageTracingDataSimulator(
                number_of_cassettes=2,
                size_of_cassette=2,
                mutation_rate=[0.1, 0.1, 0.2, -0.1],
            )

        # test that state distribution adds up to 1
        with self.assertRaises(DataSimulatorError):
            data_sim = cas.sim.Cas9LineageTracingDataSimulator(
                number_of_cassettes=2,
                size_of_cassette=2,
                mutation_priors={1: 0.5, 2: 0.2},
            )

        with self.assertRaises(DataSimulatorError):
            data_sim = cas.sim.Cas9LineageTracingDataSimulator(
                number_of_cassettes=2,
                size_of_cassette=2,
                mutation_priors={1: 0.5, 2: 0.6},
            )

    def test_get_cassettes(self):

        cassettes = self.basic_lineage_tracing_data_simulator.get_cassettes()

        expected_cassettes = [0, 3, 6]

        self.assertCountEqual(expected_cassettes, cassettes)

    def test_introduce_states(self):

        # set random seed
        np.random.seed(123412232)

        character_array = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        updated_character_array = self.basic_lineage_tracing_data_simulator.introduce_states(
            character_array, [0, 3, 5, 6]
        )

        expected_character_array = [3, 0, 0, 3, 0, 3, 3, 0, 0]

        for i in range(len(expected_character_array)):
            self.assertEqual(
                expected_character_array[i], updated_character_array[i]
            )

    def test_silence_cassettes(self):

        # set random seed
        np.random.seed(1)

        character_array = [0, 0, 0, 0, 0, 0, 0, 0, 0]

        updated_character_array = self.basic_lineage_tracing_data_simulator.silence_cassettes(
            character_array, 0.1
        )

        expected_character_array = [0, 0, 0, 0, 0, 0, -1, -1, -1]

        for i in range(len(expected_character_array)):
            self.assertEqual(
                expected_character_array[i], updated_character_array[i]
            )

    def test_collapse_sites(self):

        character_array = [0, 0, 0, 0, 0, 0, 0, 0, 0]

        updated_character_array, remaining_cuts = self.basic_lineage_tracing_data_simulator.collapse_sites(
            character_array, [0, 1]
        )
        self.assertCountEqual([], remaining_cuts)
        expected_character_array = [-1, -1, 0, 0, 0, 0, 0, 0, 0]
        for i in range(len(expected_character_array)):
            self.assertEqual(
                expected_character_array[i], updated_character_array[i]
            )

        updated_character_array, remaining_cuts = self.basic_lineage_tracing_data_simulator.collapse_sites(
            character_array, [0, 5, 9]
        )
        self.assertCountEqual([0, 5, 9], remaining_cuts)
        expected_character_array = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        for i in range(len(expected_character_array)):
            self.assertEqual(
                expected_character_array[i], updated_character_array[i]
            )

if __name__ == "__main__":
    unittest.main()
