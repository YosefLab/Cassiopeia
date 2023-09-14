"""
Tests the Cas9-based lineage tracing data simulator in
cassiopeia.simulator.Cas9LineageTracingDataSimulator.
"""
import unittest

import networkx as nx
import numpy as np
import pandas as pd

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

        self.basic_lineage_tracing_data_simulator = (
            cas.sim.Cas9LineageTracingDataSimulator(
                number_of_cassettes=3,
                size_of_cassette=3,
                mutation_rate=0.3,
                state_priors={1: 0.1, 2: 0.1, 3: 0.75, 4: 0.05},
                heritable_silencing_rate=1e-5,
                stochastic_silencing_rate=1e-2,
                random_seed=123412232,
            )
        )

        self.basic_lineage_tracing_data_simulator_no_collapse = (
            cas.sim.Cas9LineageTracingDataSimulator(
                number_of_cassettes=3,
                size_of_cassette=3,
                mutation_rate=0.3,
                state_priors={1: 0.1, 2: 0.1, 3: 0.75, 4: 0.05},
                heritable_silencing_rate=0,
                stochastic_silencing_rate=0,
                random_seed=123412232,
                collapse_sites_on_cassette=False,
            )
        )

        self.basic_lineage_tracing_no_resection = (
            cas.sim.Cas9LineageTracingDataSimulator(
                number_of_cassettes=9,
                size_of_cassette=1,
                mutation_rate=0.3,
                state_priors={1: 0.1, 2: 0.1, 3: 0.75, 4: 0.05},
                heritable_silencing_rate=0,
                stochastic_silencing_rate=0,
                random_seed=123412232,
            )
        )

        self.lineage_tracing_data_simulator_state_distribution = (
            cas.sim.Cas9LineageTracingDataSimulator(
                number_of_cassettes=3,
                size_of_cassette=3,
                mutation_rate=0.3,
                state_generating_distribution=lambda: np.random.exponential(
                    1e-5
                ),
                number_of_states=10,
                heritable_silencing_rate=1e-5,
                stochastic_silencing_rate=1e-2,
                random_seed=123412232,
            )
        )

        self.lineage_tracing_data_simulator_missing_data = (
            cas.sim.Cas9LineageTracingDataSimulator(
                number_of_cassettes=3,
                size_of_cassette=3,
                mutation_rate=0.3,
                state_priors={1: 0.1, 2: 0.1, 3: 0.75, 4: 0.05},
                heritable_silencing_rate=1e-5,
                stochastic_silencing_rate=1e-2,
                heritable_missing_data_state=-2,
                stochastic_missing_data_state=-1,
                random_seed=123412232,
            )
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
            9, len(self.basic_lineage_tracing_data_simulator.mutation_priors_per_character)
        )

        self.assertEqual(
            4, len(self.basic_lineage_tracing_data_simulator.mutation_priors)
        )

        self.assertEqual(
            [0.3] * number_of_characters,
            self.basic_lineage_tracing_data_simulator.mutation_rate_per_character,
        )

        for i in range(number_of_characters):
            self.assertAlmostEqual(
                1.0,
                np.sum(
                    [v for v in 
                    self.basic_lineage_tracing_data_simulator.mutation_priors_per_character[i]
                    .values()]
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

        # test for invalid mutation rate type
        with self.assertRaises(DataSimulatorError):
            data_sim = cas.sim.Cas9LineageTracingDataSimulator(
                number_of_cassettes=2,
                size_of_cassette=2, 
                mutation_rate="invalid"
            )

        # test for positive mutation rates
        with self.assertRaises(DataSimulatorError):
            data_sim = cas.sim.Cas9LineageTracingDataSimulator(
                number_of_cassettes=2, size_of_cassette=2, mutation_rate=-0.2
            )

        # test mutation rates list too sort
        with self.assertRaises(DataSimulatorError):
            data_sim = cas.sim.Cas9LineageTracingDataSimulator(
                number_of_cassettes=2,
                size_of_cassette=2,
                mutation_rate=[0.1, 0.1, 0.2],
            )

        # test mutation rates list too long
        with self.assertRaises(DataSimulatorError):
            data_sim = cas.sim.Cas9LineageTracingDataSimulator(
                number_of_cassettes=2,
                size_of_cassette=2,
                mutation_rate=[0.1, 0.1, 0.2, 0.1, 0.1],
            )

        # check for positive mutation rates in a specified array
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
                state_priors={1: 0.5, 2: 0.2},
            )

        # test that state distribution adds up to 1
        with self.assertRaises(DataSimulatorError):
            data_sim = cas.sim.Cas9LineageTracingDataSimulator(
                number_of_cassettes=2,
                size_of_cassette=2,
                state_priors={1: 0.5, 2: 0.6},
            )

        # test incorrect state prior length
        with self.assertRaises(DataSimulatorError):
            data_sim = cas.sim.Cas9LineageTracingDataSimulator(
                number_of_cassettes=2,
                size_of_cassette=2,
                state_priors=[{1: 0.5, 2: 0.6, 3: 0.1}] * 3,
            )

        # incorrect state prior type
        with self.assertRaises(DataSimulatorError):
            data_sim = cas.sim.Cas9LineageTracingDataSimulator(
                number_of_cassettes=2,
                size_of_cassette=2,
                state_priors="invalid",
            )

        # incorrect state prior type
        with self.assertRaises(DataSimulatorError):
            data_sim = cas.sim.Cas9LineageTracingDataSimulator(
                number_of_cassettes=2, 
                size_of_cassette=2,
                state_priors=[1,1,1]
            )

    def test_get_cassettes(self):

        cassettes = self.basic_lineage_tracing_data_simulator.get_cassettes()

        expected_cassettes = [0, 3, 6]

        self.assertCountEqual(expected_cassettes, cassettes)

    def test_introduce_states(self):

        # set random seed
        np.random.seed(123412232)

        character_array = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        updated_character_array = (
            self.basic_lineage_tracing_data_simulator.introduce_states(
                character_array, [0, 3, 5, 6]
            )
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

        updated_character_array = (
            self.basic_lineage_tracing_data_simulator.silence_cassettes(
                character_array, 0.1
            )
        )

        expected_character_array = [0, 0, 0, 0, 0, 0, -1, -1, -1]

        for i in range(len(expected_character_array)):
            self.assertEqual(
                expected_character_array[i], updated_character_array[i]
            )

    def test_collapse_sites(self):

        character_array = [0, 0, 0, 0, 0, 0, 0, 0, 0]

        (
            updated_character_array,
            remaining_cuts,
        ) = self.basic_lineage_tracing_data_simulator.collapse_sites(
            character_array, [0, 1]
        )
        self.assertCountEqual([], remaining_cuts)
        expected_character_array = [-1, -1, 0, 0, 0, 0, 0, 0, 0]
        for i in range(len(expected_character_array)):
            self.assertEqual(
                expected_character_array[i], updated_character_array[i]
            )

        (
            updated_character_array,
            remaining_cuts,
        ) = self.basic_lineage_tracing_data_simulator.collapse_sites(
            character_array, [0, 5, 9]
        )
        self.assertCountEqual([0, 5, 9], remaining_cuts)
        expected_character_array = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        for i in range(len(expected_character_array)):
            self.assertEqual(
                expected_character_array[i], updated_character_array[i]
            )

        # test with alternative missing state indicator
        character_array = [0, 0, 0, 0, 0, 0, 0, 0, 0]

        (
            updated_character_array,
            remaining_cuts,
        ) = self.lineage_tracing_data_simulator_missing_data.collapse_sites(
            character_array, [0, 1]
        )
        self.assertCountEqual([], remaining_cuts)
        expected_character_array = [-2, -2, 0, 0, 0, 0, 0, 0, 0]
        for i in range(len(expected_character_array)):
            self.assertEqual(
                expected_character_array[i], updated_character_array[i]
            )

    def test_simulator_basic(self):

        self.basic_lineage_tracing_data_simulator.overlay_data(self.basic_tree)

        character_matrix = self.basic_tree.character_matrix

        self.assertEqual(9, character_matrix.shape[1])
        self.assertEqual(len(self.basic_tree.leaves), character_matrix.shape[0])

        expected_character_matrix = pd.DataFrame.from_dict(
            {
                "7": [0, 0, 0, 0, 3, 0, 0, 0, 3],
                "8": [0, 0, 3, 0, 0, 3, 0, 0, 3],
                "9": [3, 4, 0, 0, 0, 0, 3, 0, 0],
                "10": [3, 1, 0, 0, 0, 0, 0, 0, 3],
                "11": [-1, -1, -1, -1, -1, -1, 0, -1, -1],
                "12": [-1, -1, -1, -1, -1, -1, 3, 0, 0],
                "13": [-1, -1, -1, 0, -1, -1, -1, -1, -1],
                "14": [-1, -1, -1, 0, -1, -1, 0, 3, 0],
            },
            orient="index",
            columns=[0, 1, 2, 3, 4, 5, 6, 7, 8],
        )

        pd.testing.assert_frame_equal(
            expected_character_matrix, character_matrix
        )

        # check inheritance patterns
        for n in self.basic_tree.depth_first_traverse_nodes(postorder=False):

            if self.basic_tree.is_root(n):
                self.assertCountEqual(
                    [0] * 9, self.basic_tree.get_character_states(n)
                )
                continue

            parent = self.basic_tree.parent(n)

            child_array = self.basic_tree.get_character_states(n)
            parent_array = self.basic_tree.get_character_states(parent)
            for i in range(len(child_array)):

                if parent_array[i] == -1:
                    self.assertEqual(-1, child_array[i])

                if parent_array[i] != 0:
                    self.assertNotEqual(0, child_array[i])

    def test_no_collapse(self):
        self.basic_lineage_tracing_data_simulator_no_collapse.overlay_data(
            self.basic_tree
        )
        character_matrix = self.basic_tree.character_matrix

        self.assertEqual(9, character_matrix.shape[1])
        self.assertEqual(len(self.basic_tree.leaves), character_matrix.shape[0])

        expected_character_matrix = pd.DataFrame.from_dict(
            {
                "7": [0, 0, 0, 0, 3, 0, 0, 0, 3],
                "8": [0, 0, 3, 0, 0, 3, 0, 0, 3],
                "9": [3, 4, 0, 0, 0, 0, 3, 0, 0],
                "10": [3, 1, 0, 0, 0, 0, 0, 0, 3],
                "11": [1, 4, 2, 0, 0, 3, 3, 0, 0],
                "12": [1, 4, 2, 0, 3, 3, 3, 0, 3],
                "13": [1, 4, 2, 3, 3, 0, 0, 0, 0],
                "14": [1, 4, 2, 3, 3, 3, 0, 0, 0],
            },
            orient="index",
            columns=[0, 1, 2, 3, 4, 5, 6, 7, 8],
        )

        pd.testing.assert_frame_equal(
            expected_character_matrix, character_matrix
        )

    def test_no_resection(self):

        self.basic_lineage_tracing_no_resection.overlay_data(self.basic_tree)

        character_matrix = self.basic_tree.character_matrix

        self.assertEqual(9, character_matrix.shape[1])
        self.assertEqual(len(self.basic_tree.leaves), character_matrix.shape[0])

        expected_character_matrix = pd.DataFrame.from_dict(
            {
                "7": [0, 3, 3, 0, 0, 3, 3, 0, 0],
                "8": [0, 0, 3, 0, 0, 3, 0, 0, 0],
                "9": [4, 0, 0, 0, 3, 0, 0, 0, 2],
                "10": [0, 3, 3, 0, 3, 0, 0, 3, 0],
                "11": [3, 0, 3, 0, 3, 3, 1, 3, 0],
                "12": [3, 3, 3, 3, 3, 3, 0, 3, 3],
                "13": [3, 0, 0, 0, 3, 3, 0, 3, 3],
                "14": [0, 3, 0, 3, 3, 3, 3, 3, 3],
            },
            orient="index",
            columns=[0, 1, 2, 3, 4, 5, 6, 7, 8],
        )

        pd.testing.assert_frame_equal(
            expected_character_matrix, character_matrix
        )

        # check inheritance patterns
        for n in self.basic_tree.depth_first_traverse_nodes(postorder=False):

            if self.basic_tree.is_root(n):
                self.assertCountEqual(
                    [0] * 9, self.basic_tree.get_character_states(n)
                )
                continue

            parent = self.basic_tree.parent(n)

            child_array = self.basic_tree.get_character_states(n)
            parent_array = self.basic_tree.get_character_states(parent)
            for i in range(len(child_array)):

                if parent_array[i] == -1:
                    self.assertEqual(-1, child_array[i])

                if parent_array[i] != 0:
                    self.assertNotEqual(0, child_array[i])

    def test_simulator_with_state_generating_distribution(self):

        self.lineage_tracing_data_simulator_state_distribution.overlay_data(
            self.basic_tree
        )

        self.assertEqual(
            10,
            len(
                self.lineage_tracing_data_simulator_state_distribution.mutation_priors_per_character[0]
            ),
        )

        character_matrix = self.basic_tree.character_matrix

        self.assertEqual(9, character_matrix.shape[1])
        self.assertEqual(len(self.basic_tree.leaves), character_matrix.shape[0])

        # check inheritance patterns
        for n in self.basic_tree.depth_first_traverse_nodes(postorder=False):

            if self.basic_tree.is_root(n):
                self.assertCountEqual(
                    [0] * 9, self.basic_tree.get_character_states(n)
                )
                continue

            parent = self.basic_tree.parent(n)

            child_array = self.basic_tree.get_character_states(n)
            parent_array = self.basic_tree.get_character_states(parent)
            for i in range(len(child_array)):

                if parent_array[i] == -1:
                    self.assertEqual(-1, child_array[i])

                if parent_array[i] != 0:
                    self.assertNotEqual(0, child_array[i])

    def test_simulator_with_per_character_priors(self):

        sim_from_dictionary = (
            cas.sim.Cas9LineageTracingDataSimulator(
                number_of_cassettes=2,
                size_of_cassette=3,
                number_of_states=4,
                state_priors={i: 0.25 for i in range(4)}
            )
        )

        sim_from_array_len_3 = (
            cas.sim.Cas9LineageTracingDataSimulator(
                number_of_cassettes=2,
                size_of_cassette=3,
                number_of_states=4,
                state_priors=[{i: 0.25 for i in range(4)}] * 3
            )
        )

        sim_from_array_len_6 = (
            cas.sim.Cas9LineageTracingDataSimulator(
                number_of_cassettes=2,
                size_of_cassette=3,
                number_of_states=4,
                state_priors=np.array([{i: 0.25 for i in range(4)}] * 6)
            )
        )
        
        self.assertEqual(sim_from_dictionary.mutation_priors_per_character,
                            sim_from_array_len_3.mutation_priors_per_character)
        
        self.assertEqual(sim_from_array_len_3.mutation_priors_per_character,
                            sim_from_array_len_6.mutation_priors_per_character)
        
    def test_simulator_with_per_character_rates(self):

        sim_from_float = (
            cas.sim.Cas9LineageTracingDataSimulator(
                number_of_cassettes=2,
                size_of_cassette=3,
                number_of_states=4,
                mutation_rate = .1
            )
        )

        sim_from_array_len_3 = (
            cas.sim.Cas9LineageTracingDataSimulator(
                number_of_cassettes=2,
                size_of_cassette=3,
                number_of_states=4,
                mutation_rate = [.1] * 3
            )
        )

        sim_from_array_len_6 = (
            cas.sim.Cas9LineageTracingDataSimulator(
                number_of_cassettes=2,
                size_of_cassette=3,
                number_of_states=4,
                mutation_rate = np.array([.1] * 6)
            )
        )
        
        self.assertEqual(sim_from_float.mutation_rate_per_character ,
                            sim_from_array_len_3.mutation_rate_per_character)
        
        self.assertEqual(sim_from_array_len_3.mutation_rate_per_character ,
                            sim_from_array_len_6.mutation_rate_per_character)


if __name__ == "__main__":
    unittest.main()
