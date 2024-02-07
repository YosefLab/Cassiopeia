"""
Tests the Sequential-based lineage tracing data simulator in
cassiopeia.simulator.SequentialLineageTracingDataSimulator.
"""
import unittest

import networkx as nx
import numpy as np
import pandas as pd

import cassiopeia as cas
from cassiopeia.simulator.DataSimulator import DataSimulatorError

class TestSequentialLineageTracingDataSimulator(unittest.TestCase):
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
        self.priors = {1: 0.1, 2: 0.1, 3: 0.75, 4: 0.05}

        self.tracing_data_simulator = (
            cas.sim.SequentialLineageTracingDataSimulator(
                number_of_cassettes=3,
                size_of_cassette=3,
                initiation_rate=0.3,
                continuation_rate=0.4,
                state_priors=self.priors,
                heritable_silencing_rate=0,
                stochastic_silencing_rate=0,
                random_seed=123412232,
            )
        )

        self.tracing_data_simulator_with_missing = (
            cas.sim.SequentialLineageTracingDataSimulator(
                number_of_cassettes=3,
                size_of_cassette=3,
                initiation_rate=0.3,
                continuation_rate=0.4,
                state_priors=self.priors,
                heritable_silencing_rate=.2,
                stochastic_silencing_rate=.1,
                random_seed=123412232,
            )
        )

    def test_basic_setup(self):

        number_of_characters = (
            self.tracing_data_simulator.number_of_cassettes
            * self.tracing_data_simulator.size_of_cassette
        )
        self.assertEqual(9, number_of_characters)

        self.assertEqual(
            0,
            self.tracing_data_simulator.heritable_silencing_rate,
        )

        self.assertEqual(
            0,
            self.tracing_data_simulator.stochastic_silencing_rate,
        )

        self.assertEqual(
            4, len(self.tracing_data_simulator.state_priors)
        )

        self.assertEqual(
            0.3, self.tracing_data_simulator.initiation_rate
        )

        self.assertEqual(
            0.4, self.tracing_data_simulator.continuation_rate
        )

    def test_setup_errors(self):

        # test number of cassettes is not a positive integer
        with self.assertRaises(DataSimulatorError):
            data_sim = cas.sim.SequentialLineageTracingDataSimulator(
                number_of_cassettes=0, size_of_cassette=2,
                state_priors=self.priors
            )

        with self.assertRaises(DataSimulatorError):
            data_sim = cas.sim.SequentialLineageTracingDataSimulator(
                number_of_cassettes=0.1, size_of_cassette=2,
                state_priors=self.priors
            )

        # test size of cassette is not a positive integer
        with self.assertRaises(DataSimulatorError):
            data_sim = cas.sim.SequentialLineageTracingDataSimulator(
                number_of_cassettes=2, size_of_cassette=0,
                state_priors=self.priors
            )

        with self.assertRaises(DataSimulatorError):
            data_sim = cas.sim.SequentialLineageTracingDataSimulator(
                number_of_cassettes=2, size_of_cassette=0.1,
                state_priors=self.priors
            )

        # test for invalid continuation rate type
        with self.assertRaises(DataSimulatorError):
            data_sim = cas.sim.SequentialLineageTracingDataSimulator(
                number_of_cassettes=2,
                size_of_cassette=2, 
                continuation_rate="invalid",
                state_priors=self.priors
            )

        # test for invalid initiation rate type
        with self.assertRaises(DataSimulatorError):
            data_sim = cas.sim.SequentialLineageTracingDataSimulator(
                number_of_cassettes=2,
                size_of_cassette=2, 
                initiation_rate="invalid",
                state_priors=self.priors
            )

        # test for positive continuation rate
        with self.assertRaises(DataSimulatorError):
            data_sim = cas.sim.SequentialLineageTracingDataSimulator(
                number_of_cassettes=2, size_of_cassette=2, 
                continuation_rate=-0.2,state_priors=self.priors
            )

        # test for positive initiation rate
        with self.assertRaises(DataSimulatorError):
            data_sim = cas.sim.SequentialLineageTracingDataSimulator(
                number_of_cassettes=2, size_of_cassette=2, 
                initiation_rate=-0.2,state_priors=self.priors
            )

        # test that state distribution adds up to 1
        with self.assertRaises(DataSimulatorError):
            data_sim = cas.sim.SequentialLineageTracingDataSimulator(
                number_of_cassettes=2,
                size_of_cassette=2,
                state_priors={1: 0.5, 2: 0.2},
            )

        # test negative state prior
        with self.assertRaises(DataSimulatorError):
            data_sim = cas.sim.SequentialLineageTracingDataSimulator(
                number_of_cassettes=2,
                size_of_cassette=2,
                state_priors={1: 1.2, 2: -0.2},
            )

        # incorrect state prior type
        with self.assertRaises(DataSimulatorError):
            data_sim = cas.sim.SequentialLineageTracingDataSimulator(
                number_of_cassettes=2,
                size_of_cassette=2,
                state_priors="invalid",
            )

    def test_simulator_basic(self):

        self.tracing_data_simulator.overlay_data(self.basic_tree)

        character_matrix = self.basic_tree.character_matrix

        self.assertEqual(9, character_matrix.shape[1])
        self.assertEqual(len(self.basic_tree.leaves), character_matrix.shape[0])

        expected_character_matrix = pd.DataFrame.from_dict(
            {
                "7": [0, 0, 0, 0, 0, 0, 0, 0, 0],
                "8": [0, 0, 0, 0, 0, 0, 3, 0, 0],
                "9": [0, 0, 0, 0, 0, 0, 0, 0, 0],
                "10": [0, 0, 0, 3, 0, 0, 3, 0, 0],
                "11": [0, 0, 0, 1, 3, 0, 0, 0, 0],
                "12": [0, 0, 0, 1, 0, 0, 0, 0, 0],
                "13": [3, 0, 0, 1, 3, 3, 1, 3, 0],
                "14": [3, 0, 0, 1, 3, 4, 2, 0, 0],
            },
            orient="index",
            columns=[0, 1, 2, 3, 4, 5, 6, 7, 8],
        )

        pd.testing.assert_frame_equal(
            expected_character_matrix, character_matrix
        )

        self.basic_tree.reconstruct_ancestral_characters()

        # check inheritance patterns
        for n in self.basic_tree.depth_first_traverse_nodes(postorder=False):

            if self.basic_tree.is_root(n):
                continue

            parent = self.basic_tree.parent(n)

            child_array = self.basic_tree.get_character_states(n)
            parent_array = self.basic_tree.get_character_states(parent)
            for i in range(len(child_array)):

                if parent_array[i] == -1:
                    self.assertEqual(-1, child_array[i])

                if parent_array[i] != 0:
                    self.assertNotEqual(0, child_array[i])

    def test_simulator_with_missing(self):

        self.tracing_data_simulator_with_missing.overlay_data(self.basic_tree)

        character_matrix = self.basic_tree.character_matrix

        expected_character_matrix = pd.DataFrame.from_dict(
            {
                "7": [0, 0, 0, -1, -1, -1, 0, 0, 0],
                "8": [-1, -1, -1, -1, -1, -1, 0, 0, 0],
                "9": [0, 0, 0, 4, 0, 0, -1, -1, -1],
                "10": [-1, -1, -1, 4, 3, 0, -1, -1, -1],
                "11": [-1, -1, -1, 0, 0, 0, -1, -1, -1],
                "12": [3, 3, 0, 3, 0, 0, -1, -1, -1],
                "13": [-1, -1, -1, 0, 0, 0, -1, -1, -1],
                "14": [-1, -1, -1, -1, -1, -1, -1, -1, -1],
            },
            orient="index",
            columns=[0, 1, 2, 3, 4, 5, 6, 7, 8],
        ) 

        pd.testing.assert_frame_equal(
            expected_character_matrix, character_matrix
        )


    def test_branch_multiple_edits(self):
        topology = nx.DiGraph()
        topology.add_edges_from(
            [
                ("0", "1"),
                ("0", "2"),
            ]
        )

        tree = cas.data.CassiopeiaTree(tree=topology)
        tree.set_times(
            {
                "0": 0,
                "1": 100,
                "2": 100,
            }
        )

        data_simulator = cas.sim.SequentialLineageTracingDataSimulator(
            number_of_cassettes=3,
            size_of_cassette=3,
            initiation_rate=0.3,
            continuation_rate=0.4,
            state_priors={1:0.5,2:0.5},
            heritable_silencing_rate=0,
            stochastic_silencing_rate=0,
            random_seed=123412232,
        )
        data_simulator.overlay_data(tree)

        expected_character_matrix = pd.DataFrame.from_dict(
            {
                "1": [1, 2, 2, 1, 1, 1, 1, 2, 2],
                "2": [2, 2, 2, 2, 2, 1, 1, 1, 1],
            },
            orient="index",
            columns=[0, 1, 2, 3, 4, 5, 6, 7, 8],
        )

        pd.testing.assert_frame_equal(
            expected_character_matrix, tree.character_matrix
        )

if __name__ == "__main__":
    unittest.main()