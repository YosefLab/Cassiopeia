"""
This file tests the utilities stored in cassiopeia/data/utilities.py
"""

import unittest

import networkx as nx
import numpy as np
import pandas as pd
import treedata as td

from cassiopeia.data import CassiopeiaTree
from cassiopeia.data import utilities as data_utilities
from cassiopeia.mixins.errors import CassiopeiaError
from cassiopeia.preprocess import utilities as preprocessing_utilities


class TestDataUtilities(unittest.TestCase):
    def setUp(self):

        # this should obey PP for easy checking of ancestral states
        self.character_matrix = pd.DataFrame.from_dict(
            {
                "node3": [1, 0, 0, 0, 0, 0, 0, 0],
                "node7": [1, 1, 0, 0, 0, 0, 0, 0],
                "node9": [1, 1, 1, 0, 0, 0, 0, 0],
                "node11": [1, 1, 1, 1, 0, 0, 0, 0],
                "node13": [1, 1, 1, 1, 1, 0, 0, 0],
                "node15": [1, 1, 1, 1, 1, 1, 0, 0],
                "node17": [1, 1, 1, 1, 1, 1, 1, 0],
                "node18": [1, 1, 1, 1, 1, 1, 1, 1],
                "node5": [2, 0, 0, 0, 0, 0, 0, 0],
                "node6": [2, 2, 0, 0, 0, 0, 0, 0],
            },
            orient="index",
        )

        self.priors = {
            0: {1: 0.5, 2: 0.5},
            1: {1: 0.4, 2: 0.6},
            2: {1: 1.0},
            3: {1: 1.0},
            4: {1: 1.0},
            5: {1: 1.0},
            6: {1: 1.0},
            7: {1: 1.0},
        }

        # Test allele table
        at_dict = {
            "cellBC": ["cellA", "cellA", "cellA", "cellB", "cellC"],
            "intBC": ["A", "B", "C", "A", "C"],
            "r1": ["None", "ATC", "GGG", "ATA", "GAA"],
            "r2": ["None", "AAA", "GAA", "TTT", "GAA"],
            "r3": ["ATC", "TTT", "ATA", "ATA", "ATA"],
            "UMI": [5, 10, 1, 30, 30],
        }

        self.allele_table = pd.DataFrame.from_dict(at_dict)
        self.indel_to_prior = pd.DataFrame.from_dict(
            {
                "ATC": 0.5,
                "GGG": 0.2,
                "GAA": 0.1,
                "AAA": 0.05,
                "TTT": 0.05,
                "ATA": 0.1,
            },
            orient="index",
            columns=["freq"],
        )

        # Test allele table without normal cassiopeia columns
        self.non_cassiopeia_allele_table = self.allele_table.copy()
        self.non_cassiopeia_allele_table.rename(
            columns={"r1": "cs1", "r2": "cs2", "r3": "cs3"}, inplace=True
        )

    def test_bootstrap_character_matrices_no_priors(self):

        random_state = np.random.RandomState(123431235)

        bootstrap_samples = data_utilities.sample_bootstrap_character_matrices(
            self.character_matrix, num_bootstraps=10, random_state=random_state
        )

        self.assertEqual(len(bootstrap_samples), 10)

        for bootstrap_matrix, bootstrap_priors in bootstrap_samples:
            self.assertCountEqual(
                self.character_matrix.index, bootstrap_matrix.index
            )
            self.assertEqual(
                self.character_matrix.shape[1], bootstrap_matrix.shape[1]
            )

            self.assertRaises(
                AssertionError,
                pd.testing.assert_frame_equal,
                self.character_matrix,
                bootstrap_matrix,
            )

    def test_bootstrap_character_matrices_with_priors(self):

        random_state = np.random.RandomState(12345)

        bootstrap_samples = data_utilities.sample_bootstrap_character_matrices(
            self.character_matrix,
            num_bootstraps=10,
            prior_probabilities=self.priors,
            random_state=random_state,
        )

        self.assertEqual(len(bootstrap_samples), 10)

        for bootstrap_matrix, bootstrap_priors in bootstrap_samples:
            self.assertCountEqual(
                self.character_matrix.index, bootstrap_matrix.index
            )
            self.assertEqual(
                self.character_matrix.shape[1], bootstrap_matrix.shape[1]
            )

            self.assertRaises(
                AssertionError,
                pd.testing.assert_frame_equal,
                self.character_matrix,
                bootstrap_matrix,
            )

            self.assertEqual(
                len(bootstrap_priors), self.character_matrix.shape[1]
            )

    def test_bootstrap_allele_tables(self):

        random_state = np.random.RandomState(123431235)

        (
            character_matrix,
            _,
            _,
        ) = preprocessing_utilities.convert_alleletable_to_character_matrix(
            self.allele_table
        )

        bootstrap_samples = data_utilities.sample_bootstrap_allele_tables(
            self.allele_table, num_bootstraps=10, random_state=random_state
        )

        self.assertEqual(len(bootstrap_samples), 10)

        for (
            bootstrap_matrix,
            bootstrap_priors,
            boostarp_state_to_indel,
            bootstrap_intbcs,
        ) in bootstrap_samples:

            self.assertEqual(
                len(bootstrap_intbcs),
                len(self.allele_table["intBC"].unique()) * 3,
            )

            self.assertCountEqual(
                character_matrix.index, bootstrap_matrix.index
            )
            self.assertEqual(
                character_matrix.shape[1], bootstrap_matrix.shape[1]
            )

            self.assertRaises(
                AssertionError,
                pd.testing.assert_frame_equal,
                character_matrix,
                bootstrap_matrix,
            )

    def test_bootstrap_allele_tables_non_cassiopeia_allele_table(self):

        random_state = np.random.RandomState(123431235)

        (
            character_matrix,
            _,
            _,
        ) = preprocessing_utilities.convert_alleletable_to_character_matrix(
            self.non_cassiopeia_allele_table, cut_sites=["cs1", "cs2", "cs3"]
        )

        bootstrap_samples = data_utilities.sample_bootstrap_allele_tables(
            self.non_cassiopeia_allele_table,
            num_bootstraps=10,
            random_state=random_state,
            cut_sites=["cs1", "cs2", "cs3"],
        )

        self.assertEqual(len(bootstrap_samples), 10)

        for (
            bootstrap_matrix,
            bootstrap_priors,
            boostrap_state_to_indel,
            bootstrap_intbcs,
        ) in bootstrap_samples:

            self.assertEqual(
                len(bootstrap_intbcs),
                len(self.non_cassiopeia_allele_table["intBC"].unique()) * 3,
            )

            self.assertCountEqual(
                character_matrix.index, bootstrap_matrix.index
            )
            self.assertEqual(
                character_matrix.shape[1], bootstrap_matrix.shape[1]
            )

            self.assertRaises(
                AssertionError,
                pd.testing.assert_frame_equal,
                character_matrix,
                bootstrap_matrix,
            )

    def test_bootstrap_allele_tables_priors(self):

        random_state = np.random.RandomState(12345)

        (
            character_matrix,
            _,
            _,
        ) = preprocessing_utilities.convert_alleletable_to_character_matrix(
            self.allele_table
        )

        bootstrap_samples = data_utilities.sample_bootstrap_allele_tables(
            self.allele_table,
            num_bootstraps=10,
            indel_priors=self.indel_to_prior,
            random_state=random_state,
        )

        self.assertEqual(len(bootstrap_samples), 10)

        for (
            bootstrap_matrix,
            bootstrap_priors,
            boostrap_state_to_indel,
            bootstrap_intbcs,
        ) in bootstrap_samples:

            self.assertEqual(
                len(bootstrap_intbcs),
                len(self.allele_table["intBC"].unique()) * 3,
            )

            self.assertCountEqual(
                character_matrix.index, bootstrap_matrix.index
            )
            self.assertEqual(
                character_matrix.shape[1], bootstrap_matrix.shape[1]
            )

            self.assertRaises(
                AssertionError,
                pd.testing.assert_frame_equal,
                character_matrix,
                bootstrap_matrix,
            )

            self.assertEqual(len(bootstrap_priors), character_matrix.shape[1])

    def test_to_newick_no_branch_lengths(self):
        tree = nx.DiGraph()
        tree.add_nodes_from(["A", "B", "C", "D", "E", "F"])
        tree.add_edge("F", "A", length=0.1)
        tree.add_edge("F", "B", length=0.2)
        tree.add_edge("F", "E", length=0.5)
        tree.add_edge("E", "C", length=0.3)
        tree.add_edge("E", "D", length=0.4)

        newick_string = data_utilities.to_newick(tree)
        self.assertEqual(newick_string, "(A,B,(C,D));")

    def test_to_newick_branch_lengths(self):
        tree = nx.DiGraph()
        tree.add_nodes_from(["A", "B", "C", "D", "E", "F"])
        tree.add_edge("F", "A", length=0.1)
        tree.add_edge("F", "B", length=0.2)
        tree.add_edge("F", "E", length=0.5)
        tree.add_edge("E", "C", length=0.3)
        tree.add_edge("E", "D", length=0.4)

        newick_string = data_utilities.to_newick(
            tree, record_branch_lengths=True
        )
        self.assertEqual(newick_string, "(A:0.1,B:0.2,(C:0.3,D:0.4):0.5);")

    def test_lca_characters(self):
        vecs = [[1, 0, 3, 4, 5], [1, -1, -1, 3, -1], [1, 2, 3, 2, -1]]
        ret_vec = data_utilities.get_lca_characters(
            vecs, missing_state_indicator=-1
        )
        self.assertEqual(ret_vec, [1, 0, 3, 0, 5])

    def test_lca_characters_ambiguous(self):
        vecs = [
            [(1, 1), (0, 2), (3,), (4,), (5,)],
            [1, -1, -1, 3, -1],
            [1, 2, 3, 2, -1],
        ]
        ret_vec = data_utilities.get_lca_characters(
            vecs, missing_state_indicator=-1
        )
        self.assertEqual(ret_vec, [1, 2, 3, 0, 5])

    def test_lca_characters_ambiguous2(self):

        s1 = [
            (4, 62),
            (3, 10),
            (3, 10, 16),
            (0, 3),
            (0, 2, 3),
            (0, 2, 3),
            (0, 4, 7),
            (0, 2, 23),
            (0, 1, 4, 44),
        ]
        s2 = [4, 3, -1, 0, 0, 0, (0, 7), (0, 2), (0, 4)]

        expected_reconstruction = [
            4,
            3,
            (3, 10, 16),
            0,
            0,
            0,
            (0, 7),
            (0, 2),
            (0, 4),
        ]
        ret_vec = data_utilities.get_lca_characters(
            [s1, s2], missing_state_indicator=-1
        )
        self.assertEqual(ret_vec, expected_reconstruction)

    def test_lca_characters_ambiguous_and_missing(self):
        vecs = [
            [(1, 1), (0, 2), (3, 0), (4,), (5,)],
            [1, -1, -1, 3, -1],
            [1, -1, (3, 0), 2, -1],
        ]
        ret_vec = data_utilities.get_lca_characters(
            vecs, missing_state_indicator=-1
        )
        self.assertEqual(ret_vec, [1, (0, 2), (3, 0), 0, 5])

    def test_resolve_most_abundant(self):
        state = (1, 2, 3, 3)
        self.assertEqual(data_utilities.resolve_most_abundant(state), 3)

    def test_simple_phylogenetic_weights_matrix(self):

        tree = nx.DiGraph()
        tree.add_nodes_from(["A", "B", "C", "D", "E", "F"])
        tree.add_edge("F", "A", length=0.1)
        tree.add_edge("F", "B", length=0.2)
        tree.add_edge("F", "E", length=0.5)
        tree.add_edge("E", "C", length=0.3)
        tree.add_edge("E", "D", length=0.4)

        tree = CassiopeiaTree(tree=tree)

        weight_matrix = data_utilities.compute_phylogenetic_weight_matrix(tree)

        expected_weight_matrix = pd.DataFrame.from_dict(
            {
                "A": [0.0, 0.3, 0.9, 1.0],
                "B": [0.3, 0.0, 1.0, 1.1],
                "C": [0.9, 1.0, 0.0, 0.7],
                "D": [1.0, 1.1, 0.7, 0.0],
            },
            orient="index",
            columns=["A", "B", "C", "D"],
        )

        pd.testing.assert_frame_equal(weight_matrix, expected_weight_matrix)

    def test_simple_phylogenetic_weights_matrix_inverse(self):

        tree = nx.DiGraph()
        tree.add_nodes_from(["A", "B", "C", "D", "E", "F"])
        tree.add_edge("F", "A", length=0.1)
        tree.add_edge("F", "B", length=0.2)
        tree.add_edge("F", "E", length=0.5)
        tree.add_edge("E", "C", length=0.3)
        tree.add_edge("E", "D", length=0.4)

        tree = CassiopeiaTree(tree=tree)

        weight_matrix = data_utilities.compute_phylogenetic_weight_matrix(
            tree, inverse=True
        )

        expected_weight_matrix = pd.DataFrame.from_dict(
            {
                "A": [0.0, 1.0 / 0.3, 1.0 / 0.9, 1.0],
                "B": [1.0 / 0.3, 0.0, 1.0, 1.0 / 1.1],
                "C": [1.0 / 0.9, 1.0, 0.0, 1.0 / 0.7],
                "D": [1.0, 1.0 / 1.1, 1.0 / 0.7, 0.0],
            },
            orient="index",
            columns=["A", "B", "C", "D"],
        )

        pd.testing.assert_frame_equal(weight_matrix, expected_weight_matrix)

    def test_phylogenetic_weights_matrix_inverse_fn(self):

        tree = nx.DiGraph()
        tree.add_nodes_from(["A", "B", "C", "D", "E", "F"])
        tree.add_edge("F", "A", length=0.1)
        tree.add_edge("F", "B", length=0.2)
        tree.add_edge("F", "E", length=0.5)
        tree.add_edge("E", "C", length=0.3)
        tree.add_edge("E", "D", length=0.4)

        tree = CassiopeiaTree(tree=tree)

        weight_matrix = data_utilities.compute_phylogenetic_weight_matrix(
            tree, inverse=True, inverse_fn=lambda x: -np.log(x)
        )

        expected_weight_matrix = pd.DataFrame.from_dict(
            {
                "A": [0.0, -np.log(0.3), -np.log(0.9), 0],
                "B": [-np.log(0.3), 0.0, 0, -np.log(1.1)],
                "C": [-np.log(0.9), 0, 0.0, -np.log(0.7)],
                "D": [0.0, -np.log(1.1), -np.log(0.7), 0.0],
            },
            orient="index",
            columns=["A", "B", "C", "D"],
        )

        pd.testing.assert_frame_equal(weight_matrix, expected_weight_matrix)

    def test_net_relatedness_index(self):

        distances = np.array(
            [[0, 1, 2, 4], [1, 0, 3, 6], [2, 3, 0, 5], [4, 6, 5, 0]]
        )
        indices_1 = np.array([0, 1])
        indices_2 = np.array([2, 3])

        nri = data_utilities.net_relatedness_index(
            distances, indices_1, indices_2
        )
        self.assertAlmostEqual(15.0 / 4.0, nri, delta=0.0001)

    def test_inter_cluster_distance_basic(self):

        tree = nx.DiGraph()
        tree.add_nodes_from(["A", "B", "C", "D", "E", "F"])
        tree.add_edge("F", "A", length=0.1)
        tree.add_edge("F", "B", length=0.2)
        tree.add_edge("F", "E", length=0.5)
        tree.add_edge("E", "C", length=0.3)
        tree.add_edge("E", "D", length=0.4)

        meta_data = pd.DataFrame.from_dict(
            {
                "A": ["TypeA", 10],
                "B": ["TypeA", 5],
                "C": ["TypeB", 3],
                "D": ["TypeB", 22],
            },
            orient="index",
            columns=["CellType", "nUMI"],
        )

        tree = CassiopeiaTree(tree=tree, cell_meta=meta_data)

        inter_cluster_distances = (
            data_utilities.compute_inter_cluster_distances(
                tree, meta_item="CellType"
            )
        )

        expected_distances = pd.DataFrame.from_dict(
            {"TypeA": [0.15, 1.0], "TypeB": [1.0, 0.35]},
            orient="index",
            columns=["TypeA", "TypeB"],
        )

        pd.testing.assert_frame_equal(
            expected_distances, inter_cluster_distances
        )

        self.assertRaises(
            CassiopeiaError,
            data_utilities.compute_inter_cluster_distances,
            tree,
            "nUMI",
        )

    def test_inter_cluster_distance_custom_input(self):

        tree = nx.DiGraph()
        tree.add_nodes_from(["A", "B", "C", "D", "E", "F"])
        tree.add_edge("F", "A", length=0.1)
        tree.add_edge("F", "B", length=0.2)
        tree.add_edge("F", "E", length=0.5)
        tree.add_edge("E", "C", length=0.3)
        tree.add_edge("E", "D", length=0.4)

        meta_data = pd.DataFrame.from_dict(
            {
                "A": ["TypeA", 10],
                "B": ["TypeA", 5],
                "C": ["TypeB", 3],
                "D": ["TypeB", 22],
            },
            orient="index",
            columns=["CellType", "nUMI"],
        )

        weight_matrix = pd.DataFrame.from_dict(
            {
                "A": [0.0, 0.5, 1.2, 0.4],
                "B": [0.5, 0.0, 3.0, 1.1],
                "C": [1.2, 3.0, 0.0, 0.8],
                "D": [0.4, 1.1, 0.8, 0.0],
            },
            orient="index",
            columns=["A", "B", "C", "D"],
        )

        tree = CassiopeiaTree(tree=tree)

        inter_cluster_distances = (
            data_utilities.compute_inter_cluster_distances(
                tree,
                meta_data=meta_data["CellType"],
                dissimilarity_map=weight_matrix,
            )
        )

        expected_distances = pd.DataFrame.from_dict(
            {"TypeA": [0.25, 1.425], "TypeB": [1.425, 0.4]},
            orient="index",
            columns=["TypeA", "TypeB"],
        )

        pd.testing.assert_frame_equal(
            expected_distances,
            inter_cluster_distances,
            check_exact=False,
            atol=0.001,
        )
    

    def test_cassiopeiatree_to_treedata(self):
        graph = nx.DiGraph()
        graph.add_edges_from([('0','1'),('0','2'),('1','3'),('1','4'),('2','5'),('2','6')])
        cm = pd.DataFrame(
            [[0, 1, 3],
            [0, 1, 4],
            [1, 0, 1],
            [1, 0, 2]],
            index= ['3', '4', '5', '6'],
            columns=['char1', 'char2', 'char3']
        )
        cas_tree = CassiopeiaTree(tree = graph, character_matrix = cm)

        obs_tdata = data_utilities.cassiopeia_to_treedata(cas_tree)
    
        self.assertIsNone(obs_tdata.X)
        self.assertIn("lineage", obs_tdata.obst)
        self.assertEqual(obs_tdata.obsm["character_matrix"].shape, (4, 3))
        self.assertEqual(len(obs_tdata.obst["lineage"].nodes), 7)
    
    def test_no_metadata(self):
        graph = nx.DiGraph()
        graph.add_edge('0', '1')
        cm = pd.DataFrame([[1]], index=['1'], columns=['char1'])
        cas_tree = CassiopeiaTree(tree=graph, character_matrix=cm)
        tdata = data_utilities.cassiopeia_to_treedata(cas_tree)
        self.assertIsNotNone(tdata.obs)  # Should create minimal obs
        self.assertIsNotNone(tdata.var)  # Should create minimal var
    
    def test_missing_cm_tree(self):
        cas_tree = CassiopeiaTree()
        with self.assertRaises(CassiopeiaError):
            data_utilities.cassiopeia_to_treedata(cas_tree)

if __name__ == "__main__":
    unittest.main()

