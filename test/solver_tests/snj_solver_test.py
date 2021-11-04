"""
Test SpectralNeighborJoiningSolver in Cassiopeia.solver.
"""
import unittest
from typing import Dict, Optional
from unittest import mock

import itertools
import networkx as nx
import numpy as np
from numpy.testing._private.utils import assert_equal
import pandas as pd


# Only for debugging purposes
# if True:
#     import sys
#     sys.path.insert(0, '../..')

import cassiopeia as cas
from cassiopeia.solver.dissimilarity_functions import exponential_negative_hamming_distance

def find_triplet_structure(triplet, T):
    a, b, c = triplet[0], triplet[1], triplet[2]
    a_ancestors = [node for node in nx.ancestors(T, a)]
    b_ancestors = [node for node in nx.ancestors(T, b)]
    c_ancestors = [node for node in nx.ancestors(T, c)]
    ab_common = len(set(a_ancestors) & set(b_ancestors))
    ac_common = len(set(a_ancestors) & set(c_ancestors))
    bc_common = len(set(b_ancestors) & set(c_ancestors))
    structure = "-"
    if ab_common > bc_common and ab_common > ac_common:
        structure = "ab"
    elif ac_common > bc_common and ac_common > ab_common:
        structure = "ac"
    elif bc_common > ab_common and bc_common > ac_common:
        structure = "bc"
    return structure


class TestSpectralNeighborJoiningSolver(unittest.TestCase):
    def setUp(self):
        self.snj_solver = cas.solver.SpectralNeighborJoiningSolver(add_root=True)

        # --------------------- General SNJ ---------------------
        cm_general = pd.DataFrame.from_dict(
            {
                "a": [0, 1, 2],
                "b": [1, 1, 2],
                "c": [2, 2, 2],
                "d": [1, 1, 1],
                "e": [0, 0, 0],
            },
            orient="index",
            columns=["x1", "x2", "x3"],
        )

        sim_general = pd.DataFrame.from_dict(
            {
                "a": [0.0, 1/3, 1.0, 1.0, 2/3],
                "b": [1/3, 0.0, 4/3, 2/3, 1.0],
                "c": [1.0, 4/3, 0.0, 2.0, 1.0],
                "d": [1.0, 2/3, 2.0, 0.0, 1.0],
                "e": [2/3, 1.0, 1.0, 1.0, 0.0],
            },
            orient="index",
            columns=["a", "b", "c", "d", "e"],
        )

        self.cm_general = cm_general
        self.sim_general = sim_general
        self.tree_general = cas.data.CassiopeiaTree(
            character_matrix=cm_general, dissimilarity_map=sim_general, root_sample_name="b"
        )

        # ---------------- Lineage Tracing SNJ ----------------

        cm_lineage = pd.DataFrame.from_dict(
            {
                "a": [1, 1, 0],
                "b": [1, 2, 0],
                "c": [1, 2, 1],
                "d": [2, 0, 0],
                "e": [2, 0, 2],
            },
            orient="index",
            columns=["x1", "x2", "x3"],
        )

        self.tree_lineage = cas.data.CassiopeiaTree(character_matrix=cm_lineage)

        # ------------- CM with Duplicates -----------------------
        cm_dupe = pd.DataFrame.from_dict(
            {
                "a": [1, 1, 0],
                "b": [1, 2, 0],
                "c": [1, 2, 1],
                "d": [2, 0, 0],
                "e": [2, 0, 2],
                "f": [2, 0, 2],
            },
            orient="index",
            columns=["x1", "x2", "x3"],
        )

        self.tree_dupe = cas.data.CassiopeiaTree(
            character_matrix=cm_dupe
        )

        # ------------- SNJ with modified hamming dissimilarity ------------
        priors = {0: {1: 0.5, 2: 0.5}, 1: {1: 0.2, 2: 0.8}, 2: {1: 0.3, 2: 0.7}}
        self.tree_lineage_priors = cas.data.CassiopeiaTree(
            character_matrix=cm_lineage, priors=priors
        )

    def test_constructor(self):
        self.assertIsNotNone(self.snj_solver.dissimilarity_function)
        self.assertIsNotNone(self.tree_general.get_dissimilarity_map())

        nothing_solver = cas.solver.SpectralNeighborJoiningSolver(
            dissimilarity_function=None, add_root=False
        )

        no_root_tree = cas.data.CassiopeiaTree(
            character_matrix=self.cm_general,
            dissimilarity_map=self.sim_general,
        )

        with self.assertRaises(cas.solver.DistanceSolver.DistanceSolverError):
            nothing_solver.solve(no_root_tree)

        no_root_solver = cas.solver.SpectralNeighborJoiningSolver(
            dissimilarity_function=None, add_root=True
        )

        with self.assertRaises(cas.solver.DistanceSolver.DistanceSolverError):
            no_root_solver.solve(no_root_tree)

        root_only_tree = cas.data.CassiopeiaTree(
            character_matrix=self.cm_general, root_sample_name="b"
        )

        with self.assertRaises(cas.solver.DistanceSolver.DistanceSolverError):
            nothing_solver.solve(root_only_tree)

        snj_solver_fn = cas.solver.SpectralNeighborJoiningSolver(
            add_root=True 
        ) 
        snj_solver_fn.solve(self.tree_general)

        self.assertEqual(
            self.tree_general.get_dissimilarity_map().loc["a", "b"], 1/3
        )

    def test_compute_lambda_pairwise(self):
        self.lambda_pairwise = [[i] for i in range(self.sim_general.values.shape[0])]

        # run to generate similarity map
        self.snj_solver.get_dissimilarity_map(self.tree_lineage)

        svd2_vals = self.snj_solver.compute_lambda(self.lambda_pairwise)

        expected_q = pd.DataFrame.from_dict(
            {
                "state0": [np.inf, 0.43661943, 0.31344894, 0.7780976 , 0.67849699],
                "state1": [0.43661943, np.inf, 0.09190727, 1.01535639, 0.86536494],
                "state2": [0.31344894, 0.09190727, np.inf, 0.95520247, 1.01083351],
                "state3": [0.7780976 , 1.01535639, 0.95520247, np.inf, 0.04942259],
                "state4": [0.67849699, 0.86536494, 1.01083351, 0.04942259, np.inf],
            },
            orient="index",
            columns=["state0", "state2", "state3", "state4", "state5"],
        )

        self.assertTrue(np.allclose(svd2_vals, expected_q, atol=0.1))

    def test_compute_lambda_N3(self):
        self.lambda_pairwise = [[0], [1, 2], [3, 4]]

        # run to generate similarity map
        self.snj_solver.get_dissimilarity_map(self.tree_lineage)
        
        svd2_vals = self.snj_solver.compute_lambda(self.lambda_pairwise)

        expected_q = np.array([
            [np.inf, 0.04942259, 0.09190727],
            [0.04942259, np.inf, 2.05480467],
            [0.09190727, 2.05480467, np.inf]
            ])

        self.assertTrue(np.allclose(svd2_vals, expected_q, atol=0.1))

    def test_update_dissimilarity_map_base(self):
        # run to generate similarity map
        self.snj_solver.get_dissimilarity_map(self.tree_lineage)
        self.snj_solver.lambda_indices = [[0], [1], [2], [3], [4]]
        lambda_matrix_arr = self.snj_solver.compute_lambda(
            self.snj_solver.lambda_indices
            )
        
        node_names = self.sim_general.index
        lambda_matrix = pd.DataFrame(
            lambda_matrix_arr, 
            index=node_names, 
            columns=node_names
            )

        node_i, node_j = (
                lambda_matrix.index[0],
                lambda_matrix.index[1],
            )
        
        svd2_vals = self.snj_solver.update_dissimilarity_map(
            lambda_matrix,
            (node_i, node_j),
            'new_node'
        )

        expected_lambda = np.array([
            [np.inf, 0.95520247, 1.01083351, 0.04942259],
            [0.95520247, np.inf, 0.04942259, 1.01083351],
            [1.01083351, 0.04942259, np.inf, 0.95520247],
            [0.04942259, 1.01083351, 0.95520247, np.inf]
       ])

        self.assertTrue(np.allclose(svd2_vals, expected_lambda, atol=0.1))
        self.assertListEqual(svd2_vals.index.values.tolist(), ['c', 'd', 'e', 'new_node'])
        self.assertListEqual(svd2_vals.columns.values.tolist(), ['c', 'd', 'e', 'new_node'])

    def test_update_dissimilarity_map_N3(self):
        # run to generate similarity map
        self.snj_solver.get_dissimilarity_map(self.tree_lineage)
        self.snj_solver.lambda_indices = [[0], [1, 2], [3, 4]]
        lambda_matrix_arr = self.snj_solver.compute_lambda(
            self.snj_solver.lambda_indices
            )
        
        node_names = ['g1', 'g2', 'g3']
        lambda_matrix = pd.DataFrame(
            lambda_matrix_arr, 
            index=node_names, 
            columns=node_names
            )

        node_i, node_j = (
                lambda_matrix.index[0],
                lambda_matrix.index[1],
            )
        
        svd2_vals = self.snj_solver.update_dissimilarity_map(
            lambda_matrix,
            (node_i, node_j),
            'new_node'
        )

        expected_lambda = np.zeros([2, 2])

        self.assertTrue(np.allclose(svd2_vals, expected_lambda, atol=0.1))

    def test_get_dissimilarity_map(self): #todo: move to distancesolver test?
        new_solver = cas.solver.SpectralNeighborJoiningSolver(add_root=False)

        lambda_matrix = new_solver.get_dissimilarity_map(self.tree_general, None)
        lambda_indices = new_solver.lambda_indices

        expected_lambda_matrix = np.array([
            [np.inf, 0.37210129, 0.3240793 , 0.13297122, 0.42458367],
            [0.37210129, np.inf, 0.55367472, 0.39066669, 0.40739702],
            [0.3240793 , 0.55367472, np.inf, 0.38248774, 0.24899081],
            [0.13297122, 0.39066669, 0.38248774, np.inf, 0.53773514],
            [0.42458367, 0.40739702, 0.24899081, 0.53773514, np.inf]
            ])

        expected_lambda_indices = [[0], [1], [2], [3], [4]]

        self.assertTrue(np.allclose(
            lambda_matrix, 
            expected_lambda_matrix, 
            atol=0.1
            ))

        self.assertTrue(np.allclose(
            lambda_indices, 
            expected_lambda_indices, 
            atol=0
            ))
    
    def test_find_cherry(self):
        fake_map1 = np.array([
            [1, 1, 1, 0],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [0, 1, 1, 1],
            ])

        fake_map2 = np.array([
            [1, 1, 1, 0],
            [1, 1, 1, 0],
            [1, 1, 1, 1],
            [0, 0, 1, 1],
            ])

        cherry1 = self.snj_solver.find_cherry(fake_map1)
        cherry2 = self.snj_solver.find_cherry(fake_map2)

        self.assertEqual(len(cherry1), 2)
        self.assertEqual(cherry1[0], 0)
        self.assertEqual(cherry1[1], 3)
        self.assertEqual(cherry2[0], 0)
        self.assertEqual(cherry2[1], 3)


    # def test_basic_solver(self):

    #     self.snj_solver.solve(self.tree_general)

    #     # test leaves exist in tree
    #     _leaves = self.tree_general.leaves

    #     self.assertEqual(
    #         len(_leaves), self.sim_general.shape[0] - 1
    #     )
    #     for _leaf in _leaves:
    #         self.assertIn(_leaf, self.sim_general.index.values)

    #     # test for expected number of edges
    #     edges = list(self.tree_general.edges)
    #     self.assertEqual(len(edges), 6)

    #     # test relationships between samples
    #     expected_tree = nx.DiGraph()
    #     expected_tree.add_edges_from(
    #         [
    #             ("5", "a"),
    #             ("5", "e"),
    #             ("6", "5"),
    #             ("b", "6"),
    #             ("6", "7"),
    #             ("7", "d"),
    #             ("7", "c"),
    #         ]
    #     )

    #     observed_tree = self.tree_general.get_tree_topology()
    #     triplets = itertools.combinations(["a", "c", "d", "e"], 3)
    #     for triplet in triplets:
    #         expected_triplet = find_triplet_structure(triplet, expected_tree)
    #         observed_triplet = find_triplet_structure(triplet, observed_tree)
    #         self.assertEqual(expected_triplet, observed_triplet)

    #     self.snj_solver.solve(self.tree_general, collapse_mutationless_edges=True)
    #     expected_tree = nx.DiGraph()
    #     expected_tree.add_nodes_from(["a", "b", "c", "d", "e", "5", "6", "7"])
    #     expected_tree.add_edges_from(
    #         [("6", "a"), ("6", "e"), ("b", "6"), ("6", "d"), ("6", "c")]
    #     )
    #     observed_tree = self.tree_general.get_tree_topology()
    #     triplets = itertools.combinations(["a", "c", "d", "e"], 3)
    #     for triplet in triplets:
    #         expected_triplet = find_triplet_structure(triplet, expected_tree)
    #         observed_triplet = find_triplet_structure(triplet, observed_tree)
    #         self.assertEqual(expected_triplet, observed_triplet)

    #     # compare tree distances
    #     observed_tree = observed_tree.to_undirected()
    #     expected_tree = expected_tree.to_undirected()
    #     for i in range(len(_leaves)):
    #         sample1 = _leaves[i]
    #         for j in range(i + 1, len(_leaves)):
    #             sample2 = _leaves[j]
    #             self.assertEqual(
    #                 nx.shortest_path_length(observed_tree, sample1, sample2),
    #                 nx.shortest_path_length(expected_tree, sample1, sample2),
    #             )

    # def test_snj_solver_weights(self):
    #     self.snj_solver.solve(self.pp_tree_priors)
    #     observed_tree = self.pp_tree_priors.get_tree_topology()

    #     expected_tree = nx.DiGraph()
    #     expected_tree.add_edges_from(
    #         [
    #             ("root", "7"),
    #             ("7", "6"),
    #             ("6", "d"),
    #             ("6", "e"),
    #             ("7", "8"),
    #             ("8", "a"),
    #             ("8", "9"),
    #             ("9", "b"),
    #             ("9", "c"),
    #         ]
    #     )

    #     triplets = itertools.combinations(["a", "b", "c", "d", "e"], 3)
    #     for triplet in triplets:
    #         expected_triplet = find_triplet_structure(triplet, expected_tree)
    #         observed_triplet = find_triplet_structure(triplet, observed_tree)
    #         self.assertEqual(expected_triplet, observed_triplet)

    #     self.snj_solver.solve(
    #         self.pp_tree_priors, collapse_mutationless_edges=True
    #     )
    #     observed_tree = self.pp_tree_priors.get_tree_topology()
    #     for triplet in triplets:
    #         expected_triplet = find_triplet_structure(triplet, expected_tree)
    #         observed_triplet = find_triplet_structure(triplet, observed_tree)
    #         self.assertEqual(expected_triplet, observed_triplet)

    # def test_pp_solver(self):

    #     self.snj_solver_delta.solve(self.pp_tree)
    #     observed_tree = self.pp_tree.get_tree_topology()

    #     expected_tree = nx.DiGraph()
    #     expected_tree.add_edges_from(
    #         [
    #             ("root", "9"),
    #             ("9", "8"),
    #             ("9", "7"),
    #             ("7", "6"),
    #             ("7", "a"),
    #             ("6", "b"),
    #             ("6", "c"),
    #             ("8", "e"),
    #             ("8", "d"),
    #         ]
    #     )

    #     triplets = itertools.combinations(["a", "b", "c", "d", "e"], 3)
    #     for triplet in triplets:
    #         expected_triplet = find_triplet_structure(triplet, expected_tree)
    #         observed_triplet = find_triplet_structure(triplet, observed_tree)
    #         self.assertEqual(expected_triplet, observed_triplet)

    #     self.snj_solver_delta.solve(
    #         self.pp_tree, collapse_mutationless_edges=True
    #     )
    #     observed_tree = self.pp_tree.get_tree_topology()
    #     for triplet in triplets:
    #         expected_triplet = find_triplet_structure(triplet, expected_tree)
    #         observed_triplet = find_triplet_structure(triplet, observed_tree)
    #         self.assertEqual(expected_triplet, observed_triplet)

    # def test_duplicate_sample_neighbor_joining(self):

    #     self.snj_solver_delta.solve(self.duplicate_tree)
    #     observed_tree = self.duplicate_tree.get_tree_topology()

    #     expected_tree = nx.DiGraph()
    #     expected_tree.add_edges_from(
    #         [
    #             ("root", "9"),
    #             ("9", "8"),
    #             ("9", "7"),
    #             ("7", "6"),
    #             ("7", "a"),
    #             ("6", "b"),
    #             ("6", "c"),
    #             ("8", "10"),
    #             ("10", "e"),
    #             ("10", "f"),
    #             ("8", "d"),
    #         ]
    #     )

    #     triplets = itertools.combinations(["a", "b", "c", "d", "e", "f"], 3)
    #     for triplet in triplets:
    #         expected_triplet = find_triplet_structure(triplet, expected_tree)
    #         observed_triplet = find_triplet_structure(triplet, observed_tree)
    #         self.assertEqual(expected_triplet, observed_triplet)

    def test_setup_root_finder_missing_dissimilarity_map(self):
        tree = cas.data.CassiopeiaTree(character_matrix=self.cm_general)

        self.snj_solver.setup_root_finder(tree)

        self.assertEqual(tree.root_sample_name, "root")

    # def test_setup_root_finder_existing_dissimilarity_map(self):
    #     tree = cas.data.CassiopeiaTree(
    #         character_matrix=self.cm,
    #         dissimilarity_map=self.sim_general,
    #     )
    #     with mock.patch.object(
    #         tree, "compute_dissimilarity_map"
    #     ) as compute_dissimilarity_map:
    #         self.snj_solver_delta.setup_root_finder(tree)
    #         compute_dissimilarity_map.assert_not_called()
    #     self.assertEqual(tree.root_sample_name, "root")
    #     dissimilarity_map = tree.get_dissimilarity_map()
    #     self.assertEqual(
    #         {"a", "b", "c", "d", "e", "root"}, set(dissimilarity_map.index)
    #     )
    #     self.assertEqual(
    #         {"a", "b", "c", "d", "e", "root"}, set(dissimilarity_map.columns)
    #     )
    #     for leaf in self.cm.index:
    #         delta = delta_fn(
    #             np.array([0] * tree.n_character),
    #             self.cm.loc[leaf].values,
    #             tree.missing_state_indicator,
    #             None,
    #         )
    #         self.assertEqual(dissimilarity_map.loc[leaf, "root"], delta)
    #         self.assertEqual(dissimilarity_map.loc["root", leaf], delta)
    #     self.assertEqual(dissimilarity_map.loc["root", "root"], 0)


if __name__ == "__main__":
    unittest.main()
