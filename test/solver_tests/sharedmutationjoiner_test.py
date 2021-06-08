"""
Test SharedMutationJoiningSolver in Cassiopeia.solver.
"""
import os
import unittest
import sys
from functools import partial
from typing import Dict, Optional
from unittest import mock

import itertools
import networkx as nx
import numba
import numpy as np
import pandas as pd
import scipy

from cassiopeia.data.CassiopeiaTree import CassiopeiaTree
from cassiopeia.data import utilities as data_utilities
from cassiopeia.solver.SharedMutationJoiningSolver import (
    SharedMutationJoiningSolver,
    SharedMutationJoiningSolverWarning,
)
from cassiopeia.solver import dissimilarity_functions
from cassiopeia.solver import solver_utilities


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


class TestSharedMutationJoiningSolver(unittest.TestCase):
    def setUp(self):

        # --------------------- General NJ ---------------------
        cm = pd.DataFrame.from_dict(
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

        delta = pd.DataFrame.from_dict(
            {
                "a": [0, 2, 1, 1, 0],
                "b": [2, 0, 1, 2, 0],
                "c": [1, 1, 0, 0, 0],
                "d": [1, 2, 0, 0, 0],
                "e": [0, 0, 0, 0, 0],
            },
            orient="index",
            columns=["a", "b", "c", "d", "e"],
        )

        self.basic_similarity_map = delta
        self.basic_tree = CassiopeiaTree(
            character_matrix=cm, dissimilarity_map=delta
        )

        self.smj_solver = SharedMutationJoiningSolver(
            similarity_function=dissimilarity_functions.hamming_similarity_without_missing
        )
        self.smj_solver_no_numba = SharedMutationJoiningSolver(
            similarity_function=partial(
                dissimilarity_functions.cluster_dissimilarity,
                dissimilarity_functions.hamming_similarity_without_missing,
            )
        )

        # ---------------- Lineage Tracing NJ ----------------

        pp_cm = pd.DataFrame.from_dict(
            {
                "a": [1, 2, 2],
                "b": [1, 2, 1],
                "c": [1, 2, 0],
                "d": [2, 0, 0],
                "e": [2, 0, 2],
            },
            orient="index",
            columns=["x1", "x2", "x3"],
        )

        self.pp_tree = CassiopeiaTree(character_matrix=pp_cm)

        self.smj_solver_pp = SharedMutationJoiningSolver(
            similarity_function=dissimilarity_functions.hamming_similarity_without_missing
        )

        # ------------- CM with Duplicates and Missing Data -----------------------
        duplicates_cm = pd.DataFrame.from_dict(
            {
                "a": [1, -1, 0],
                "b": [2, -1, 2],
                "c": [2, 0, 2],
                "d": [2, 0, -1],
                "e": [2, 0, 2],
                "f": [2, -1, 2],
            },
            orient="index",
            columns=["x1", "x2", "x3"],
        )

        self.duplicate_tree = CassiopeiaTree(character_matrix=duplicates_cm)

        # ------------- Hamming similarity with weights ------------
        priors = {0: {1: 0.5, 2: 0.5}, 1: {1: 0.2, 2: 0.8}, 2: {1: 0.9, 2: 0.1}}
        self.pp_tree_priors = CassiopeiaTree(
            character_matrix=pp_cm, priors=priors
        )
        self.smj_solver_modified_pp = SharedMutationJoiningSolver(
            similarity_function=dissimilarity_functions.hamming_similarity_without_missing
        )

    def test_init(self):
        # This should numbaize
        solver = SharedMutationJoiningSolver(
            similarity_function=dissimilarity_functions.hamming_similarity_without_missing
        )
        self.assertTrue(
            isinstance(
                solver.similarity_function, numba.core.registry.CPUDispatcher
            )
        )
        self.assertTrue(
            isinstance(
                solver._SharedMutationJoiningSolver__update_similarity_map,
                numba.core.registry.CPUDispatcher,
            )
        )

        # This shouldn't numbaize
        with self.assertWarns(SharedMutationJoiningSolverWarning):
            solver = SharedMutationJoiningSolver(
                similarity_function=partial(
                    dissimilarity_functions.cluster_dissimilarity,
                    dissimilarity_functions.hamming_similarity_without_missing,
                )
            )
            self.assertFalse(
                isinstance(
                    solver.similarity_function,
                    numba.core.registry.CPUDispatcher,
                )
            )
            self.assertFalse(
                isinstance(
                    solver._SharedMutationJoiningSolver__update_similarity_map,
                    numba.core.registry.CPUDispatcher,
                )
            )

    def test_find_cherry(self):
        cherry = self.smj_solver.find_cherry(self.basic_similarity_map.values)
        delta = self.basic_similarity_map
        node_i, node_j = (delta.index[cherry[0]], delta.index[cherry[1]])

        self.assertIn((node_i, node_j), [("a", "b"), ("b", "a")])

    def test_create_similarity_map(self):
        character_matrix = self.pp_tree_priors.get_current_character_matrix()
        weights = solver_utilities.transform_priors(
            self.pp_tree_priors.priors, "negative_log"
        )

        similarity_map = data_utilities.compute_dissimilarity_map(
            character_matrix.to_numpy(),
            character_matrix.shape[0],
            dissimilarity_functions.hamming_similarity_without_missing,
            weights,
            self.pp_tree_priors.missing_state_indicator,
        )

        similarity_map = scipy.spatial.distance.squareform(similarity_map)

        similarity_map = pd.DataFrame(
            similarity_map,
            index=character_matrix.index,
            columns=character_matrix.index,
        )

        expected_similarity = -np.log(0.5) - np.log(0.8)
        self.assertEqual(similarity_map.loc["a", "b"], expected_similarity)
        expected_similarity = -np.log(0.1)
        self.assertEqual(similarity_map.loc["a", "e"], expected_similarity)

    def test_update_similarity_map_and_character_matrix(self):

        nb_similarity = numba.jit(
            dissimilarity_functions.hamming_similarity_without_missing,
            nopython=True,
        )
        nb_weights = numba.typed.Dict.empty(
            numba.types.int64,
            numba.types.DictType(numba.types.int64, numba.types.float64),
        )

        cm = self.basic_tree.get_current_character_matrix()
        delta = self.basic_similarity_map

        cherry = self.smj_solver.find_cherry(delta.values)
        node_i, node_j = (delta.index[cherry[0]], delta.index[cherry[1]])

        delta = self.smj_solver.update_similarity_map_and_character_matrix(
            cm, nb_similarity, delta, (node_i, node_j), "ab", weights=nb_weights
        )

        expected_delta = pd.DataFrame.from_dict(
            {
                "ab": [0, 1, 1, 0],
                "c": [1, 0, 0, 0],
                "d": [1, 0, 0, 0],
                "e": [0, 0, 0, 0],
            },
            orient="index",
            columns=["ab", "c", "d", "e"],
        )

        for sample in expected_delta.index:
            for sample2 in expected_delta.index:
                self.assertEqual(
                    delta.loc[sample, sample2],
                    expected_delta.loc[sample, sample2],
                )

        cherry = self.smj_solver.find_cherry(delta.values)
        node_i, node_j = (delta.index[cherry[0]], delta.index[cherry[1]])

        delta = self.smj_solver.update_similarity_map_and_character_matrix(
            cm,
            nb_similarity,
            delta,
            (node_i, node_j),
            "abc",
            weights=nb_weights,
        )

        expected_delta = pd.DataFrame.from_dict(
            {"abc": [0, 0, 0], "d": [0, 0, 0], "e": [0, 0, 0]},
            orient="index",
            columns=["abc", "d", "e"],
        )

        for sample in expected_delta.index:
            for sample2 in expected_delta.index:
                self.assertEqual(
                    delta.loc[sample, sample2],
                    expected_delta.loc[sample, sample2],
                )

        expected_cm = pd.DataFrame.from_dict(
            {"abc": [0, 0, 2], "d": [1, 1, 1], "e": [0, 0, 0]},
            orient="index",
            columns=["x1", "x2", "x3"],
        )

        for sample in expected_cm.index:
            for col in expected_cm.columns:
                self.assertEqual(
                    cm.loc[sample, col], expected_cm.loc[sample, col]
                )

    def test_basic_solver(self):

        self.smj_solver.solve(self.basic_tree)

        # test that the dissimilarity map and character matrix were not altered
        cm = pd.DataFrame.from_dict(
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
        for i in self.basic_similarity_map.index:
            for j in self.basic_similarity_map.columns:
                self.assertEqual(
                    self.basic_similarity_map.loc[i, j],
                    self.basic_tree.get_dissimilarity_map().loc[i, j],
                )
        for i in self.basic_tree.get_current_character_matrix().index:
            for j in self.basic_tree.get_current_character_matrix().columns:
                self.assertEqual(
                    cm.loc[i, j],
                    self.basic_tree.get_current_character_matrix().loc[i, j],
                )

        # test leaves exist in tree
        _leaves = self.basic_tree.leaves

        self.assertEqual(len(_leaves), self.basic_similarity_map.shape[0])
        for _leaf in _leaves:
            self.assertIn(_leaf, self.basic_similarity_map.index.values)

        # test for expected number of edges
        edges = list(self.basic_tree.edges)
        self.assertEqual(len(edges), 8)

        # test relationships between samples
        expected_tree = nx.DiGraph()
        expected_tree.add_nodes_from(
            ["a", "b", "c", "d", "e", "5", "6", "7", "8"]
        )
        expected_tree.add_edges_from(
            [
                ("5", "a"),
                ("5", "b"),
                ("6", "5"),
                ("6", "c"),
                ("7", "d"),
                ("7", "e"),
                ("8", "6"),
                ("8", "7"),
            ]
        )

        T = self.basic_tree.get_tree_topology()
        triplets = itertools.combinations(["a", "b", "c", "d", "e"], 3)
        for triplet in triplets:

            expected_triplet = find_triplet_structure(triplet, expected_tree)
            observed_triplet = find_triplet_structure(triplet, T)
            self.assertEqual(expected_triplet, observed_triplet)

        # compare tree distances
        T = T.to_undirected()
        expected_tree = expected_tree.to_undirected()
        for i in range(len(_leaves)):
            sample1 = _leaves[i]
            for j in range(i + 1, len(_leaves)):
                sample2 = _leaves[j]
                self.assertEqual(
                    nx.shortest_path_length(T, sample1, sample2),
                    nx.shortest_path_length(expected_tree, sample1, sample2),
                )

    def test_solver_no_numba(self):

        self.smj_solver_no_numba.solve(self.basic_tree)

        # test that the dissimilarity map and character matrix were not altered
        cm = pd.DataFrame.from_dict(
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
        for i in self.basic_similarity_map.index:
            for j in self.basic_similarity_map.columns:
                self.assertEqual(
                    self.basic_similarity_map.loc[i, j],
                    self.basic_tree.get_dissimilarity_map().loc[i, j],
                )
        for i in self.basic_tree.get_current_character_matrix().index:
            for j in self.basic_tree.get_current_character_matrix().columns:
                self.assertEqual(
                    cm.loc[i, j],
                    self.basic_tree.get_current_character_matrix().loc[i, j],
                )

        # test leaves exist in tree
        _leaves = self.basic_tree.leaves

        self.assertEqual(len(_leaves), self.basic_similarity_map.shape[0])
        for _leaf in _leaves:
            self.assertIn(_leaf, self.basic_similarity_map.index.values)

        # test for expected number of edges
        edges = list(self.basic_tree.edges)
        self.assertEqual(len(edges), 8)

        # test relationships between samples
        expected_tree = nx.DiGraph()
        expected_tree.add_nodes_from(
            ["a", "b", "c", "d", "e", "5", "6", "7", "8"]
        )
        expected_tree.add_edges_from(
            [
                ("5", "a"),
                ("5", "b"),
                ("6", "5"),
                ("6", "c"),
                ("7", "d"),
                ("7", "e"),
                ("8", "6"),
                ("8", "7"),
            ]
        )

        T = self.basic_tree.get_tree_topology()
        triplets = itertools.combinations(["a", "b", "c", "d", "e"], 3)
        for triplet in triplets:

            expected_triplet = find_triplet_structure(triplet, expected_tree)
            observed_triplet = find_triplet_structure(triplet, T)
            self.assertEqual(expected_triplet, observed_triplet)

        # compare tree distances
        T = T.to_undirected()
        expected_tree = expected_tree.to_undirected()
        for i in range(len(_leaves)):
            sample1 = _leaves[i]
            for j in range(i + 1, len(_leaves)):
                sample2 = _leaves[j]
                self.assertEqual(
                    nx.shortest_path_length(T, sample1, sample2),
                    nx.shortest_path_length(expected_tree, sample1, sample2),
                )

    def test_smj_solver_weights(self):

        self.smj_solver_modified_pp.solve(self.pp_tree_priors)
        T = self.pp_tree_priors.get_tree_topology()

        expected_tree = nx.DiGraph()
        expected_tree.add_nodes_from(
            ["a", "b", "c", "d", "e", "5", "6", "7", "8"]
        )
        expected_tree.add_edges_from(
            [
                ("5", "a"),
                ("5", "e"),
                ("6", "b"),
                ("6", "c"),
                ("7", "5"),
                ("7", "d"),
                ("8", "6"),
                ("8", "7"),
            ]
        )

        triplets = itertools.combinations(["a", "b", "c", "d", "e"], 3)
        for triplet in triplets:
            expected_triplet = find_triplet_structure(triplet, expected_tree)
            observed_triplet = find_triplet_structure(triplet, T)
            self.assertEqual(expected_triplet, observed_triplet)

    def test_pp_solver(self):

        self.smj_solver_pp.solve(self.pp_tree)
        T = self.pp_tree.get_tree_topology()

        pp_cm = pd.DataFrame.from_dict(
            {
                "a": [1, 2, 2],
                "b": [1, 2, 1],
                "c": [1, 2, 0],
                "d": [2, 0, 0],
                "e": [2, 0, 2],
            },
            orient="index",
            columns=["x1", "x2", "x3"],
        )
        self.assertIsNone(self.pp_tree.get_dissimilarity_map())
        for i in self.pp_tree.get_current_character_matrix().index:
            for j in self.pp_tree.get_current_character_matrix().columns:
                self.assertEqual(
                    pp_cm.loc[i, j],
                    self.pp_tree.get_current_character_matrix().loc[i, j],
                )

        expected_tree = nx.DiGraph()
        expected_tree.add_nodes_from(
            ["a", "b", "c", "d", "e", "5", "6", "7", "8"]
        )
        expected_tree.add_edges_from(
            [
                ("5", "a"),
                ("5", "b"),
                ("6", "5"),
                ("6", "c"),
                ("7", "d"),
                ("7", "e"),
                ("8", "6"),
                ("8", "7"),
            ]
        )

        triplets = itertools.combinations(["a", "b", "c", "d", "e"], 3)
        for triplet in triplets:
            expected_triplet = find_triplet_structure(triplet, expected_tree)
            observed_triplet = find_triplet_structure(triplet, T)
            self.assertEqual(expected_triplet, observed_triplet)

    def test_duplicate(self):
        # In this case, we see that the missing data can break up a duplicate
        # pair if the behavior is to ignore missing data

        self.smj_solver_pp.solve(self.duplicate_tree)
        T = self.duplicate_tree.get_tree_topology()

        expected_tree = nx.DiGraph()
        expected_tree.add_nodes_from(
            ["a", "b", "c", "d", "e", "f", "5", "6", "7", "8", "9"]
        )
        expected_tree.add_edges_from(
            [
                ("5", "b"),
                ("5", "c"),
                ("6", "e"),
                ("6", "f"),
                ("7", "5"),
                ("7", "6"),
                ("8", "7"),
                ("8", "d"),
                ("9", "8"),
                ("9", "a"),
            ]
        )
        triplets = itertools.combinations(["a", "b", "c", "d", "e", "f"], 3)
        for triplet in triplets:
            expected_triplet = find_triplet_structure(triplet, expected_tree)
            observed_triplet = find_triplet_structure(triplet, T)
            self.assertEqual(expected_triplet, observed_triplet)


if __name__ == "__main__":
    unittest.main()
