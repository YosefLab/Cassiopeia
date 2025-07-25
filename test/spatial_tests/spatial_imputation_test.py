"""
Test the spatial imputation workflow.
"""

import os

import unittest
from unittest import mock
from typing import Dict, Optional

import anndata
import networkx as nx
import numpy as np
import pandas as pd

import cassiopeia as cas
from cassiopeia.spatial import spatial_utilities
from cassiopeia.mixins import try_import

sq = try_import("squidpy")
SQUIDPY_INSTALLED = (sq is not None)


class TestSpatialImputation(unittest.TestCase):
    def setUp(self):

        coordinates = np.array(
            [
                [0.0, 0.0],
                [0.0, 10],
                [0.0, 50],
                [10, 0.0],
                [10, 50],
                [15, 50],
                [90, 100],
                [100, 100],
                [100, 90],
            ]
        )

        adata = anndata.AnnData(
            obs=pd.DataFrame(
                index=[f"cell_{x}" for x in range(len(coordinates))]
            )
        )
        adata.obsm["spatial"] = coordinates

        self.spatial_adata = adata
        self.spatial_graph_neigh3 = nx.Graph()
        for edge in [
            ("cell_0", "cell_1"),
            ("cell_0", "cell_2"),
            ("cell_0", "cell_3"),
            ("cell_1", "cell_2"),
            ("cell_1", "cell_3"),
            ("cell_1", "cell_4"),
            ("cell_1", "cell_5"),
            ("cell_2", "cell_4"),
            ("cell_2", "cell_5"),
            ("cell_3", "cell_4"),
            ("cell_4", "cell_5"),
            ("cell_5", "cell_6"),
            ("cell_5", "cell_7"),
            ("cell_5", "cell_8"),
            ("cell_6", "cell_7"),
            ("cell_6", "cell_8"),
            ("cell_7", "cell_8"),
        ]:
            self.spatial_graph_neigh3.add_edge(edge[0], edge[1])


        self.character_matrix = pd.DataFrame.from_dict(
            {
                "cell_0": [1, 1, 0, 1],
                "cell_1": [2, 1, 0, 2],
                "cell_2": [1, 1, 2, 3],
                "cell_3": [1, 1, 2, 4],
                "cell_4": [1, 2, 3, 5],
                "cell_5": [1, 2, 3, 6],
                "cell_6": [2, 0, 5, 8],
                "cell_7": [2, 0, 5, 9],
                "cell_8": [2, 0, 5, 10],
            },
            orient="index",
        )

        self.character_matrix_missing = pd.DataFrame.from_dict(
            {
                "cell_0": [1, 1, 0, 1],
                "cell_1": [-1, 1, 0, 2],
                "cell_2": [1, 1, 2, 3],
                "cell_3": [1, -1, 2, 4],
                "cell_4": [1, 2, 3, 5],
                "cell_5": [1, 2, 3, 6],
                "cell_6": [2, 0, 5, 8],
                "cell_7": [2, 0, -1, 9],
                "cell_8": [2, 0, 5, 10],
            },
            orient="index",
        )

    @unittest.skipUnless(SQUIDPY_INSTALLED, "Squidpy not installed.")
    def test_anndata_to_graph_radius(self):
        """Tests the radius constructor of anndata to spatial graph."""

        spatial_graph = cas.sp.get_spatial_graph_from_anndata(
            self.spatial_adata, neighborhood_radius=10
        )

        expected_graph = nx.Graph()
        for edge in [
            ("cell_0", "cell_1"),
            ("cell_0", "cell_3"),
            ("cell_2", "cell_4"),
            ("cell_4", "cell_5"),
            ("cell_6", "cell_7"),
            ("cell_7", "cell_8"),
        ]:
            expected_graph.add_edge(edge[0], edge[1])

        expected_nodes = set(expected_graph.nodes)
        expected_edges = set(expected_graph.edges)

        self.assertEqual(set(spatial_graph.nodes), expected_nodes)
        self.assertEqual(set(spatial_graph.edges), expected_edges)

    @unittest.skipUnless(SQUIDPY_INSTALLED, "Squidpy not installed.")
    def test_anndata_to_graph_size(self):
        """Tests the radius constructor of anndata to spatial graph."""

        spatial_graph = cas.sp.get_spatial_graph_from_anndata(
            self.spatial_adata, neighborhood_size=3
        )

        expected_graph = nx.Graph()
        for edge in [
            ("cell_0", "cell_1"),
            ("cell_0", "cell_2"),
            ("cell_0", "cell_3"),
            ("cell_1", "cell_2"),
            ("cell_1", "cell_3"),
            ("cell_1", "cell_4"),
            ("cell_1", "cell_5"),
            ("cell_2", "cell_4"),
            ("cell_2", "cell_5"),
            ("cell_3", "cell_4"),
            ("cell_4", "cell_5"),
            ("cell_5", "cell_6"),
            ("cell_5", "cell_7"),
            ("cell_5", "cell_8"),
            ("cell_6", "cell_7"),
            ("cell_6", "cell_8"),
            ("cell_7", "cell_8"),
        ]:
            expected_graph.add_edge(edge[0], edge[1])

        expected_nodes = set(expected_graph.nodes)
        expected_edges = set(expected_graph.edges)

        self.assertEqual(set(spatial_graph.nodes), expected_nodes)
        self.assertEqual(set(spatial_graph.edges), expected_edges)
    
    def test_impute_single_state_basic(self):

        cell = "cell_4"
        character = 0

        coordinates = pd.DataFrame(
            self.spatial_adata.obsm["spatial"],
            index=self.spatial_adata.obs_names,
        )

        number_of_hops = 1
        max_neighborhood_distance = np.inf

        imputed_state, frequency, count = spatial_utilities.impute_single_state(
            cell,
            character,
            self.character_matrix,
            self.spatial_graph_neigh3,
            number_of_hops,
            max_neighborhood_distance,
        )

        self.assertEqual(imputed_state, 1)
        self.assertEqual(frequency, 3 / 4)
        self.assertEqual(3, count)

    def test_impute_single_state_max_neighborhood_size(self):

        cell = "cell_5"
        character = 0

        coordinates = pd.DataFrame(
            self.spatial_adata.obsm["spatial"],
            index=self.spatial_adata.obs_names,
        )

        number_of_hops = 1

        imputed_state, frequency, count = spatial_utilities.impute_single_state(
            cell,
            character,
            self.character_matrix,
            self.spatial_graph_neigh3,
            number_of_hops,
            max_neighbor_distance=np.inf,
        )

        self.assertEqual(imputed_state, 2)
        self.assertEqual(frequency, 2 / 3)
        self.assertEqual(4, count)

        # now set maximum distance
        imputed_state, frequency, count = spatial_utilities.impute_single_state(
            cell,
            character,
            self.character_matrix,
            self.spatial_graph_neigh3,
            number_of_hops,
            max_neighbor_distance=15,
            coordinates=coordinates,
        )

        self.assertEqual(imputed_state, 1)
        self.assertEqual(frequency, 1.0)
        self.assertEqual(2, count)

    def test_spatial_imputation_integration_simple_one_hop(self):
        
        imputed_character_matrix = cas.sp.impute_alleles_from_spatial_data(
            self.character_matrix_missing,
            self.spatial_adata,
            imputation_hops=1,
            imputation_concordance=0.0,
            num_imputation_iterations=1,
            spatial_graph=self.spatial_graph_neigh3,
        )

        self.assertEqual(imputed_character_matrix.loc["cell_1", 0], 1)
        self.assertEqual(imputed_character_matrix.loc["cell_3", 1], 1)
        self.assertEqual(imputed_character_matrix.loc["cell_7", 2], 5)

    def test_spatial_imputation_integration_simple_min_concordance(self):

        imputed_character_matrix = cas.sp.impute_alleles_from_spatial_data(
            self.character_matrix_missing,
            self.spatial_adata,
            imputation_hops=1,
            imputation_concordance=0.8,
            num_imputation_iterations=1,
            spatial_graph=self.spatial_graph_neigh3,
        )

        self.assertEqual(imputed_character_matrix.loc["cell_1", 0], 1)
        self.assertEqual(imputed_character_matrix.loc["cell_3", 1], -1)
        self.assertEqual(imputed_character_matrix.loc["cell_7", 2], -1)

    @unittest.skipUnless(SQUIDPY_INSTALLED, "Squidpy not installed.")
    def test_spatial_imputation_integration_simple_neighborhood_radius(self):

        character_matrix_missing2 = self.character_matrix_missing.copy()
        character_matrix_missing2.loc["cell_8", 2] = 11

        imputed_character_matrix = cas.sp.impute_alleles_from_spatial_data(
            character_matrix_missing2,
            self.spatial_adata,
            imputation_hops=1,
            imputation_concordance=0.6,
            num_imputation_iterations=1,
            neighborhood_radius=15.0,
        )

        self.assertEqual(imputed_character_matrix.loc["cell_1", 0], 1)
        self.assertEqual(imputed_character_matrix.loc["cell_3", 1], 1)
        self.assertEqual(imputed_character_matrix.loc["cell_7", 2], -1)

    @unittest.skipUnless(SQUIDPY_INSTALLED, "Squidpy not installed.")
    def test_spatial_imputation_integration_size_over_radius(self):

        character_matrix_missing2 = self.character_matrix_missing.copy()
        character_matrix_missing2.loc["cell_5", 1] = -1
        character_matrix_missing2.loc["cell_6", 1] = 1

        imputed_character_matrix = cas.sp.impute_alleles_from_spatial_data(
            character_matrix_missing2,
            self.spatial_adata,
            imputation_hops=1,
            imputation_concordance=0.0,
            num_imputation_iterations=1,
            neighborhood_radius=10.0,
            neighborhood_size=3,
        )

        self.assertEqual(imputed_character_matrix.loc["cell_5", 1], 1)

    def test_spatial_imputation_integration_two_hops(self):

        character_matrix_missing2 = self.character_matrix_missing.copy()
        character_matrix_missing2.loc["cell_2", 1] = 2
        character_matrix_missing2.loc["cell_5", 2] = 5

        imputed_character_matrix = cas.sp.impute_alleles_from_spatial_data(
            character_matrix_missing2,
            self.spatial_adata,
            imputation_hops=2,
            imputation_concordance=0.0,
            num_imputation_iterations=1,
            spatial_graph=self.spatial_graph_neigh3,
        )

        self.assertEqual(imputed_character_matrix.loc["cell_1", 0], 1)
        self.assertEqual(imputed_character_matrix.loc["cell_3", 1], 2)
        self.assertEqual(imputed_character_matrix.loc["cell_7", 2], 5)

    def test_spatial_imputation_integration_no_zero(self):

        character_matrix_missing2 = self.character_matrix_missing.copy()
        character_matrix_missing2.loc["cell_0", 1] = 0
        character_matrix_missing2.loc["cell_4", 1] = 0

        imputed_character_matrix = cas.sp.impute_alleles_from_spatial_data(
            character_matrix_missing2,
            self.spatial_adata,
            imputation_hops=1,
            imputation_concordance=0.0,
            num_imputation_iterations=1,
            spatial_graph=self.spatial_graph_neigh3,
        )

        self.assertEqual(imputed_character_matrix.loc["cell_3", 1], -1)

    def test_spatial_imputation_integration_two_iterations(self):

        
        spatial_graph=self.spatial_graph_neigh3
        spatial_graph.add_edge("cell_7", "cell_9")  # add new edge

        character_matrix_missing2 = self.character_matrix_missing.copy()
        character_matrix_missing2.loc["cell_9"] = [-1, -1, -1, -1]

        imputed_character_matrix = cas.sp.impute_alleles_from_spatial_data(
            character_matrix_missing2,
            spatial_graph=spatial_graph,
            imputation_hops=1,
            imputation_concordance=0.0,
            num_imputation_iterations=1,
        )

        # make sure that after one iteration no value is imputed in cell_9
        self.assertEqual(imputed_character_matrix.loc["cell_9", 2], -1)

        imputed_character_matrix = cas.sp.impute_alleles_from_spatial_data(
            character_matrix_missing2,
            spatial_graph=spatial_graph,
            imputation_hops=1,
            imputation_concordance=0.0,
            num_imputation_iterations=2,
        )

        # after 2 iterations, there should be an imputation from cell_7
        self.assertEqual(imputed_character_matrix.loc["cell_9", 2], 5)

    def test_spatial_imputation_integration_max_neighbor_distance(self):

        character_matrix_missing2 = self.character_matrix_missing.copy()
        character_matrix_missing2.loc["cell_5", 0] = -1

        coordinates = pd.DataFrame(
            self.spatial_adata.obsm["spatial"],
            index=self.spatial_adata.obs_names,
        )

        imputed_character_matrix = cas.sp.impute_alleles_from_spatial_data(
            character_matrix_missing2,
            adata=self.spatial_adata,
            spatial_graph=self.spatial_graph_neigh3,
            imputation_hops=1,
            imputation_concordance=0.0,
            num_imputation_iterations=1,
            max_neighbor_distance=15,
            coordinates=coordinates,
        )

        self.assertEqual(imputed_character_matrix.loc["cell_5", 0], 1)

        # now without neighbor max distance
        imputed_character_matrix = cas.sp.impute_alleles_from_spatial_data(
            character_matrix_missing2,
            adata=self.spatial_adata,
            imputation_hops=1,
            imputation_concordance=0.0,
            num_imputation_iterations=1,
            spatial_graph=self.spatial_graph_neigh3,
            max_neighbor_distance=np.inf,
            coordinates=coordinates,
        )

        self.assertEqual(imputed_character_matrix.loc["cell_5", 0], 2)


if __name__ == "__main__":
    unittest.main()
