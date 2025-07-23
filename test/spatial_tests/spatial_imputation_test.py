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

    def test_anndata_to_graph_radius(self):
        """Tests the radius constructor of anndata to spatial graph."""

        spatial_graph = cas.sp.get_spatial_graph_from_anndata(
            self.spatial_adata, neighborhood_radius=10
        )

        node_map = dict(
            zip(
                range(
                    self.spatial_adata.obsp["spatial_connectivities"].shape[0]
                ),
                self.spatial_adata.obs_names,
            )
        )
        spatial_graph = nx.relabel_nodes(spatial_graph, node_map)

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

    def test_anndata_to_graph_size(self):
        """Tests the radius constructor of anndata to spatial graph."""

        spatial_graph = cas.sp.get_spatial_graph_from_anndata(
            self.spatial_adata, neighborhood_size=3
        )

        node_map = dict(
            zip(
                range(
                    self.spatial_adata.obsp["spatial_connectivities"].shape[0]
                ),
                self.spatial_adata.obs_names,
            )
        )
        spatial_graph = nx.relabel_nodes(spatial_graph, node_map)

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

        spatial_graph = cas.sp.get_spatial_graph_from_anndata(
            self.spatial_adata, neighborhood_size=3
        )
        node_map = dict(
            zip(
                range(
                    self.spatial_adata.obsp["spatial_connectivities"].shape[0]
                ),
                self.spatial_adata.obs_names,
            )
        )
        spatial_graph = nx.relabel_nodes(spatial_graph, node_map)

        coordinates = pd.DataFrame(self.spatial_adata.obsm['spatial'], index=self.spatial_adata.obs_names)

        number_of_hops = 1
        max_neighborhood_distance = np.inf

        imputed_state, frequency, count = spatial_utilities.impute_single_state(
            cell,
            character,
            self.character_matrix,
            spatial_graph,
            number_of_hops,
            max_neighborhood_distance,
        )

        self.assertEqual(imputed_state, 1)
        self.assertEqual(frequency, 3 / 4)
        self.assertEqual(3, count)

    def test_impute_single_state_max_neighborhood_size(self):

        cell = "cell_5"
        character = 0

        spatial_graph = cas.sp.get_spatial_graph_from_anndata(
            self.spatial_adata, neighborhood_size=3
        )
        node_map = dict(
            zip(
                range(
                    self.spatial_adata.obsp["spatial_connectivities"].shape[0]
                ),
                self.spatial_adata.obs_names,
            )
        )
        spatial_graph = nx.relabel_nodes(spatial_graph, node_map)
        coordinates = pd.DataFrame(self.spatial_adata.obsm['spatial'], index=self.spatial_adata.obs_names)

        number_of_hops = 1

        imputed_state, frequency, count = spatial_utilities.impute_single_state(
            cell,
            character,
            self.character_matrix,
            spatial_graph,
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
            spatial_graph,
            number_of_hops,
            max_neighbor_distance=15,
            coordinates=coordinates
        )

        self.assertEqual(imputed_state, 1)
        self.assertEqual(frequency, 1.0)
        self.assertEqual(2, count)


if __name__ == "__main__":
    unittest.main()
