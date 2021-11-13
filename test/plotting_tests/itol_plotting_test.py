"""
Tests for the itol plotting utilities in `cassiopeia.plotting.itol_utilities`.

Currently, this testing module does not test the actual uploading / exporting
of trees due to difficulties in mock authenticating on the iTOL side.
"""
import os
import unittest

import shutil
import tempfile

import networkx as nx
import numpy as np
import pandas as pd

import cassiopeia as cas
from cassiopeia.plotting import itol_utilities

ITOL_FILE = os.path.join(os.path.expanduser("~/.itolconfig"))


class TestITOLPlotting(unittest.TestCase):
    def setUp(self):

        # set up temporary directory
        self.temporary_directory = tempfile.mkdtemp()

        # random state for reproducibility
        self.random_state = np.random.RandomState(123412334)

        # indel frequency dictionary
        self.indel_priors = pd.DataFrame.from_dict(
            {"i": 0.8, "j": 0.1, "k": 0.01, "m": 0.5, "n": 0.5},
            orient="index",
            columns=["freq"],
        )

        # test allele table
        self.allele_table = pd.DataFrame.from_dict(
            {
                1: ["2", "A", 10, "i", "j", "k"],
                2: ["2", "B", 10, "j", "j", "j"],
                3: ["3", "A", 10, "i", "m", "n"],
                4: ["5", "A", 10, "i", "j", "k"],
                5: ["5", "B", 10, "none", "none", "none"],
                6: ["6", "A", 10, "i", "j", "m"],
            },
            orient="index",
            columns=["cellBC", "intBC", "UMI", "r1", "r2", "r3"],
        )

        ## create CassiopeiaTree and associated data

        # graph
        graph = nx.DiGraph()
        graph.add_edges_from(
            [
                ("root", "1"),
                ("1", "2"),
                ("1", "3"),
                ("root", "4"),
                ("4", "5"),
                ("4", "6"),
            ]
        )

        # cell meta
        cell_meta = pd.DataFrame(index=["2", "3", "5", "6"])
        cell_meta["nUMI"] = [1, 3, 5, 7]
        cell_meta["cluster"] = ["a", "a", "b", "b"]

        self.tree = cas.data.CassiopeiaTree(tree=graph, cell_meta=cell_meta)

    def test_generate_gradient_file(self):

        _file = itol_utilities.create_gradient_from_df(
            self.tree.cell_meta["nUMI"],
            self.tree,
            "test_gradient",
            self.temporary_directory,
        )

        self.assertTrue(os.path.exists(_file))

        expected_content = (
            "DATASET_GRADIENT\nSEPARATOR TAB\nCOLOR\t#00000"
            "\nCOLOR_MIN\t#ffffff\nCOLOR_MAX\t#000000\nMARGIN\t100"
            "\nDATASET_LABEL\tnUMI\nSTRIP_WIDTH\t100\nSHOW_INTERNAL\t0"
            "\nDATA\n\n2\t1\n3\t3\n5\t5\n6\t7\n"
        )

        with open(_file, "r") as f:
            content = f.read()
            self.assertEqual(expected_content, content)

    def test_generate_colorbar_file(self):

        color_map = {"a": (255, 0, 0), "b": (0, 255, 255)}
        _file = itol_utilities.create_colorbar(
            self.tree.cell_meta["cluster"],
            self.tree,
            color_map,
            "test_colorbar",
            self.temporary_directory,
            create_legend=False,
        )

        self.assertTrue(os.path.exists(_file))

        expected_content = (
            "DATASET_COLORSTRIP\nSEPARATOR TAB\nCOLOR\t#FF0000"
            "\nMARGIN\t100\nDATASET_LABEL\ttest_colorbar\nSTRIP_WIDTH\t100"
            "\nSHOW_INTERNAL\t0\n\n\nDATA\n2\trgb(255,0,0)\n3\trgb(255,0,0)"
            "\n5\trgb(0,255,255)\n6\trgb(0,255,255)\n"
        )

        with open(_file, "r") as f:
            content = f.read()
            self.assertEqual(expected_content, content)

    def test_generate_colorbar_file_with_legend(self):

        color_map = {"a": (255, 0, 0), "b": (0, 255, 255)}
        _file = itol_utilities.create_colorbar(
            self.tree.cell_meta["cluster"],
            self.tree,
            color_map,
            "test_colorbar",
            self.temporary_directory,
            create_legend=True,
        )

        self.assertTrue(os.path.exists(_file))

        expected_content = (
            "DATASET_COLORSTRIP\nSEPARATOR TAB\nCOLOR\t#FF0000"
            "\nMARGIN\t100\nDATASET_LABEL\ttest_colorbar\nSTRIP_WIDTH\t100"
            "\nSHOW_INTERNAL\t0\n\nLEGEND_TITLE\ttest_colorbar legend\n"
            "LEGEND_SHAPES\t1\t1\nLEGEND_COLORS\t#ff0000\t#00ffff"
            "\nLEGEND_LABELS\ta\tb\n\nDATA\n2\trgb(255,0,0)\n3\trgb(255,0,0)"
            "\n5\trgb(0,255,255)\n6\trgb(0,255,255)\n"
        )

        with open(_file, "r") as f:
            content = f.read()
            self.assertEqual(expected_content, content)

    def test_create_allele_heatmap_specified_colors(self):

        indel_colors = {
            "i": [0.75, 0, 0.5],
            "j": [0.05, 0.88, 0.94],
            "k": [0.69, 1.0, 1.0],
            "m": [0.798, 0.37, 0.68],
            "n": [0.56, 0.37, 0.68],
        }

        indel_color_df = pd.DataFrame(columns=["color"])
        for indel in indel_colors:
            indel_color_df.loc[indel, "color"] = indel_colors[indel]

        allele_files = itol_utilities.create_indel_heatmap(
            self.allele_table,
            self.tree,
            "test",
            self.temporary_directory,
            indel_colors=indel_color_df,
        )

        self.assertEqual(6, len(allele_files))

        for allele_file in allele_files:
            self.assertTrue(os.path.exists(allele_file))

    def test_create_allele_heatmap_specified_priors(self):

        allele_files = itol_utilities.create_indel_heatmap(
            self.allele_table,
            self.tree,
            "test",
            self.temporary_directory,
            indel_priors=self.indel_priors,
        )

        self.assertEqual(6, len(allele_files))

        for allele_file in allele_files:
            self.assertTrue(os.path.exists(allele_file))

    def test_create_allele_heatmap_no_priors_or_colormap(self):

        allele_files = itol_utilities.create_indel_heatmap(
            self.allele_table, self.tree, "test", self.temporary_directory
        )

        self.assertEqual(6, len(allele_files))

        for allele_file in allele_files:
            self.assertTrue(os.path.exists(allele_file))

    @unittest.skipUnless(
        os.path.exists(ITOL_FILE), "iTOL config file does not exist"
    )
    def test_integrated_pipeline_simple_tree(self):

        cas.pl.upload_and_export_itol(
            self.tree,
            "test_tree",
            export_filepath=f"{self.temporary_directory}/test_tree.pdf",
            itol_config="~/.itolconfig",
        )

        self.assertTrue(
            os.path.exists(
                os.path.join(self.temporary_directory, "test_tree.pdf")
            )
        )

    @unittest.skipUnless(
        os.path.exists(ITOL_FILE), "iTOL config file does not exist"
    )
    def test_integrated_pipeline_tree_with_allele_heatmap(self):

        cas.pl.upload_and_export_itol(
            self.tree,
            "test_tree",
            export_filepath=f"{self.temporary_directory}/test_tree.pdf",
            itol_config="~/.itolconfig",
            allele_table=self.allele_table,
        )

        self.assertTrue(
            os.path.exists(
                os.path.join(self.temporary_directory, "test_tree.pdf")
            )
        )

    @unittest.skipUnless(
        os.path.exists(ITOL_FILE), "iTOL config file does not exist"
    )
    def test_integrated_pipeline_tree_with_meta_data(self):

        cas.pl.upload_and_export_itol(
            self.tree,
            "test_tree",
            export_filepath=f"{self.temporary_directory}/test_tree.pdf",
            itol_config="~/.itolconfig",
            meta_data=["nUMI", "cluster"],
        )

        self.assertTrue(
            os.path.exists(
                os.path.join(self.temporary_directory, "test_tree.pdf")
            )
        )

    def tearDown(self):

        shutil.rmtree(self.temporary_directory)


if __name__ == "__main__":
    unittest.main()
