"""
Test config parsing for cassiopeia_preprocess pipeline.
"""

import os
import unittest

import pandas as pd
from pathlib import Path

from cassiopeia.preprocess import cassiopeia_preprocess
from cassiopeia.preprocess import constants
from cassiopeia.preprocess import setup_utilities


class TestCollapseUMIs(unittest.TestCase):
    def setUp(self):

        self.basic_config = {
            "general": {
                "name": "'test'",
                "output_directory": "'here'",
                "reference_filepath": "'ref.fa'",
                "input_files": ["input1.txt", "input2.txt"],
                "n_threads": 1,
            },
            "collapse": {"max_hq_mismatches": 5},
            "call_alleles": {"barcode_interval": (10, 25)},
        }

        self.basic_config_string = ""
        for key in self.basic_config:
            self.basic_config_string += f"[{key}]\n"
            for k, v in self.basic_config[key].items():
                self.basic_config_string += f"{k} = {v}\n"

        self.subset_config = {
            "general": {
                "name": "'test'",
                "output_directory": "'here'",
                "reference_filepath": "'ref.fa'",
                "input_files": ["input.txt"],
                "entry": "'align'",
                "exit": "'filter_molecule_table'",
                "n_threads": 1,
            },
            "collapse": {"max_hq_mismatches": 5},
            "call_alleles": {"barcode_interval": (10, 25)},
        }

        self.subset_config_string = ""
        for key in self.subset_config:
            self.subset_config_string += f"[{key}]\n"
            for k, v in self.subset_config[key].items():
                self.subset_config_string += f"{k} = {v}\n"

        self.failure_config = {
            "collapse": {"max_hq_mismatches": 5},
            "align": {"barcode_interval": [10, 25]},
        }

        self.failure_config_string = ""
        for key in self.failure_config:
            self.failure_config_string += f"[{key}]\n"
            for k, v in self.failure_config[key].items():
                self.failure_config_string += f"{k} = {v}\n"

    def test_read_good_config(self):

        parameters = setup_utilities.parse_config(self.basic_config_string)

        # check some default parameters
        self.assertEqual(
            parameters["error_correct_umis"]["max_umi_distance"], 2
        )
        self.assertFalse(
            parameters["error_correct_umis"]["allow_allele_conflicts"]
        )
        self.assertEqual(parameters["align"]["gap_open_penalty"], 20)
        self.assertEqual(parameters["call_alleles"]["cutsite_width"], 12)
        self.assertFalse(
            parameters["filter_molecule_table"]["allow_allele_conflicts"]
        )

        # check parameters updated correctly
        self.assertEqual(parameters["general"]["output_directory"], "here")
        self.assertEqual(parameters["general"]["reference_filepath"], "ref.fa")
        self.assertEqual(
            parameters["general"]["input_files"], ["input1.txt", "input2.txt"]
        )
        self.assertEqual(parameters["collapse"]["max_hq_mismatches"], 5)
        self.assertEqual(
            parameters["call_alleles"]["barcode_interval"], (10, 25)
        )

        self.assertIn("output_directory", parameters["convert"].keys())
        self.assertIn(
            "output_directory",
            parameters["error_correct_cellbcs_to_whitelist"].keys(),
        )
        self.assertIn("output_directory", parameters["collapse"].keys())
        self.assertIn("output_directory", parameters["resolve"].keys())
        self.assertIn(
            "output_directory", parameters["filter_molecule_table"].keys()
        )
        self.assertIn("output_directory", parameters["call_lineages"].keys())

    def test_unspecified_config_raises_error(self):

        self.assertRaises(
            setup_utilities.UnspecifiedConfigParameterError,
            setup_utilities.parse_config,
            self.failure_config_string,
        )

    def test_pipeline_setup_correct(self):

        parameters = setup_utilities.parse_config(self.basic_config_string)

        entry_point = parameters["general"]["entry"]
        exit_point = parameters["general"]["exit"]

        pipeline_procedures = setup_utilities.create_pipeline(
            entry_point, exit_point, cassiopeia_preprocess.STAGES
        )

        expected_procedures = [
            "convert",
            "filter_bam",
            "error_correct_cellbcs_to_whitelist",
            "collapse",
            "resolve",
            "align",
            "call_alleles",
            "error_correct_intbcs_to_whitelist",
            "error_correct_umis",
            "filter_molecule_table",
            "call_lineages",
        ]

        self.assertEqual(len(pipeline_procedures), len(expected_procedures))
        for i in range(len(pipeline_procedures)):
            self.assertEqual(pipeline_procedures[i], expected_procedures[i])

    def test_subset_pipeline_setup_correct(self):

        parameters = setup_utilities.parse_config(self.subset_config_string)

        entry_point = parameters["general"]["entry"]
        exit_point = parameters["general"]["exit"]

        pipeline_procedures = setup_utilities.create_pipeline(
            entry_point, exit_point, cassiopeia_preprocess.STAGES
        )

        expected_procedures = [
            "align",
            "call_alleles",
            "error_correct_intbcs_to_whitelist",
            "error_correct_umis",
            "filter_molecule_table",
        ]

        self.assertEqual(len(pipeline_procedures), len(expected_procedures))
        for i in range(len(pipeline_procedures)):
            self.assertEqual(pipeline_procedures[i], expected_procedures[i])


if __name__ == "__main__":
    unittest.main()
