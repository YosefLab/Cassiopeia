"""
Test config parsing for cassiopeia_preprocess pipeline.
"""

import os
import unittest

import pandas as pd
from pathlib import Path

from cassiopeia.preprocess import constants
from cassiopeia.preprocess import setup_utilities


class TestCollapseUMIs(unittest.TestCase):
    def setUp(self):

        self.basic_config = {
            "general": {
                "output_directory": "'here'",
                "reference_filepath": "'ref.fa'",
                "input_file": "'input.txt'",
            },
            "collapse": {"max_hq_mismatches": 5},
            "call_alleles": {"barcode_interval": (10, 25)},
        }

        self.basic_config_string = ""
        for key in self.basic_config:
            self.basic_config_string += f"[{key}]\n"
            for k, v in self.basic_config[key].items():
                self.basic_config_string += f"{k} = {v}\n"

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
        self.assertEqual(parameters["error_correct"]["max_umi_distance"], 2)
        self.assertEqual(parameters["align"]["gap_open_penalty"], 20)
        self.assertEqual(parameters["call_alleles"]["cutsite_width"], 12)

        # check parameters updated correctly
        self.assertEqual(parameters["general"]["output_directory"], "here")
        self.assertEqual(parameters["general"]["reference_filepath"], "ref.fa")
        self.assertEqual(parameters["general"]["input_file"], "input.txt")
        self.assertEqual(parameters["collapse"]["max_hq_mismatches"], 5)
        self.assertEqual(
            parameters["call_alleles"]["barcode_interval"], (10, 25)
        )

    def test_unspecified_config_raises_error(self):

        self.assertRaises(
            setup_utilities.UnspecifiedConfigParameterError,
            setup_utilities.parse_config,
            self.failure_config_string,
        )


if __name__ == "__main__":
    unittest.main()
