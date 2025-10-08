"""
Tests for the UMI Resolution module in pipeline.py.
"""

import shutil
import tempfile
import unittest

import pandas as pd

from cassiopeia.preprocess import pipeline


class TestResolveUMISequence(unittest.TestCase):
    def setUp(self):
        collapsed_umi_table_dict = {
            "cellBC": [
                "cell1",
                "cell1",
                "cell1",
                "cell2",
                "cell2",
                "cell3",
                "cell3",
            ],
            "UMI": ["UMIA", "UMIA", "UMIC", "UMIA", "UMIB", "UMIA", "UMIB"],
            "readCount": [9, 20, 11, 2, 1, 40, 30],
            "grpFlag": [0, 0, 0, 0, 0, 0, 0],
            "seq": [
                "AATCCG",
                "AAGGTT",
                "CCATTA",
                "ATACTG",
                "GGGAAT",
                "TTTCCTT",
                "CCAATTG",
            ],
            "qual": [
                "FFFFFF",
                "FFFFFF",
                "FFFFFF",
                "FFFFFF",
                "FFFFFF",
                "FFFFFF",
                "FFFFFF",
            ],
            "readName": [
                "cell1_UMIA_9_0",
                "cell1_UMIA_20_0",
                "cell1_UMIC_11_0",
                "cell2_UMIA_2",
                "cell2_UMIB_1",
                "cell3_UMIA_40",
                "cell3_UMIB_30",
            ],
        }
        self.collapsed_umi_table = pd.DataFrame.from_dict(collapsed_umi_table_dict)

        # set up temporary directory
        self.temporary_directory = tempfile.mkdtemp()

    def test_resolve_umi(self):
        resolved_mt = pipeline.resolve_umi_sequence(
            self.collapsed_umi_table, self.temporary_directory, min_umi_per_cell=1, plot=False
        )

        # check that cell1-UMIA was selected correctly
        expected_seq = "AAGGTT"
        observed_seq = resolved_mt.loc[resolved_mt["readName"] == "cell1_UMIA_20_0", "seq"].values
        self.assertEqual(expected_seq, observed_seq)

        # check that cell2 was filtered
        self.assertNotIn("cell2", resolved_mt["cellBC"].unique())

        # check that cell3 didn't lose UMIs
        self.assertEqual(2, resolved_mt[resolved_mt["cellBC"] == "cell3"].shape[0])

        # check expected reads
        expected = {"cell1": 31, "cell3": 70}
        for n, g in resolved_mt.groupby("cellBC"):
            self.assertEqual(expected[n], g["readCount"].sum())

    def test_filter_by_reads(self):
        resolved_mt = pipeline.resolve_umi_sequence(
            self.collapsed_umi_table,
            self.temporary_directory,
            min_avg_reads_per_umi=30,
            min_umi_per_cell=1,
            plot=True,
        )

        expected_cells = ["cell3"]
        expected_removed_cells = ["cell1", "cell2"]

        # print(expected_cells)

        for cell in expected_cells:
            self.assertIn(cell, resolved_mt["cellBC"].unique())

        for cell in expected_removed_cells:
            self.assertNotIn(cell, resolved_mt["cellBC"].unique())

    def tearDown(self):
        shutil.rmtree(self.temporary_directory)


if __name__ == "__main__":
    unittest.main()
