"""
Tests for the UMI Collapsing module in pipeline.py
"""
import unittest
unittest.TestLoader.sortTestMethodsUsing = None


import time

import pandas as pd
from pathlib import Path
import pysam


from cassiopeia.ProcessingPipeline.process import pipeline
from cassiopeia.ProcessingPipeline.process import UMI_utils
from cassiopeia.ProcessingPipeline.process import utilities


class TestConvertBam2DF(unittest.TestCase):
    def setUp(self):
        self.test_file = "test.bam"
        sorted_file_name = Path(
            "."
            + "/"
            + ".".join(self.test_file.split("/")[-1].split(".")[:-1])
            + "_sorted.bam"
        )
        self.sorted_file_name = sorted_file_name
        self.collapsed_file_name = sorted_file_name.with_suffix(
            ".collapsed.bam"
        )

        _, _ = UMI_utils.sort_cellranger_bam(
            self.test_file,
            str(self.sorted_file_name),
            show_progress=False,
        )

        UMI_utils.form_collapsed_clusters(
            str(self.sorted_file_name),
            max_hq_mismatches=3,
            max_indels=2,
            show_progress=False,
        )

    def test_sort_bam(self):
        
        sorted_bam = pysam.AlignmentFile(
            self.sorted_file_name, "rb", check_sq=False
        )

        cellBCs = []
        UMIs = []
        for al in sorted_bam:
            cellBCs.append(al.get_tag("CB"))
            UMIs.append(al.get_tag("UR"))

        expected_cellBC = "GACCCTCGTGGGTATG-1"
        expected_UMI = "TGGCCTTTAA"

        self.assertEqual(len(cellBCs), 16)
        self.assertEqual(expected_cellBC, cellBCs[10])
        self.assertEqual(expected_UMI, UMIs[7])

    def test_collapse_bam(self):

        collapsed_bam = pysam.AlignmentFile(
            self.collapsed_file_name, "rb", check_sq=False
        )
        cellBCs = []
        UMIs = []
        readCounts = []
        clusterIds = []
        quals = []
        for al in collapsed_bam:
            cellBCs.append(al.get_tag("CB"))
            UMIs.append(al.get_tag("UR"))
            readCounts.append(al.get_tag("ZR"))
            clusterIds.append(al.get_tag("ZC"))
            quals.append(al.query_qualities)

        expected_cellBC = "CAACCTCGTGGGTATG-1"
        expected_UMI = "TGGCCTTTAA"

        self.assertEqual(len(cellBCs), 5)
        self.assertEqual(expected_cellBC, cellBCs[0])
        self.assertEqual(expected_UMI, UMIs[1])
        self.assertEqual([7, 1, 2, 3, 3], readCounts)
        self.assertEqual(2, quals[2][0])

    def test_bam2DF(self):
        collapsed_df_file_name = self.sorted_file_name.with_suffix(
            ".collapsed.txt"
        )
        ret = utilities.convertBam2DF(
            str(self.collapsed_file_name),
            str(collapsed_df_file_name),
            create_pd=True,
        )

        expected_qual = "#@@@@@@@@@@@@@@"
        expected_readcount = 3
        expected_readname = "GACCCTCGTGGGTATG-1_GATAACATCG_000003_1"

        self.assertEqual(ret.shape, (5, 7))
        self.assertEqual(ret.iloc[2, 5], expected_qual)
        self.assertEqual(ret.iloc[3, 2], expected_readcount)
        self.assertEqual(ret.iloc[4, 6], expected_readname)

        df = pd.read_csv(collapsed_df_file_name, sep="\t")

        self.assertEqual(ret.shape, (5, 7))
        self.assertEqual(df.iloc[2, 5], expected_qual)
        self.assertEqual(df.iloc[3, 2], expected_readcount)
        self.assertEqual(df.iloc[4, 6], expected_readname)


if __name__ == "__main__":
    unittest.main()
