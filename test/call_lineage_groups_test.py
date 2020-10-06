import os
import unittest

import numpy as np
import pandas as pd
import pathlib as Path

import cassiopeia
from cassiopeia.preprocess import pipeline


class TestCallLineageGroup(unittest.TestCase):
    def setUp(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        if not os.path.exists(dir_path + "/test_files"):
            os.makedirs(dir_path + "/test_files")

        self.basic_grouping = pd.DataFrame.from_dict(
            {
                "cellBC": ["A", "A", "A", "A", "B", "B", "B", "C", "C", "C"],
                "UMI": [
                    "AACCT",
                    "AACCG",
                    "AACCC",
                    "AACGT",
                    "AACGC",
                    "AACCG",
                    "AACCT",
                    "AACTA",
                    "AAGGA",
                    "ACGTA",
                ],
                "ReadCount": [10, 30, 30, 40, 10, 110, 20, 15, 10, 10],
                "Seq": ["NC"] * 10,
                "intBC": [
                    "XX",
                    "XX",
                    "XX",
                    "YZ",
                    "XX",
                    "YZ",
                    "ZY",
                    "YX",
                    "XY",
                    "XZ",
                ],
                "r1": ["1", "1", "1", "1", "1", "1", "1", "1", "1", "1"],
                "r2": ["2", "2", "2", "2", "2", "2", "2", "2", "2", "2"],
                "r3": ["3", "3", "3", "3", "3", "3", "3", "3", "2", "2"],
                "AlignmentScore": ["20"] * 10,
                "CIGAR": ["NA"] * 10,
                "Querybegin": [0] * 10,
                "Referencebegin": [0] * 10,
            }
        )
        self.basic_grouping["readName"] = self.basic_grouping.apply(
            lambda x: "_".join([x.cellBC, x.UMI, str(x.ReadCount)]), axis=1
        )

        self.basic_grouping["allele"] = self.basic_grouping.apply(
            lambda x: "_".join([x.r1, x.r2, x.r3]), axis=1
        )

        pipeline.call_lineage_groups(
            self.basic_grouping,
            "basic_grouping.csv",
            dir_path + "/test_files",
            cell_umi_filter=0,
            min_intbc_thresh=0.5,
        )

        self.reassign = pd.DataFrame.from_dict(
            {
                "cellBC": [
                    "A",
                    "A",
                    "B",
                    "B",
                    "C",
                    "C",
                    "D",
                    "D",
                    "E",
                    "E",
                    "F",
                    "F",
                    "F",
                ],
                "UMI": [
                    "AACCT",
                    "AACCG",
                    "AACCC",
                    "AACGT",
                    "AACGC",
                    "AACCG",
                    "AACCT",
                    "AACTA",
                    "AAGGA",
                    "ACGTA",
                    "CAGTA",
                    "CTGAT",
                    "CGGTA",
                ],
                "ReadCount": [
                    10,
                    30,
                    30,
                    40,
                    10,
                    110,
                    20,
                    15,
                    10,
                    10,
                    10,
                    10,
                    10,
                ],
                "Seq": ["NC"] * 13,
                "intBC": [
                    "XX",
                    "XX",
                    "XX",
                    "XX",
                    "XX",
                    "XX",
                    "XX",
                    "YZ",
                    "YZ",
                    "XZ",
                    "XZ",
                    "XZ",
                    "XZ",
                ],
                "r1": ["1"] * 13,
                "r2": ["2"] * 13,
                "r3": ["3"] * 13,
                "AlignmentScore": ["20"] * 13,
                "CIGAR": ["NA"] * 13,
                "Querybegin": [0] * 13,
                "Referencebegin": [0] * 13,
            }
        )
        self.reassign["readName"] = self.reassign.apply(
            lambda x: "_".join([x.cellBC, x.UMI, str(x.ReadCount)]), axis=1
        )

        self.reassign["allele"] = self.reassign.apply(
            lambda x: "_".join([x.r1, x.r2, x.r3]), axis=1
        )

        pipeline.call_lineage_groups(
            self.reassign,
            "reassign.csv",
            dir_path + "/test_files",
            cell_umi_filter=0,
            min_intbc_thresh=0.25,
            kinship_thresh=0.1,
        )

        self.filter_and_reassign = pd.DataFrame.from_dict(
            {
                "cellBC": [
                    "A",
                    "B",
                    "C",
                    "C",
                    "D",
                    "D",
                    "E",
                    "E",
                    "E",
                    "F",
                    "F",
                ],
                "UMI": [
                    "AACCC",
                    "AACGT",
                    "AACGC",
                    "AACCG",
                    "AACCT",
                    "AACTA",
                    "AAGGA",
                    "ACGTA",
                    "CAGTA",
                    "CTGAT",
                    "CGGTA",
                ],
                "ReadCount": [30, 40, 10, 110, 20, 15, 10, 10, 10, 10, 10],
                "Seq": ["NC"] * 11,
                "intBC": [
                    "XX",
                    "XX",
                    "XX",
                    "YZ",
                    "XX",
                    "YZ",
                    "YZ",
                    "XZ",
                    "XZ",
                    "XZ",
                    "XZ",
                ],
                "r1": ["1"] * 11,
                "r2": ["2"] * 11,
                "r3": ["3"] * 11,
                "AlignmentScore": ["20"] * 11,
                "CIGAR": ["NA"] * 11,
                "Querybegin": [0] * 11,
                "Referencebegin": [0] * 11,
            }
        )
        self.filter_and_reassign["readName"] = self.filter_and_reassign.apply(
            lambda x: "_".join([x.cellBC, x.UMI, str(x.ReadCount)]), axis=1
        )

        self.filter_and_reassign["allele"] = self.filter_and_reassign.apply(
            lambda x: "_".join([x.r1, x.r2, x.r3]), axis=1
        )

        pipeline.call_lineage_groups(
            self.filter_and_reassign,
            "filter_and_reassign.csv",
            dir_path + "/test_files",
            cell_umi_filter=0,
            min_intbc_thresh=0.5,
            kinship_thresh=0.1,
        )

        self.doublet = pd.DataFrame.from_dict(
            {
                "cellBC": [
                    "A",
                    "A",
                    "B",
                    "B",
                    "C",
                    "C",
                    "D",
                    "D",
                    "E",
                    "E",
                    "F",
                ],
                "UMI": [
                    "AACCT",
                    "AACCG",
                    "AACCC",
                    "AACGT",
                    "AACGC",
                    "AACCG",
                    "AACCT",
                    "AACTA",
                    "AAGGA",
                    "ACGTA",
                    "CAGTA",
                ],
                "ReadCount": [10, 30, 30, 40, 10, 110, 20, 15, 10, 10, 10],
                "Seq": ["NC"] * 11,
                "intBC": [
                    "XX",
                    "XX",
                    "XX",
                    "XX",
                    "XX",
                    "XY",
                    "XY",
                    "XY",
                    "XY",
                    "XY",
                    "XZ",
                ],
                "r1": ["1"] * 11,
                "r2": ["2"] * 11,
                "r3": ["3"] * 11,
                "AlignmentScore": ["20"] * 11,
                "CIGAR": ["NA"] * 11,
                "Querybegin": [0] * 11,
                "Referencebegin": [0] * 11,
            }
        )
        self.doublet["readName"] = self.doublet.apply(
            lambda x: "_".join([x.cellBC, x.UMI, str(x.ReadCount)]), axis=1
        )

        self.doublet["allele"] = self.doublet.apply(
            lambda x: "_".join([x.r1, x.r2, x.r3]), axis=1
        )

        pipeline.call_lineage_groups(
            self.doublet,
            "doublet.csv",
            dir_path + "/test_files",
            cell_umi_filter=1,
            min_intbc_thresh=0.5,
            kinship_thresh=0.5,
            inter_doublet_threshold=0.6,
        )

    def test_format(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        aln_df = pd.read_csv(
            dir_path + "/test_files/doublet.csv", sep="\t", header=0
        )

        expected_columns = [
            "readName",
            "cellBC",
            "UMI",
            "readName",
            "ReadCount",
            "intBC",
            "r1",
            "r2",
            "r3",
            "lineageGrp",
        ]
        for column in expected_columns:
            self.assertIn(column, aln_df.columns)

    def test_basic_grouping(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        aln_df = pd.read_csv(
            dir_path + "/test_files/basic_grouping.csv", sep="\t"
        )

        expected_alignments = {
            "A_AACCT_10": 1,
            "B_AACCG_110": 1,
            "C_AACTA_15": 2,
        }

        for read_name in expected_alignments:

            expected_lineage = expected_alignments[read_name]

            self.assertEqual(
                aln_df.loc[aln_df["readName"] == read_name, "lineageGrp"].iloc[
                    0
                ],
                expected_lineage,
            )

    def test_doublet(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        aln_df = pd.read_csv(dir_path + "/test_files/doublet.csv", sep="\t")

        samples = aln_df["Sample"]
        self.assertNotIn("C", samples)
        self.assertNotIn("F", samples)

        expected_alignments = {
            "A_AACCT_10": 1,
            "B_AACCC_30": 1,
            "D_AACTA_15": 2,
            "E_ACGTA_10": 2,
        }

        for read_name in expected_alignments:

            expected_lineage = expected_alignments[read_name]

            self.assertEqual(
                aln_df.loc[aln_df["readName"] == read_name, "lineageGrp"].iloc[
                    0
                ],
                expected_lineage,
            )

    def test_reassign(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        aln_df = pd.read_csv(dir_path + "/test_files/reassign.csv", sep="\t")

        expected_alignments = {
            "A_AACCT_10": 1,
            "B_AACCC_30": 1,
            "C_AACCG_110": 1,
            "D_AACTA_15": 1,
            "E_ACGTA_10": 1,
            "F_CGGTA_10": 1,
        }

        for read_name in expected_alignments:

            expected_lineage = expected_alignments[read_name]

            self.assertEqual(
                aln_df.loc[aln_df["readName"] == read_name, "lineageGrp"].iloc[
                    0
                ],
                expected_lineage,
            )

    def test_filter_reassign(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        aln_df = pd.read_csv(
            dir_path + "/test_files/filter_and_reassign.csv", sep="\t"
        )

        rns = aln_df["readName"]
        self.assertNotIn("C_AACCG_110", rns)
        self.assertNotIn("D_AACTA_15", rns)

        expected_alignments = {
            "A_AACCC_30": 1,
            "B_AACGT_40": 1,
            "C_AACGC_10": 1,
            "D_AACCT_20": 1,
            "E_ACGTA_10": 2,
            "F_CGGTA_10": 2,
        }

        for read_name in expected_alignments:

            expected_lineage = expected_alignments[read_name]

            self.assertEqual(
                aln_df.loc[aln_df["readName"] == read_name, "lineageGrp"].iloc[
                    0
                ],
                expected_lineage,
            )


if __name__ == "__main__":
    unittest.main()
