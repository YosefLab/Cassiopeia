import os
import unittest

import pandas as pd
from cassiopeia.preprocess import lineage_utils, pipeline


class TestCallLineageGroup(unittest.TestCase):
    def setUp(self):

        self.dir_path = os.path.dirname(os.path.realpath(__file__))
        if not os.path.exists(self.dir_path + "/test_files"):
            os.makedirs(self.dir_path + "/test_files")

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
                "readCount": [10, 30, 30, 40, 10, 110, 20, 15, 10, 10],
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
            lambda x: "_".join([x.cellBC, x.UMI, str(x.readCount)]), axis=1
        )

        self.basic_grouping["allele"] = self.basic_grouping.apply(
            lambda x: "_".join([x.r1, x.r2, x.r3]), axis=1
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
                "readCount": [
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
            lambda x: "_".join([x.cellBC, x.UMI, str(x.readCount)]), axis=1
        )

        self.reassign["allele"] = self.reassign.apply(
            lambda x: "_".join([x.r1, x.r2, x.r3]), axis=1
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
                "readCount": [30, 40, 10, 110, 20, 15, 10, 10, 10, 10, 10],
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
            lambda x: "_".join([x.cellBC, x.UMI, str(x.readCount)]), axis=1
        )

        self.filter_and_reassign["allele"] = self.filter_and_reassign.apply(
            lambda x: "_".join([x.r1, x.r2, x.r3]), axis=1
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
                "readCount": [10, 30, 30, 40, 10, 110, 20, 15, 10, 10, 10],
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
            lambda x: "_".join([x.cellBC, x.UMI, str(x.readCount)]), axis=1
        )

        self.doublet["allele"] = self.doublet.apply(
            lambda x: "_".join([x.r1, x.r2, x.r3]), axis=1
        )

    def test_format(self):

        aln_df = pipeline.call_lineage_groups(
            self.doublet,
            self.dir_path + "/test_files",
            min_umi_per_cell=1,
            min_intbc_thresh=0.5,
            kinship_thresh=0.5,
            inter_doublet_threshold=0.6,
        )

        expected_columns = [
            "cellBC",
            "UMI",
            "readCount",
            "intBC",
            "r1",
            "r2",
            "r3",
            "lineageGrp",
        ]
        for column in expected_columns:
            self.assertIn(column, aln_df.columns)

    def test_basic_grouping(self):

        aln_df = pipeline.call_lineage_groups(
            self.basic_grouping,
            self.dir_path + "/test_files",
            min_umi_per_cell=0,
            min_intbc_thresh=0.5,
        )

        expected_rows = {
            ("A", "XX"): (1, 3),
            ("B", "XX"): (1, 1),
            ("C", "XZ"): (2, 1),
        }

        for pair in expected_rows:

            expected_lineage = expected_rows[pair]

            self.assertEqual(
                aln_df.loc[
                    (aln_df["cellBC"] == pair[0])
                    & (aln_df["intBC"] == pair[1]),
                    "lineageGrp",
                ].iloc[0],
                expected_lineage[0],
            )

            self.assertEqual(
                aln_df.loc[
                    (aln_df["cellBC"] == pair[0])
                    & (aln_df["intBC"] == pair[1]),
                    "UMI",
                ].iloc[0],
                expected_lineage[1],
            )

    def test_doublet(self):

        aln_df = pipeline.call_lineage_groups(
            self.doublet,
            self.dir_path + "/test_files",
            min_umi_per_cell=1,
            min_intbc_thresh=0.5,
            kinship_thresh=0.5,
            inter_doublet_threshold=0.6,
        )

        expected_rows = {
            ("A", "XX"): (1, 2),
            ("B", "XX"): (1, 2),
            ("D", "XY"): (2, 2),
            ("E", "XY"): (2, 2),
        }

        for pair in expected_rows:

            expected_lineage = expected_rows[pair]

            self.assertEqual(
                aln_df.loc[
                    (aln_df["cellBC"] == pair[0])
                    & (aln_df["intBC"] == pair[1]),
                    "lineageGrp",
                ].iloc[0],
                expected_lineage[0],
            )

            self.assertEqual(
                aln_df.loc[
                    (aln_df["cellBC"] == pair[0])
                    & (aln_df["intBC"] == pair[1]),
                    "UMI",
                ].iloc[0],
                expected_lineage[1],
            )

    def test_reassign(self):

        aln_df = pipeline.call_lineage_groups(
            self.reassign,
            self.dir_path + "/test_files",
            min_umi_per_cell=0,
            min_intbc_thresh=0.25,
            kinship_thresh=0.1,
        )

        expected_rows = {
            ("A", "XX"): (1, 2),
            ("B", "XX"): (1, 2),
            ("C", "XX"): (1, 2),
            ("D", "XX"): (1, 1),
            ("E", "XZ"): (1, 1),
            ("F", "XZ"): (1, 3),
        }

        for pair in expected_rows:

            expected_lineage = expected_rows[pair]

            self.assertEqual(
                aln_df.loc[
                    (aln_df["cellBC"] == pair[0])
                    & (aln_df["intBC"] == pair[1]),
                    "lineageGrp",
                ].iloc[0],
                expected_lineage[0],
            )

            self.assertEqual(
                aln_df.loc[
                    (aln_df["cellBC"] == pair[0])
                    & (aln_df["intBC"] == pair[1]),
                    "UMI",
                ].iloc[0],
                expected_lineage[1],
            )

    def test_filter_reassign(self):

        aln_df = pipeline.call_lineage_groups(
            self.filter_and_reassign,
            self.dir_path + "/test_files",
            min_umi_per_cell=0,
            min_intbc_thresh=0.5,
            kinship_thresh=0.1,
        )

        self.assertNotIn("YZ", aln_df["intBC"])

        expected_rows = {
            ("A", "XX"): (1, 1),
            ("B", "XX"): (1, 1),
            ("C", "XX"): (1, 1),
            ("D", "XX"): (1, 1),
            ("E", "XZ"): (2, 2),
            ("F", "XZ"): (2, 2),
        }

        for pair in expected_rows:

            expected_lineage = expected_rows[pair]

            self.assertEqual(
                aln_df.loc[
                    (aln_df["cellBC"] == pair[0])
                    & (aln_df["intBC"] == pair[1]),
                    "lineageGrp",
                ].iloc[0],
                expected_lineage[0],
            )

            self.assertEqual(
                aln_df.loc[
                    (aln_df["cellBC"] == pair[0])
                    & (aln_df["intBC"] == pair[1]),
                    "UMI",
                ].iloc[0],
                expected_lineage[1],
            )

    def test_filter_lineage_group_to_allele_table_single_lineage(self):

        aln_df = lineage_utils.filtered_lineage_group_to_allele_table(
            [self.basic_grouping]
        )

        self.assertIn("lineageGrp", aln_df)

        expected_rows = {
            ("A", "XX"): (1, 3),
            ("B", "XX"): (1, 1),
            ("C", "XZ"): (1, 1),
        }

        for pair in expected_rows:

            expected_lineage = expected_rows[pair]

            self.assertEqual(
                aln_df.loc[
                    (aln_df["cellBC"] == pair[0])
                    & (aln_df["intBC"] == pair[1]),
                    "lineageGrp",
                ].iloc[0],
                expected_lineage[0],
            )

            self.assertEqual(
                aln_df.loc[
                    (aln_df["cellBC"] == pair[0])
                    & (aln_df["intBC"] == pair[1]),
                    "UMI",
                ].iloc[0],
                expected_lineage[1],
            )


if __name__ == "__main__":
    unittest.main()
