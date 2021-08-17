import unittest

import numpy as np
import pandas as pd

import cassiopeia
from cassiopeia.preprocess import pipeline


class TestFilterMoleculeTable(unittest.TestCase):
    def setUp(self):

        self.base_filter_case = pd.DataFrame.from_dict(
            {
                "cellBC": ["A", "A", "A", "B", "B", "C", "C", "C", "C"],
                "UMI": [
                    "AACCT",
                    "AACCG",
                    "AACCC",
                    "AACCT",
                    "AACCG",
                    "AACCT",
                    "AACCG",
                    "AAGGA",
                    "AACCT",
                ],
                "readCount": [10, 30, 30, 40, 10, 110, 20, 15, 10],
                "Seq": ["NC"] * 9,
                "intBC": ["NC"] * 9,
                "r1": ["1"] * 9,
                "r2": ["2"] * 9,
                "r3": ["3"] * 9,
                "AlignmentScore": ["20"] * 9,
                "CIGAR": ["NA"] * 9,
                "Querybegin": [0] * 9,
                "Referencebegin": [0] * 9,
            }
        )
        self.base_filter_case["readName"] = self.base_filter_case.apply(
            lambda x: "_".join([x.cellBC, x.UMI, str(x.readCount)]), axis=1
        )

        self.base_filter_case["allele"] = self.base_filter_case.apply(
            lambda x: "_".join([x.r1, x.r2, x.r3]), axis=1
        )

        self.doublets_case = pd.DataFrame.from_dict(
            {
                "cellBC": ["A", "A", "A", "A", "C", "C", "C", "C", "C"],
                "UMI": [
                    "AACCT",
                    "AACCG",
                    "AACCC",
                    "AACGT",
                    "AACCG",
                    "AACCT",
                    "AACTA",
                    "AAGGA",
                    "ACGTA",
                ],
                "readCount": [10, 30, 30, 40, 10, 110, 20, 15, 10],
                "Seq": ["NC"] * 9,
                "intBC": ["T", "T", "T", "T", "A", "A", "A", "A", "A"],
                "r1": ["1", "1", "2", "2", "1", "1", "1", "1", "1"],
                "r2": ["2", "2", "2", "2", "2", "2", "2", "2", "2"],
                "r3": ["3", "3", "3", "3", "3", "3", "3", "2", "2"],
                "AlignmentScore": ["20"] * 9,
                "CIGAR": ["NA"] * 9,
                "Querybegin": [0] * 9,
                "Referencebegin": [0] * 9,
            }
        )
        self.doublets_case["readName"] = self.doublets_case.apply(
            lambda x: "_".join([x.cellBC, x.UMI, str(x.readCount)]), axis=1
        )

        self.doublets_case["allele"] = self.doublets_case.apply(
            lambda x: "_".join([x.r1, x.r2, x.r3]), axis=1
        )

        self.intBC_case = pd.DataFrame.from_dict(
            {
                "cellBC": ["A", "A", "A", "A", "C", "C", "C", "C", "C"],
                "UMI": [
                    "AACCT",
                    "AACCG",
                    "AACCC",
                    "AACGT",
                    "AACCG",
                    "AACCT",
                    "AACTA",
                    "AAGGA",
                    "ACGTA",
                ],
                "readCount": [10, 30, 30, 40, 10, 110, 20, 15, 10],
                "Seq": ["NC"] * 9,
                "intBC": ["AT", "TA", "TA", "AA", "TA", "TA", "AA", "AA", "AA"],
                "r1": ["1", "1", "1", "1", "1", "1", "1", "1", "1"],
                "r2": ["2", "2", "2", "2", "2", "2", "2", "2", "2"],
                "r3": ["3", "3", "3", "3", "3", "3", "3", "3", "2"],
                "AlignmentScore": ["20"] * 9,
                "CIGAR": ["NA"] * 9,
                "Querybegin": [0] * 9,
                "Referencebegin": [0] * 9,
            }
        )
        self.intBC_case["readName"] = self.intBC_case.apply(
            lambda x: "_".join([x.cellBC, x.UMI, str(x.readCount)]), axis=1
        )

        self.intBC_case["allele"] = self.intBC_case.apply(
            lambda x: "_".join([x.r1, x.r2, x.r3]), axis=1
        )

    def test_format(self):

        aln_df = pipeline.filter_molecule_table(
            self.base_filter_case, ".", min_umi_per_cell=2
        )

        expected_columns = [
            "cellBC",
            "UMI",
            "AlignmentScore",
            "CIGAR",
            "Seq",
            "readName",
            "readCount",
            "intBC",
            "r1",
            "r2",
            "r3",
            "Querybegin",
            "Referencebegin",
        ]
        for column in expected_columns:
            self.assertIn(column, aln_df.columns)

    def test_umi_and_cellbc_filter(self):

        aln_df = pipeline.filter_molecule_table(
            self.base_filter_case, ".", min_umi_per_cell=3, min_reads_per_umi=11
        )

        expected_alignments = {
            "C_AACCT_110": 110,
            "C_AACCG_20": 20,
            "C_AAGGA_15": 15,
        }

        for read_name in aln_df["readName"]:

            expected_readcount = expected_alignments[read_name]

            self.assertEqual(
                aln_df.loc[aln_df["readName"] == read_name, "readCount"].iloc[
                    0
                ],
                expected_readcount,
            )

    def test_doublet_and_map(self):

        aln_df = pipeline.filter_molecule_table(
            self.doublets_case,
            ".",
            min_umi_per_cell=1,
            min_reads_per_umi=0,
            doublet_threshold=0.4,
        )

        expected_alignments = {
            "C_AACCG_10": "1_2_3",
            "C_AACCT_110": "1_2_3",
            "C_AACTA_20": "1_2_3",
        }

        for read_name in aln_df["readName"]:

            expected_allele = expected_alignments[read_name]

            self.assertEqual(
                aln_df.loc[aln_df["readName"] == read_name, "allele"].iloc[0],
                expected_allele,
            )

    def test_error_correct_intBC(self):

        aln_df = pipeline.filter_molecule_table(
            self.intBC_case,
            ".",
            min_umi_per_cell=1,
            min_reads_per_umi=0,
            doublet_threshold=None,
        )

        expected_alignments = {
            "A_AACCT_10": "AT",
            "A_AACCG_30": "TA",
            "A_AACCC_30": "TA",
            "A_AACGT_40": "TA",
            "C_AACCG_10": "TA",
            "C_AACCT_110": "TA",
            "C_AACTA_20": "AA",
            "C_AAGGA_15": "AA",
        }

        for read_name in aln_df["readName"]:

            expected_intbc = expected_alignments[read_name]

            self.assertEqual(
                aln_df.loc[aln_df["readName"] == read_name, "intBC"].iloc[0],
                expected_intbc,
            )

    def test_filter_allow_conflicts(self):
        aln_df = pipeline.filter_molecule_table(
            self.doublets_case,
            ".",
            min_umi_per_cell=1,
            min_reads_per_umi=0,
            doublet_threshold=0.4,
            allow_allele_conflicts=True,
        )

        expected_alignments = {
            "C_AACCT_110": "A",
            "A_AACGT_40": "T",
            "A_AACCG_30": "T",
            "A_AACCC_30": "T",
            "C_AACTA_20": "A",
            "C_AAGGA_15": "A",
            "A_AACCT_10": "T",
            "C_AACCG_10": "A",
            "C_ACGTA_10": "A",
        }
        for read_name in aln_df["readName"]:

            expected_intbc = expected_alignments[read_name]

            self.assertEqual(
                aln_df.loc[aln_df["readName"] == read_name, "intBC"].iloc[0],
                expected_intbc,
            )


if __name__ == "__main__":
    unittest.main()
