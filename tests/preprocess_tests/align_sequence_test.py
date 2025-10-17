"""
Tests for the sequence alignment in pipeline.py.
"""

import unittest

import pandas as pd

import cassiopeia


class TestAlignSequence(unittest.TestCase):
    def setUp(self):
        self.queries = pd.DataFrame.from_dict(
            {
                "cellBC": ["A", "A", "A", "B", "B", "C", "C", "C"],
                "UMI": ["1", "2", "3", "1", "2", "1", "2", "3"],
                "readCount": [20, 30, 30, 40, 40, 10, 10, 15],
                "seq": [
                    "AACCTTGG",
                    "ACTG",
                    "AACCTTGGACTGCATCG",
                    "AATTAA",
                    "ACTGGACT",
                    "AACCTTGGGG",
                    "AAAAAAAAAAA",
                    "TACTCTATA",
                ],
            }
        )
        self.queries["readName"] = self.queries.apply(
            lambda x: "_".join([x.cellBC, x.UMI, str(x.readCount)]), axis=1
        )

        self.reference = "AACCTTGG"

    def test_alignment_dataframe_structure(self):
        aln_df = cassiopeia.pp.align_sequences(
            self.queries,
            ref=self.reference,
            gap_open_penalty=20,
            gap_extend_penalty=1,
            n_threads=2,
        )

        self.assertEqual(aln_df.shape[0], self.queries.shape[0])

        for cellBC in self.queries["cellBC"].unique():
            self.assertIn(cellBC, aln_df["cellBC"].unique())

        expected_columns = [
            "cellBC",
            "UMI",
            "AlignmentScore",
            "CIGAR",
            "QueryBegin",
            "ReferenceBegin",
            "Seq",
            "readName",
            "readCount",
        ]

        for column in expected_columns:
            self.assertIn(column, aln_df.columns)

    def test_extremely_large_gap_open_penalty(self):
        aln_df = cassiopeia.pp.align_sequences(
            self.queries,
            ref=self.reference,
            gap_open_penalty=255,
            gap_extend_penalty=1,
        )

        # since the gap open penalty is so large, enforce that
        # no gaps should occur
        for _ind, row in aln_df.iterrows():
            self.assertNotIn("D", row.CIGAR)
            self.assertNotIn("I", row.CIGAR)

    def test_default_alignment_works(self):
        aln_df = cassiopeia.pp.align_sequences(
            self.queries,
            ref=self.reference,
            gap_open_penalty=2,
            gap_extend_penalty=1,
        )

        expected_alignments = {
            "A_1_20": ("8M", 40),
            "A_2_30": ("2M2D2M", 17),
            "A_3_30": ("8M", 40),
            "B_1_40": ("2M2D2M", 17),
            "B_2_40": ("2M2D3M", 22),
            "C_1_10": ("8M", 40),
            "C_2_10": ("2M", 10),
            "C_3_15": ("2M1I2M1I1M", 21),
        }

        for read_name in aln_df["readName"].unique():
            expected_cigar = expected_alignments[read_name][0]
            expected_score = expected_alignments[read_name][1]

            self.assertEqual(
                aln_df.loc[aln_df["readName"] == read_name, "CIGAR"].iloc[0],
                expected_cigar,
            )
            self.assertEqual(
                aln_df.loc[aln_df["readName"] == read_name, "AlignmentScore"].iloc[0],
                expected_score,
            )

    def test_global_alignment(self):
        aln_df = cassiopeia.pp.align_sequences(
            self.queries,
            ref=self.reference,
            gap_open_penalty=2,
            gap_extend_penalty=1,
            method="global",
        )

        expected_alignments = {
            "A_1_20": ("8M", 40),
            "A_2_30": ("1M2D2M1D1M1D", 15),
            "A_3_30": ("8M9I", 40),
            "B_1_40": ("2M2D2M2D2I", 14),
            "B_2_40": ("1M2D2M1D2M3I", 20),
            "C_1_10": ("8M2I", 40),
            "C_2_10": ("2M6D9I", 3),
            "C_3_15": ("1I1M1D1M1I2M1I1M1I2D", 15),
        }

        for read_name in aln_df["readName"].unique():
            expected_cigar = expected_alignments[read_name][0]
            expected_score = expected_alignments[read_name][1]

            self.assertEqual(
                aln_df.loc[aln_df["readName"] == read_name, "CIGAR"].iloc[0],
                expected_cigar,
            )
            self.assertEqual(
                aln_df.loc[aln_df["readName"] == read_name, "AlignmentScore"].iloc[0],
                expected_score,
            )


if __name__ == "__main__":
    unittest.main()
