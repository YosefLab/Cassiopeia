import os
import unittest

import pandas as pd

import cassiopeia


class TestErrorCorrectIntBCstoWhitelist(unittest.TestCase):
    def setUp(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        test_files_path = os.path.join(dir_path, "test_files")
        self.whitelist_fp = os.path.join(test_files_path, "intbc_whitelist.txt")
        self.whitelist = ["ACTT", "TAAG"]

        self.multi_case = pd.DataFrame.from_dict(
            {
                "cellBC": [
                    "A",
                    "A",
                    "A",
                    "B",
                    "B",
                    "C",
                    "C",
                    "C",
                    "C",
                    "D",
                    "D",
                ],
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
                    "AACCT",
                    "AAGGG",
                ],
                "readCount": [20, 30, 30, 40, 50, 10, 10, 15, 10, 10, 10],
                "Seq": [
                    "AACCTTGG",
                    "AACCTTGG",
                    "AACCTTCC",
                    "AACCTTGG",
                    "AACCTTGC",
                    "AACCTTCC",
                    "AACCTTCG",
                    "AACCTCAG",
                    "AACCTTGG",
                    "AACCTTGG",
                    "AACCTAAA",
                ],
                "intBC": [
                    "ACTT",
                    "AAGG",
                    "ACTA",
                    "AAGN",
                    "TACT",
                    "TAAG",
                    "TNNG",
                    "ANNN",
                    "GCTT",
                    "NNNN",
                    "AAAA",
                ],
                "r1": ["1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1"],
                "r2": ["2", "2", "2", "2", "2", "2", "2", "2", "2", "2", "2"],
                "r3": ["3", "3", "3", "3", "3", "3", "3", "3", "3", "3", "3"],
                "AlignmentScore": [
                    "20",
                    "20",
                    "20",
                    "20",
                    "20",
                    "20",
                    "20",
                    "20",
                    "20",
                    "20",
                    "20",
                ],
                "CIGAR": [
                    "NA",
                    "NA",
                    "NA",
                    "NA",
                    "NA",
                    "NA",
                    "NA",
                    "NA",
                    "NA",
                    "NA",
                    "NA",
                ],
            }
        )
        self.multi_case["readName"] = self.multi_case.apply(
            lambda x: "_".join([x.cellBC, x.UMI, str(x.readCount)]), axis=1
        )

        self.multi_case["allele"] = self.multi_case.apply(
            lambda x: "_".join([x.r1, x.r2, x.r3]), axis=1
        )
        self.corrections = {
            "ACTT": "ACTT",
            "TAAG": "TAAG",
            "ACTA": "ACTT",
            "TNNG": "TAAG",
            "ANNN": "ACTT",
        }

    def test_correct(self):
        df = cassiopeia.pp.error_correct_intbcs_to_whitelist(
            self.multi_case, self.whitelist_fp, intbc_dist_thresh=1
        )
        expected_df = self.multi_case.copy()
        expected_df["intBC"] = expected_df["intBC"].map(self.corrections)
        expected_df.dropna(subset=["intBC"], inplace=True)

        pd.testing.assert_frame_equal(df, expected_df)

    def test_correct_whitelist_list(self):
        df = cassiopeia.pp.error_correct_intbcs_to_whitelist(
            self.multi_case, self.whitelist, intbc_dist_thresh=1
        )
        expected_df = self.multi_case.copy()
        expected_df["intBC"] = expected_df["intBC"].map(self.corrections)
        expected_df.dropna(subset=["intBC"], inplace=True)

        pd.testing.assert_frame_equal(df, expected_df)


if __name__ == "__main__":
    unittest.main()
