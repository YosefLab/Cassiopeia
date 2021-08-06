"""
Tests for the allele calling in alignment_utilities.py and pipeline.py
"""
from cassiopeia.mixins.errors import PreprocessError
from cassiopeia.mixins.warnings import PreprocessWarning
import unittest

import numpy as np
import pandas as pd

import cassiopeia
from cassiopeia.preprocess import alignment_utilities


class TestCallAlleles(unittest.TestCase):
    def setUp(self):

        self.basic_ref = "ACNNTTAATT"
        self.basic_barcode_interval = (2, 4)  # 0 indexed
        self.basic_cutsites = [7]  # 0 indexed

        self.long_ref = (
            "AATCCAGCTAGCTGTGCAGCNNNNNNNNNNNNNNATTCAACTGCAGTAATGCTA"
            "CCTCGTACTCACGCTTTCCAAGTGCTTGGCGTCGCATCTCGGTCCTTTGTACGCC"
            "GAAAAATGGCCTGACAACTAAGCTACGGCACGCTGCCATGTTGGGTCATAACGA"
            "TATCTCTGGTTCATCCGTGACCGAACATGTCATGGAGTAGCAGGAGCTATTAAT"
            "TCGCGGAGGACAATGCGGTTCGTAGTCACTGTCTTCCGCAATCGTCCATCGCTC"
            "CTGCAGGTGGCCTAGAGGGCCCGTTTAAACCCGCTGATCAGCCTCGACTGTGCC"
            "TTCTAGTTGCCAGCCATCTGTTGTTTGCCCCTCCCCCGTGCCTTCCTTGACCCT"
            "GGAAGGTGCCACTCCCACTGTCCTTTCCTAATAAAATGAGGAAATTGCATCGCA"
            "TTGTCTGAGTAGGTGTCATTCTATTCTGGGGGGTGGGGTGGGGCAGGACAGCAA"
            "GGGGGAGGATTGGGAAGACAATAGCAGGCATGCTGGGGATGCGGTGGGCTCTAT"
            "GGTCTAGAGCGGGCCCGGTACTAACCAAACTGGATCTCTGCTGTCCCTGTAATA"
            "AACCCGAAAATTTTGAATTTTTGTAATTTGTTTTTGTAATTCTTTAGTTTGTAT"
            "GTCTGTTGCTATTATGTCTACTATTCTTTCCCCTGCACTGTACCCCCCAATCCC"
            "CCCTTTTCTTTTAAAATTGTGGATGAATACTGCCATTTGTCTGCAGA"
        )
        self.long_barcode_interval = (20, 34)
        self.long_cutsites = [112, 166, 220]

        self.alignment_dataframe = pd.DataFrame.from_dict(
            {
                "cellBC": ["A", "A", "B", "C"],
                "UMI": ["ATC", "TTG", "ACC", "CCA"],
                "readCount": [10, 20, 10, 10],
                "CIGAR": ["10M", "6M2D2M", "10M", "8M1I2M"],
                "QueryBegin": [0, 0, 0, 0],
                "ReferenceBegin": [0, 0, 0, 0],
                "AlignmentScore": [100, 90, 100, 70],
                "Seq": [
                    "ACGGTTAATT",
                    "ACATTTTTT",
                    "ACCCTTAATT",
                    "ACGATTAAGTGTT",
                ],
            }
        )

        self.alignment_dataframe["readName"] = self.alignment_dataframe.apply(
            lambda x: x.cellBC + "_" + x.UMI + "_" + str(x.readCount), axis=1
        )

        new_row = pd.DataFrame.from_dict({
            "cellBC": "C",
            "UMI": "CTG",
            "readCount": 10,
            "CIGAR": "3M",
            "QueryBegin": 0,
            "ReferenceBegin": 0,
            "AlignmentScore": 100,
            "Seq": "ACG",
            "readName": "C_CTG_10",
        }, orient='index').T
        self.alignment_dataframe_with_missing = pd.concat([self.alignment_dataframe, new_row])


    def test_basic_cigar_string_match(self):

        query = "ACGGTTAATT"
        cigar = "10M"
        cutsite_window = 1
        context = False

        intBC, indels = alignment_utilities.parse_cigar(
            cigar,
            query,
            self.basic_ref,
            0,
            0,
            self.basic_barcode_interval,
            self.basic_cutsites,
            cutsite_window,
            context,
        )

        self.assertEqual(intBC, "GG")
        self.assertEqual(len(self.basic_cutsites), len(indels))
        self.assertEqual(indels[0], "None")

    def test_basic_cigar_string_deletion(self):

        cigar = "6M2D2M"
        query = "ACGGTTTT"
        cutsite_window = 1
        context = False

        intBC, indels = alignment_utilities.parse_cigar(
            cigar,
            query,
            self.basic_ref,
            0,
            0,
            self.basic_barcode_interval,
            self.basic_cutsites,
            cutsite_window,
            context,
        )

        self.assertEqual(intBC, "GG")
        self.assertEqual(len(self.basic_cutsites), len(indels))
        self.assertEqual(indels[0], "7:2D")

    def test_basic_cigar_string_deletion_with_context(self):

        cigar = "6M2D2M"
        query = "ACGGTTTT"
        cutsite_window = 1
        context = True
        context_size = 1

        intBC, indels = alignment_utilities.parse_cigar(
            cigar,
            query,
            self.basic_ref,
            0,
            0,
            self.basic_barcode_interval,
            self.basic_cutsites,
            cutsite_window,
            context,
            context_size=context_size,
        )

        self.assertEqual(intBC, "GG")
        self.assertEqual(len(self.basic_cutsites), len(indels))
        self.assertEqual(indels[0], "T[7:2D]T")

    def test_basic_cigar_string_insertion(self):

        cigar = "8M3I2M"
        query = "ACGGTTAAGTGTT"
        cutsite_window = 1
        context = False

        intBC, indels = alignment_utilities.parse_cigar(
            cigar,
            query,
            self.basic_ref,
            0,
            0,
            self.basic_barcode_interval,
            self.basic_cutsites,
            cutsite_window,
            context,
        )

        self.assertEqual(intBC, "GG")
        self.assertEqual(len(self.basic_cutsites), len(indels))
        self.assertEqual(indels[0], "9:3I")

    def test_basic_cigar_string_insertion_with_context(self):

        cigar = "8M3I2M"
        query = "ACGGTTAAGTGTT"
        cutsite_window = 1
        context = True

        intBC, indels = alignment_utilities.parse_cigar(
            cigar,
            query,
            self.basic_ref,
            0,
            0,
            self.basic_barcode_interval,
            self.basic_cutsites,
            cutsite_window,
            context,
            context_size=1,
        )

        self.assertEqual(intBC, "GG")
        self.assertEqual(len(self.basic_cutsites), len(indels))
        self.assertEqual(indels[0], "A[9:3I]GTGT")

    def test_long_cigar_parsing_no_context(self):

        cigar = "110M1I174M"
        query = (
            "AATCCAGCTAGCTGTGCAGCATCTGACAAGCTCTATTCAACTGCAGTAATGCTACCTCGTA"
            "CTCACGCTTTCCAAGTGCTTGGCGTCGCATCTCGGTCCTTTGTACGCCGAAAAAATGGCCTG"
            "ACAACTAAGCTACGGCACGCTGCCATGTTGGGTCATAACGATATCTCTGGTTCATCCGTGAC"
            "CGAACATGTCATGGAGTAGCAGGAGCTATTAATTCGCGGAGGACAATGAGGTTCGTAGTCAC"
            "TGTCTTCCGCAATCGTACATCGCTCCTGCAGGTGGCCT"
        )
        cutsite_window = 12
        context = False

        intBC, indels = alignment_utilities.parse_cigar(
            cigar,
            query,
            self.long_ref,
            0,
            0,
            self.long_barcode_interval,
            self.long_cutsites,
            cutsite_window,
            context,
            context_size=0,
        )

        self.assertEqual(intBC, "ATCTGACAAGCTCT")

        expected_cuts = ["111:1I", "None", "None"]

        for i in range(len(indels)):

            self.assertEqual(indels[i], expected_cuts[i])

    def test_long_cigar_parsing_with_context(self):

        cigar = "110M1I174M"
        query = (
            "AATCCAGCTAGCTGTGCAGCATCTGACAAGCTCTATTCAACTGCAGTAATGCTACCTCGTA"
            "CTCACGCTTTCCAAGTGCTTGGCGTCGCATCTCGGTCCTTTGTACGCCGAAAAAATGGCCTG"
            "ACAACTAAGCTACGGCACGCTGCCATGTTGGGTCATAACGATATCTCTGGTTCATCCGTGAC"
            "CGAACATGTCATGGAGTAGCAGGAGCTATTAATTCGCGGAGGACAATGAGGTTCGTAGTCAC"
            "TGTCTTCCGCAATCGTACATCGCTCCTGCAGGTGGCCT"
        )
        cutsite_window = 12
        context = True

        intBC, indels = alignment_utilities.parse_cigar(
            cigar,
            query,
            self.long_ref,
            0,
            0,
            self.long_barcode_interval,
            self.long_cutsites,
            cutsite_window,
            context,
            context_size=5,
        )

        self.assertEqual(intBC, "ATCTGACAAGCTCT")

        expected_cuts = [
            "CGCCG[111:1I]AAAAAA",
            "GATAT[None]CTCTG",
            "ATTCG[None]CGGAG",
        ]

        for i in range(len(indels)):

            self.assertEqual(indels[i], expected_cuts[i])

    def test_intersite_deletion_parsing(self):

        cigar = "112M108D173M"
        query = (
            "AATCCAGCTAGCTGTGCAGCTTGTTTTAAACCAGATTCAACTGCAGTAATGCTACCTCGT"
            "ACTCACGCTTTCCAAGTGCTTGGCGTAGCATCTAGGTCCTAAGTACGCCGAACGGCTGACAATGC"
            "GGTTCGTAGTCACTGTCTACCGCAAACGTCAATCGCTCATCCAGGTGGCCAAGAGGGCACGTTTA"
            "CACACGCTGATCATCCTCGACTGTGCCCTCTAGTAGCCAGCCAGAGGTTGTGTGCCCCTCCCCCG"
            "GGCCGTCCGTGACCCTGGAAGGTGCCACTC"
        )
        cutsite_window = 12
        context = True

        intBC, indels = alignment_utilities.parse_cigar(
            cigar,
            query,
            self.long_ref,
            0,
            0,
            self.long_barcode_interval,
            self.long_cutsites,
            cutsite_window,
            context,
            context_size=5,
        )

        self.assertEqual(intBC, "TTGTTTTAAACCAG")

        expected_cuts = [
            "CCGAA[113:108D]CGGCT",
            "CCGAA[113:108D]CGGCT",
            "CCGAA[113:108D]CGGCT",
        ]

        for i in range(len(indels)):

            self.assertEqual(indels[i], expected_cuts[i])

    def test_complex_cigar_parsing_intersite_deletion(self):

        cigar = "98M13D55M54D130M"
        query = (
            "AATCCAGCTAGCTGTGCAGCTGACAGGGAAGCAAATTCAACTGCAGTAATGCTACCTCGT"
            "ACTCACGCTTTCCAAGTGCTTGGCGTCGCATCTCGGTCAAAAAGGCCTTACAACAAAGCTACTGA"
            "ACGAAGCAATGTTAGGACATAAAGATATCGGAGGACAATGATGTAAGTAGTCACTGTCTTCCTAA"
            "ATAGTCAAACTCTCCAGAACATTGAATAGAGGGCCCGAAAAAACCATCTCAAAAGCCTCTACACA"
            "GACTTCTAGAATACAAACAACTGATCTT"
        )
        cutsite_window = 12
        context = True

        intBC, indels = alignment_utilities.parse_cigar(
            cigar,
            query,
            self.long_ref,
            0,
            0,
            self.long_barcode_interval,
            self.long_cutsites,
            cutsite_window,
            context,
            context_size=5,
        )

        self.assertEqual(intBC, "TGACAGGGAAGCAA")

        expected_cuts = [
            "CGGTC[99:13D]AAAAA",
            "GATAT[167:54D]CGGAG",
            "GATAT[167:54D]CGGAG",
        ]

        for i in range(len(indels)):

            self.assertEqual(indels[i], expected_cuts[i])

    def test_call_alleles_function(self):

        molecule_table = cassiopeia.pp.call_alleles(
            self.alignment_dataframe,
            ref=self.basic_ref,
            barcode_interval=self.basic_barcode_interval,
            cutsite_locations=self.basic_cutsites,
            cutsite_width=1,
            context=False,
        )

        expected_columns = list(self.alignment_dataframe.columns) + [
            "r1",
            "allele",
            "intBC",
        ]

        for column in expected_columns:
            self.assertIn(column, molecule_table.columns)

        expected_indels = {
            "A_ATC_10": "None",
            "A_TTG_20": "7:2D",
            "B_ACC_10": "None",
            "C_CCA_10": "9:1I",
        }

        expected_intbcs = {
            "A_ATC_10": "GG",
            "A_TTG_20": "AT",
            "B_ACC_10": "CC",
            "C_CCA_10": "GA",
        }

        for _, row in molecule_table.iterrows():

            self.assertEqual(row.r1, expected_indels[row.readName])
            self.assertEqual(row.intBC, expected_intbcs[row.readName])

    def test_missing_data_in_allele_throws_warning(self):
        
        with self.assertWarns(PreprocessWarning):
            molecule_table = cassiopeia.pp.call_alleles(
                self.alignment_dataframe_with_missing,
                ref=self.basic_ref,
                barcode_interval=self.basic_barcode_interval,
                cutsite_locations=self.basic_cutsites,
                cutsite_width=1,
                context=False,
            )


if __name__ == "__main__":
    unittest.main()
