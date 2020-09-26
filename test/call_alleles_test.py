"""
Tests for the allele calling in alignment_utilities.py and pipeline.py
"""
import unittest

import numpy as np
import pandas as pd

from cassiopeia.ProcessingPipeline.process import pipeline
from cassiopeia.ProcessingPipeline.process import alignment_utilities


class TestCallAlleles(unittest.TestCase):
    def setUp(self):

        self.basic_ref = "ACNNTTAATT"
        self.basic_barcode_interval = (2, 4)
        self.basic_cutsites = [7]

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
            self.basic_barcode_interval,
            self.basic_cutsites,
            cutsite_window,
            context,
        )

        self.assertEqual(intBC, "GG")
        self.assertEquals(len(self.basic_cutsites), len(indels))
        self.assertEquals(indels[0], "None")

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
            self.basic_barcode_interval,
            self.basic_cutsites,
            cutsite_window,
            context,
        )

        self.assertEqual(intBC, "GG")
        self.assertEquals(len(self.basic_cutsites), len(indels))
        self.assertEquals(indels[0], "6:2D")

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
            self.basic_barcode_interval,
            self.basic_cutsites,
            cutsite_window,
            context,
            context_size=context_size,
        )

        self.assertEqual(intBC, "GG")
        self.assertEquals(len(self.basic_cutsites), len(indels))
        self.assertEquals(indels[0], "T[6:2D]T")

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
            self.basic_barcode_interval,
            self.basic_cutsites,
            cutsite_window,
            context,
        )

        self.assertEqual(intBC, "GG")
        self.assertEquals(len(self.basic_cutsites), len(indels))
        self.assertEquals(indels[0], "8:3I")

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
            self.basic_barcode_interval,
            self.basic_cutsites,
            cutsite_window,
            context,
            context_size=1,
        )

        self.assertEqual(intBC, "GG")
        self.assertEquals(len(self.basic_cutsites), len(indels))
        self.assertEquals(indels[0], "A[8:3I]GTGT")


if __name__ == "__main__":
    unittest.main()
