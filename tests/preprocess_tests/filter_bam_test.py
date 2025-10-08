"""
Tests for correcting raw barcodes to a whitelist pipeline.py
"""
import os
import tempfile
import unittest

import pysam
from cassiopeia.preprocess import pipeline


class TestFilterBam(unittest.TestCase):
    def setUp(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        test_files_path = os.path.join(dir_path, "test_files")

        self.bam_10xv3_fp = os.path.join(test_files_path, "10xv3_unmapped.bam")

    def test_filter(self):
        bam_fp = pipeline.filter_bam(self.bam_10xv3_fp, tempfile.mkdtemp(), 10)
        with pysam.AlignmentFile(bam_fp, "rb", check_sq=False) as f:
            alignments = list(f.fetch(until_eof=True))
        self.assertEqual(len(alignments), 2)

        bam_fp = pipeline.filter_bam(self.bam_10xv3_fp, tempfile.mkdtemp(), 20)
        with pysam.AlignmentFile(bam_fp, "rb", check_sq=False) as f:
            alignments = list(f.fetch(until_eof=True))
        self.assertEqual(len(alignments), 0)


if __name__ == "__main__":
    unittest.main()
