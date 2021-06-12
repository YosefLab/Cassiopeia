"""
Tests for correcting raw barcodes to a whitelist pipeline.py
"""
import os
import unittest
import tempfile

import pysam
import ngs_tools as ngs

from cassiopeia.preprocess import pipeline


class TestErrorCorrectBarcodes(unittest.TestCase):
    def setUp(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        test_files_path = os.path.join(dir_path, "test_files")

        self.bam_10xv3_fp = os.path.join(test_files_path, "10xv3_unmapped.bam")
        self.whitelist_10xv3 = ["TACGTCATCTCCTACG", "TTAGATCGTTAGAAAG"]
        self.bam_slideseq2_fp = os.path.join(
            test_files_path, "slideseq2_unmapped.bam"
        )
        self.whitelist_slideseq2 = ["CTTTGNTCAAAGTT"]

    def test_10xv3(self):
        bam_fp = pipeline.error_correct_barcodes(
            self.bam_10xv3_fp, tempfile.mkdtemp(), self.whitelist_10xv3
        )
        with pysam.AlignmentFile(bam_fp, "rb", check_sq=False) as f:
            alignments = list(f.fetch(until_eof=True))
        self.assertEqual(2, len(alignments))
        self.assertEqual(
            ["TACGTCATCTCCTACG", "TTAGATCGTTAGAAAG"],
            [al.get_tag("CB") for al in alignments],
        )

    def test_slideseq2(self):
        bam_fp = pipeline.error_correct_barcodes(
            self.bam_slideseq2_fp, tempfile.mkdtemp(), self.whitelist_slideseq2
        )
        with pysam.AlignmentFile(bam_fp, "rb", check_sq=False) as f:
            alignments = list(f.fetch(until_eof=True))
        self.assertEqual(2, len(alignments))
        self.assertEqual([True, False], [al.has_tag("CB") for al in alignments])
        self.assertEqual("CTTTGNTCAAAGTT", alignments[0].get_tag("CB"))


if __name__ == "__main__":
    unittest.main()
