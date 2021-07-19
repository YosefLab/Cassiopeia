"""
Tests for converting FASTQs to an unmapped BAM in pipeline.py
"""
import os
import unittest
import tempfile

import pysam
import ngs_tools as ngs

from cassiopeia.preprocess import pipeline


class TestConvertFastqsToUnmappedBam(unittest.TestCase):
    def setUp(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        test_files_path = os.path.join(dir_path, "test_files")

        self.fastq_10xv3_fps = [
            os.path.join(test_files_path, "10xv3_1.fastq.gz"),
            os.path.join(test_files_path, "10xv3_2.fastq.gz"),
        ]
        self.fastq_indropsv3_fps = [
            os.path.join(test_files_path, "indropsv3_1.fastq.gz"),
            os.path.join(test_files_path, "indropsv3_2.fastq.gz"),
            os.path.join(test_files_path, "indropsv3_3.fastq.gz"),
        ]
        self.fastq_slideseq2_fps = [
            os.path.join(test_files_path, "slideseq2_1.fastq.gz"),
            os.path.join(test_files_path, "slideseq2_2.fastq.gz"),
        ]

    def test_dropseq(self):
        # NOTE: using 10xv3 fastqs just for testing
        bam_fp = pipeline.convert_fastqs_to_unmapped_bam(
            self.fastq_10xv3_fps, "dropseq", tempfile.mkdtemp(), name="test"
        )
        with pysam.AlignmentFile(bam_fp, "rb", check_sq=False) as f:
            alignments = list(f.fetch(until_eof=True))
        self.assertEqual(2, len(alignments))
        self.assertEqual(
            [
                "M03718:773:000000000-JKHP3:1:1101:18272:1693",
                "M03718:773:000000000-JKHP3:1:1101:17963:1710",
            ],
            [al.query_name for al in alignments],
        )
        self.assertEqual(
            [
                read.sequence
                for read in ngs.fastq.Fastq(self.fastq_10xv3_fps[1])
            ],
            [al.query_sequence for al in alignments],
        )
        self.assertEqual(
            [
                read.qualities.string
                for read in ngs.fastq.Fastq(self.fastq_10xv3_fps[1])
            ],
            [
                pysam.array_to_qualitystring(al.query_qualities)
                for al in alignments
            ],
        )
        self.assertEqual(
            {
                ("UR", "TACGCCAA"),
                ("UY", "GGFECEE0"),
                ("CR", "TACGTCATCTCC"),
                ("CY", "1111AFAFFFBF"),
                ("RG", "test"),
            },
            set(alignments[0].get_tags()),
        )
        self.assertEqual(
            {
                ("UR", "AAACATTC"),
                ("UY", "FFGGBFGF"),
                ("CR", "TTAGATCGTTAG"),
                ("CY", "1>>11DFAFAAA"),
                ("RG", "test"),
            },
            set(alignments[1].get_tags()),
        )

    def test_10xv2(self):
        # NOTE: using 10xv3 fastqs just for testing
        bam_fp = pipeline.convert_fastqs_to_unmapped_bam(
            self.fastq_10xv3_fps, "10xv2", tempfile.mkdtemp(), name="test"
        )
        with pysam.AlignmentFile(bam_fp, "rb", check_sq=False) as f:
            alignments = list(f.fetch(until_eof=True))
        self.assertEqual(2, len(alignments))
        self.assertEqual(
            [
                "M03718:773:000000000-JKHP3:1:1101:18272:1693",
                "M03718:773:000000000-JKHP3:1:1101:17963:1710",
            ],
            [al.query_name for al in alignments],
        )
        self.assertEqual(
            [
                read.sequence
                for read in ngs.fastq.Fastq(self.fastq_10xv3_fps[1])
            ],
            [al.query_sequence for al in alignments],
        )
        self.assertEqual(
            [
                read.qualities.string
                for read in ngs.fastq.Fastq(self.fastq_10xv3_fps[1])
            ],
            [
                pysam.array_to_qualitystring(al.query_qualities)
                for al in alignments
            ],
        )
        self.assertEqual(
            {
                ("UR", "CCAAAACAGT"),
                ("UY", "CEE0C0BA0D"),
                ("CR", "TACGTCATCTCCTACG"),
                ("CY", "1111AFAFFFBFGGFE"),
                ("RG", "test"),
            },
            set(alignments[0].get_tags()),
        )
        self.assertEqual(
            {
                ("UR", "ATTCCTGAGT"),
                ("UY", "BFGFGFF10F"),
                ("CR", "TTAGATCGTTAGAAAC"),
                ("CY", "1>>11DFAFAAAFFGG"),
                ("RG", "test"),
            },
            set(alignments[1].get_tags()),
        )

    def test_10xv3(self):
        bam_fp = pipeline.convert_fastqs_to_unmapped_bam(
            self.fastq_10xv3_fps, "10xv3", tempfile.mkdtemp(), name="test"
        )
        with pysam.AlignmentFile(bam_fp, "rb", check_sq=False) as f:
            alignments = list(f.fetch(until_eof=True))
        self.assertEqual(2, len(alignments))
        self.assertEqual(
            [
                "M03718:773:000000000-JKHP3:1:1101:18272:1693",
                "M03718:773:000000000-JKHP3:1:1101:17963:1710",
            ],
            [al.query_name for al in alignments],
        )
        self.assertEqual(
            [
                read.sequence
                for read in ngs.fastq.Fastq(self.fastq_10xv3_fps[1])
            ],
            [al.query_sequence for al in alignments],
        )
        self.assertEqual(
            [
                read.qualities.string
                for read in ngs.fastq.Fastq(self.fastq_10xv3_fps[1])
            ],
            [
                pysam.array_to_qualitystring(al.query_qualities)
                for al in alignments
            ],
        )
        self.assertEqual(
            {
                ("UR", "CCAAAACAGTTT"),
                ("UY", "CEE0C0BA0DFG"),
                ("CR", "TACGTCATCTCCTACG"),
                ("CY", "1111AFAFFFBFGGFE"),
                ("RG", "test"),
            },
            set(alignments[0].get_tags()),
        )
        self.assertEqual(
            {
                ("UR", "ATTCCTGAGTCA"),
                ("UY", "BFGFGFF10FG1"),
                ("CR", "TTAGATCGTTAGAAAC"),
                ("CY", "1>>11DFAFAAAFFGG"),
                ("RG", "test"),
            },
            set(alignments[1].get_tags()),
        )

    def test_indropsv3(self):
        bam_fp = pipeline.convert_fastqs_to_unmapped_bam(
            self.fastq_indropsv3_fps,
            "indropsv3",
            tempfile.mkdtemp(),
            name="test",
        )
        with pysam.AlignmentFile(bam_fp, "rb", check_sq=False) as f:
            alignments = list(f.fetch(until_eof=True))
        self.assertEqual(2, len(alignments))
        self.assertEqual(
            [
                "M03718:773:000000000-JKHP3:1:1101:18272:1693",
                "M03718:773:000000000-JKHP3:1:1101:17963:1710",
            ],
            [al.query_name for al in alignments],
        )
        self.assertEqual(
            [
                read.sequence
                for read in ngs.fastq.Fastq(self.fastq_10xv3_fps[1])
            ],
            [al.query_sequence for al in alignments],
        )
        self.assertEqual(
            [
                read.qualities.string
                for read in ngs.fastq.Fastq(self.fastq_10xv3_fps[1])
            ],
            [
                pysam.array_to_qualitystring(al.query_qualities)
                for al in alignments
            ],
        )
        self.assertEqual(
            {
                ("UR", "CCAAAA"),
                ("UY", "FFBFGG"),
                ("CR", "TACGTCATCTCCTACG"),
                ("CY", "1111AFAF1111AFAF"),
                ("RG", "test"),
            },
            set(alignments[0].get_tags()),
        )
        self.assertEqual(
            {
                ("UR", "TTAGAA"),
                ("UY", "FAAAFF"),
                ("CR", "TTAGATCGTTAGATCG"),
                ("CY", "1>>11DFA1>>11DFA"),
                ("RG", "test"),
            },
            set(alignments[1].get_tags()),
        )

    def test_slideseq2(self):
        bam_fp = pipeline.convert_fastqs_to_unmapped_bam(
            self.fastq_slideseq2_fps,
            "slideseq2",
            tempfile.mkdtemp(),
            name="test",
        )
        with pysam.AlignmentFile(bam_fp, "rb", check_sq=False) as f:
            alignments = list(f.fetch(until_eof=True))
        self.assertEqual(2, len(alignments))
        self.assertEqual(
            [
                "NB501583:801:H7JLTBGXH:1:11101:20912:1050",
                "NB501583:801:H7JLTBGXH:1:11101:8670:1050",
            ],
            [al.query_name for al in alignments],
        )
        self.assertEqual(
            [
                read.sequence
                for read in ngs.fastq.Fastq(self.fastq_slideseq2_fps[1])
            ],
            [al.query_sequence for al in alignments],
        )
        self.assertEqual(
            [
                read.qualities.string
                for read in ngs.fastq.Fastq(self.fastq_slideseq2_fps[1])
            ],
            [
                pysam.array_to_qualitystring(al.query_qualities)
                for al in alignments
            ],
        )
        self.assertEqual(
            {
                ("UR", "TTTTTTTTT"),
                ("UY", "EEEEEEEEE"),
                ("CR", "CTTTGNTCAATGTT"),
                ("CY", "AAAAA#EEAEEEEE"),
                ("RG", "test"),
            },
            set(alignments[0].get_tags()),
        )
        self.assertEqual(
            {
                ("UR", "AGTGTCTCA"),
                ("UY", "EAEAEAEEE"),
                ("CR", "CTCTTNATCCTCAT"),
                ("CY", "AAAAA#EEE/EAE/"),
                ("RG", "test"),
            },
            set(alignments[1].get_tags()),
        )


if __name__ == "__main__":
    unittest.main()
