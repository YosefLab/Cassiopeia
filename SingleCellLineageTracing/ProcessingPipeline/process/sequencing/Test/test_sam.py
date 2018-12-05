import unittest
import pysam
import Sequencing.sam as sam

MATCH = sam.BAM_CMATCH
DEL = sam.BAM_CDEL
INS = sam.BAM_CINS
SOFT_CLIP = sam.BAM_CSOFT_CLIP

class TestCropAlToRefInt(unittest.TestCase):

    def test_simple(self):
        al = pysam.AlignedSegment()
        al.reference_start = 100
        al.cigar = [(MATCH, 100)]
        al.seq = 'A' * 100

        cropped = pysam.AlignedSegment()
        cropped.reference_start = 150
        cropped.cigar = [(SOFT_CLIP, 50), (MATCH, 10), (SOFT_CLIP, 40)]
        cropped.seq = 'A' * 100

        output = sam.crop_al_to_ref_int(al, 150, 159)
        self.assertEqual(output, cropped)

        output = sam.crop_al_to_ref_int(al, 150, 158)
        self.assertNotEqual(output, cropped)

    def test_ends_in_deletion(self):
        al = pysam.AlignedSegment()
        al.reference_start = 100
        al.cigar = [(MATCH, 50), (DEL, 10), (MATCH, 50)]
        al.seq = 'A' * 100
        
        cropped = pysam.AlignedSegment()
        cropped.reference_start = 100
        cropped.cigar = [(MATCH, 50), (SOFT_CLIP, 50)]
        cropped.seq = 'A' * 100
        
        output = sam.crop_al_to_ref_int(al, 100, 155)
        self.assertEqual(output, cropped)
        
        output = sam.crop_al_to_ref_int(al, 100, 160)
        self.assertNotEqual(output, cropped)
    
    def test_ends_just_after_deletion(self):
        al = pysam.AlignedSegment()
        al.reference_start = 100
        al.cigar = [(MATCH, 50), (DEL, 10), (MATCH, 50)]
        al.seq = 'A' * 100
        
        cropped = pysam.AlignedSegment()
        cropped.reference_start = 100
        cropped.cigar = [(MATCH, 50), (DEL, 10), (MATCH, 1), (SOFT_CLIP, 49)]
        cropped.seq = 'A' * 100
        
        output = sam.crop_al_to_ref_int(al, 100, 160)
        self.assertEqual(output, cropped)
        
        output = sam.crop_al_to_ref_int(al, 100, 155)
        self.assertNotEqual(output, cropped)
    
    def test_starts_in_deletion(self):
        al = pysam.AlignedSegment()
        al.reference_start = 100
        al.cigar = [(MATCH, 50), (DEL, 10), (MATCH, 50)]
        al.seq = 'A' * 100
        
        cropped = pysam.AlignedSegment()
        cropped.reference_start = 160
        cropped.cigar = [(SOFT_CLIP, 50), (MATCH, 50)]
        cropped.seq = 'A' * 100
        
        output = sam.crop_al_to_ref_int(al, 155, 210)
        self.assertEqual(output, cropped)
        
        output = sam.crop_al_to_ref_int(al, 149, 210)
        self.assertNotEqual(output, cropped)
    
    def test_starts_just_before_deletion(self):
        al = pysam.AlignedSegment()
        al.reference_start = 100
        al.cigar = [(MATCH, 50), (DEL, 10), (MATCH, 50)]
        al.seq = 'A' * 100
        
        cropped = pysam.AlignedSegment()
        cropped.reference_start = 149
        cropped.cigar = [(SOFT_CLIP, 49), (MATCH, 1), (DEL, 10), (MATCH, 50)]
        cropped.seq = 'A' * 100
        
        output = sam.crop_al_to_ref_int(al, 149, 210)
        self.assertEqual(output, cropped)
        
        output = sam.crop_al_to_ref_int(al, 150, 210)
        self.assertNotEqual(output, cropped)
    
    def test_ends_just_before_insertion(self):
        al = pysam.AlignedSegment()
        al.reference_start = 100
        al.cigar = [(MATCH, 50), (INS, 10), (MATCH, 50)]
        al.seq = 'A' * 110
        
        cropped = pysam.AlignedSegment()
        cropped.reference_start = 100
        cropped.cigar = [(MATCH, 50), (SOFT_CLIP, 60)]
        cropped.seq = 'A' * 110
        
        output = sam.crop_al_to_ref_int(al, 100, 149)
        self.assertEqual(output, cropped)
        
        output = sam.crop_al_to_ref_int(al, 100, 150)
        self.assertNotEqual(output, cropped)

    def test_ends_just_after_insertion(self):
        al = pysam.AlignedSegment()
        al.reference_start = 100
        al.cigar = [(MATCH, 50), (INS, 10), (MATCH, 50)]
        al.seq = 'A' * 110
        
        cropped = pysam.AlignedSegment()
        cropped.reference_start = 100
        cropped.cigar = [(MATCH, 50), (INS, 10), (MATCH, 1), (SOFT_CLIP, 49)]
        cropped.seq = 'A' * 110
        
        output = sam.crop_al_to_ref_int(al, 100, 150)
        self.assertEqual(output, cropped)
        
        output = sam.crop_al_to_ref_int(al, 100, 149)
        self.assertNotEqual(output, cropped)
    
    def test_starts_just_before_insertion(self):
        al = pysam.AlignedSegment()
        al.reference_start = 100
        al.cigar = [(MATCH, 50), (INS, 10), (MATCH, 50)]
        al.seq = 'A' * 110
        
        cropped = pysam.AlignedSegment()
        cropped.reference_start = 149
        cropped.cigar = [(SOFT_CLIP, 49), (MATCH, 1), (INS, 10), (MATCH, 50)]
        cropped.seq = 'A' * 110
        
        output = sam.crop_al_to_ref_int(al, 149, 210)
        self.assertEqual(output, cropped)
        
        output = sam.crop_al_to_ref_int(al, 150, 210)
        self.assertNotEqual(output, cropped)
    
    def test_starts_just_after_insertion(self):
        al = pysam.AlignedSegment()
        al.reference_start = 100
        al.cigar = [(MATCH, 50), (INS, 10), (MATCH, 50)]
        al.seq = 'A' * 110
        
        cropped = pysam.AlignedSegment()
        cropped.reference_start = 150
        cropped.cigar = [(SOFT_CLIP, 60), (MATCH, 50)]
        cropped.seq = 'A' * 110
        
        output = sam.crop_al_to_ref_int(al, 150, 210)
        self.assertEqual(output, cropped)
        
        output = sam.crop_al_to_ref_int(al, 149, 210)
        self.assertNotEqual(output, cropped)

if __name__ == '__main__':
    unittest.main()
