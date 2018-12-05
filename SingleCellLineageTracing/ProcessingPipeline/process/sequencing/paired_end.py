import numpy as np
import pysam
import logging
from collections import Counter
from . import sam
from . import utilities
from . import fastq

def keep_same_names(R1_aligned, R2_aligned):
    return R1_aligned.qname, R2_aligned.qname

def is_disoriented(R1_aligned, R2_aligned):
    disoriented = False
    if R1_aligned.is_reverse:
        if R2_aligned.is_reverse:
            disoriented = True
        if R1_aligned.pos < R2_aligned.pos:
            disoriented = True
    else:
        if not R2_aligned.is_reverse:
            disoriented = True
        if R2_aligned.pos < R1_aligned.pos:
            disoriented = True

    return disoriented

def get_reference_extent(R1_m, R2_m):
    if R1_m.tid != R2_m.tid:
        raise ValueError(R1_m, R2_m)
    
    min_start = min(R1_m.reference_start, R2_m.reference_start)
    max_end = max(R1_m.reference_end, R2_m.reference_end)

    # No +1 needed in tlen subtraction because reference_end points
    # one past the end.
    tlen = max_end - min_start

    return tlen

def is_discordant(R1_m, R2_m, max_insert_length):
    discordant = False
    if R1_m.tid != R2_m.tid:
        discordant = True
    else:
        tlen = get_reference_extent(R1_m, R2_m)
        if tlen > max_insert_length:
            discordant = True

    return discordant

def remove_soft_clipping(mapping):
    if mapping.cigar[0][0] == sam.BAM_CSOFT_CLIP:
        clipped_from_start = mapping.cigar[0][1]
        cigar_start_index = 1
        cigar_start = mapping.cigar[:1]
    else:
        clipped_from_start = 0
        cigar_start_index = 0
        cigar_start = []
    
    if mapping.cigar[-1][0] == sam.BAM_CSOFT_CLIP:
        clipped_from_end = mapping.cigar[-1][1]
        cigar_end_index = -1
        cigar_end = mapping.cigar[-1:]
    else:
        clipped_from_end = 0
        cigar_end_index = None
        cigar_end = []

    seq = mapping.seq
    qual = mapping.qual
    from_start_slice = slice(None, clipped_from_start)
    from_end_slice = slice(len(seq) - clipped_from_end, None)
    remaining_slice = slice(clipped_from_start, len(seq) - clipped_from_end)

    clipped = {
        'from_start': {
            'seq': seq[from_start_slice],
            'qual': qual[from_start_slice],
            'cigar': cigar_start,
        },
        'from_end': {
            'seq': seq[from_end_slice],
            'qual': qual[from_end_slice],
            'cigar': cigar_end,
        },
    }

    mapping.cigar = mapping.cigar[cigar_start_index:cigar_end_index]
    mapping.seq = seq[remaining_slice]
    mapping.qual = qual[remaining_slice]

    return clipped

def combine_paired_mappings(R1_mapping, R2_mapping, verbose=False):
    ''' Takes two pysam mappings representing opposite ends of a fragment and
    combines them into one mapping, (ab)using BAM_CREF_SKIP to bridge the gap
    (if any) between them.
    '''
    R1_strand = sam.get_strand(R1_mapping)

    if R1_strand == '+':
        left_mapping, right_mapping = R1_mapping, R2_mapping
    elif R1_strand == '-':
        left_mapping, right_mapping = R2_mapping, R1_mapping
                
    # Soft-clipping at the 3' end of a read should only happen if this is
    # read-through into soft-clipping at the 5' end of the other read.
    # If there is non-physical soft-clipping in this pair, give up now.

    if left_mapping.cigar[-1][0] == sam.BAM_CSOFT_CLIP and \
       left_mapping.reference_end != right_mapping.reference_end:
        return False
    if right_mapping.cigar[0][0] == sam.BAM_CSOFT_CLIP and \
       right_mapping.reference_start != left_mapping.reference_start:
        return False

    # Otherwise, remove all soft-clipping from the mappings, storing the 5'
    # soft-clipped seq and quals from both reads to add back at the end.
    
    left_clipped = remove_soft_clipping(left_mapping)
    right_clipped = remove_soft_clipping(right_mapping)

    left_md = dict(left_mapping.tags)['MD']
    right_md = dict(right_mapping.tags)['MD']

    right_aligned_pairs = sam.cigar_to_aligned_pairs(right_mapping.cigar, right_mapping.reference_start)

    right_after_overlap_pair_index = len(right_aligned_pairs)
    for i, (read, ref) in enumerate(right_aligned_pairs):
        if ref != None and ref >= left_mapping.aend:
            right_after_overlap_pair_index = i
            break
    
    right_overlap_pairs = right_aligned_pairs[:right_after_overlap_pair_index]
    right_after_overlap_pairs = right_aligned_pairs[right_after_overlap_pair_index:]
    
    right_reads_after = [read for read, ref in right_after_overlap_pairs if read != None and read != 's']
    right_refs_after = [ref for read, ref in right_after_overlap_pairs if ref != None]
    
    right_overlap_cigar = sam.aligned_pairs_to_cigar(right_overlap_pairs)
    right_after_overlap_cigar = sam.aligned_pairs_to_cigar(right_after_overlap_pairs)
    right_after_overlap_md = sam.truncate_md_string_from_beginning(right_md, len(right_refs_after))
    
    right_after_overlap_read_start = len(right_mapping.seq) - len(right_reads_after)

    right_overlap_seq = right_mapping.seq[:right_after_overlap_read_start] 
    right_overlap_qual = right_mapping.query_qualities[:right_after_overlap_read_start] 

    right_after_overlap_seq = right_mapping.seq[right_after_overlap_read_start:]
    right_after_overlap_qual = right_mapping.qual[right_after_overlap_read_start:]
    
    
    left_aligned_pairs = sam.cigar_to_aligned_pairs(left_mapping.cigar, left_mapping.reference_start)

    left_before_overlap_pair_index = -1
    for i, (read, ref) in list(enumerate(left_aligned_pairs))[::-1]:
        if ref != None and ref < right_mapping.pos:
            left_before_overlap_pair_index = i
            break

    left_overlap_pairs = left_aligned_pairs[left_before_overlap_pair_index + 1:]
    left_before_overlap_pairs = left_aligned_pairs[:left_before_overlap_pair_index + 1]

    left_reads_before = [read for read, ref in left_before_overlap_pairs if read != None and read != 's']
    left_refs_before = [ref for read, ref in left_before_overlap_pairs if ref != None]
    
    left_overlap_cigar = sam.aligned_pairs_to_cigar(left_overlap_pairs)
    left_before_overlap_cigar = sam.aligned_pairs_to_cigar(left_before_overlap_pairs)
    left_before_overlap_md = sam.truncate_md_string_up_to(left_md, len(left_refs_before))
    
    left_overlap_read_start = len(left_reads_before)
    left_overlap_seq = left_mapping.seq[left_overlap_read_start:] 
    left_overlap_qual = left_mapping.query_qualities[left_overlap_read_start:] 

    left_before_overlap_seq = left_mapping.seq[:left_overlap_read_start]
    left_before_overlap_qual = left_mapping.qual[:left_overlap_read_start]

    if left_overlap_pairs or right_overlap_pairs:
        gap_length = 0

        left_has_splicing = sam.contains_splicing(left_mapping)
        right_has_splicing = sam.contains_splicing(right_mapping)

        if left_overlap_cigar == right_overlap_cigar:
            # If the two mappings agree about the location of indels in their overlap,
            # use the seq from the mapping with the higher average quality in the
            # overlap.
            left_mean_qual = np.mean(left_overlap_qual)
            right_mean_qual = np.mean(right_overlap_qual)

            if left_mean_qual > right_mean_qual:
                use_overlap_from = 'left'
            else:
                use_overlap_from = 'right'
        elif left_has_splicing != right_has_splicing:
            # A temporary(?) heuristic - if one read has splicing and the other
            # doesn't, use the overlap from the one with splicing under the
            # assumption that the other just has a few bases overhanging the
            # splice junction.
            if left_has_splicing:
                use_overlap_from = 'left'
            else:
                use_overlap_from = 'right'
        else:
            # If the two mappings disagree about the location of indels in their overlap,
            # we need a heuristic for picking which mapping we believe reflects the
            # true structure of the input fragment. The most innocuous explanation
            # is that a 'true' indel happened to lie close to the edge of one of the
            # mappings. A more problematic situation is a 'false' indel (that is,
            # produced during cluster generation or sequencing-by-synthesis, NOT
            # template production). Our strategy is: realign the overlapping part of
            # left mapping starting from the left edge of the overlap according to the
            # cigar of the right mapping and realign the overlapping part of the right
            # mapping starting from the right edge of the overlap according to the cigar
            # of the left mapping. Count the number of mismatches produced by each.
            # If the left overlap can accomodate the right cigar with fewer mismatches,
            # use the right cigar and seq. If the right overlap can accomodate the left
            # cigar with fewer mismatches, use the left cigar and seq.

            # The leftmost aligned_pair from the right mapping is guaranteed by the
            # mapping process to not involve a gap.
            _, overlap_ref_start = right_overlap_pairs[0]
            # Similarly, the rightmost aligned_pair from the left mapping can't be a
            # gap.
            _, overlap_ref_end = left_overlap_pairs[-1]

            realigned_left_cigar = sam.truncate_cigar_blocks_up_to(right_mapping.cigar, len(left_overlap_seq))
            realigned_right_cigar = sam.truncate_cigar_blocks_from_beginning(left_mapping.cigar, len(right_overlap_seq))

            ref_dict = sam.merge_ref_dicts(sam.ref_dict_from_mapping(left_mapping),
                                           sam.ref_dict_from_mapping(right_mapping),
                                          )

            try:
                left_using_right_mismatches = realigned_mismatches(left_overlap_seq, overlap_ref_start, realigned_left_cigar, ref_dict)
                right_using_left_mismatches = realigned_mismatches_backwards(right_overlap_seq, overlap_ref_end, realigned_right_cigar, ref_dict)
            except (ValueError, TypeError):
                print(left_mapping)
                print(right_mapping)
                raise
            
            if verbose:
                logging.info('disagreements in {0}'.format(left_mapping.qname))
                logging.info('left overlap cigar is  {0}'.format(str(left_overlap_cigar)))
                logging.info('right overlap cigar is {0}'.format(str(right_overlap_cigar)))
                logging.info('left_using_right_mismatches - {0}'.format(len(left_using_right_mismatches)))
                logging.info('right_using_left_mismatches - {0}'.format(len(right_using_left_mismatches)))

            if len(left_using_right_mismatches) < len(right_using_left_mismatches):
                use_overlap_from = 'right'
            elif len(right_using_left_mismatches) < len(left_using_right_mismatches):
                use_overlap_from = 'left'
            else:
                logging.info('disagreements in {0}'.format(left_mapping.qname))
                logging.info('left overlap cigar is  {0}'.format(str(left_overlap_cigar)))
                logging.info('right overlap cigar is {0}'.format(str(right_overlap_cigar)))
                logging.info('left_using_right_mismatches - {0}'.format(len(left_using_right_mismatches)))
                logging.info('right_using_left_mismatches - {0}'.format(len(right_using_left_mismatches)))
                logging.info('ambiguous disagreement')
                return False

    else:
        gap_length = right_mapping.pos - left_mapping.aend
        # It doesn't matter what use_overlap_from is set to; there is no overlap
        use_overlap_from = 'left'
        
    combined_mapping = pysam.AlignedRead()
    combined_mapping.qname = left_mapping.qname
    combined_mapping.tid = left_mapping.tid
    combined_mapping.mapq = min(left_mapping.mapq, right_mapping.mapq)
    combined_mapping.rnext = -1
    combined_mapping.pnext = -1
    combined_mapping.pos = left_mapping.pos

    if R1_strand == '-':
        combined_mapping.is_reverse = True

    gap_cigar = [(sam.BAM_CREF_SKIP, gap_length)]
    
    if use_overlap_from == 'left':
        combined_mapping.seq = left_mapping.seq + right_after_overlap_seq
        combined_mapping.qual = left_mapping.qual + right_after_overlap_qual
        combined_mapping.cigar = left_mapping.cigar + gap_cigar + right_after_overlap_cigar
    
        combined_md = sam.combine_md_strings(left_md, right_after_overlap_md)
        combined_mapping.setTag('MD', combined_md)

        overlap_seq_tag = right_overlap_seq
        overlap_qual_tag = fastq.encode_sanger(right_overlap_qual)

    elif use_overlap_from == 'right':
        combined_mapping.seq = left_before_overlap_seq + right_mapping.seq
        combined_mapping.qual = left_before_overlap_qual + right_mapping.qual
        combined_mapping.cigar = left_before_overlap_cigar + gap_cigar + right_mapping.cigar

        combined_md = sam.combine_md_strings(left_before_overlap_md, right_md)
        combined_mapping.setTag('MD', combined_md)
        
        overlap_seq_tag = left_overlap_seq
        overlap_qual_tag = fastq.encode_sanger(left_overlap_qual)

    if len(overlap_seq_tag) > 0:
        # Having empty tags causes problems, so don't create them.
        combined_mapping.setTag('Xs', overlap_seq_tag)
        combined_mapping.setTag('Xq', overlap_qual_tag)
        combined_mapping.setTag('Xw', use_overlap_from)

    qual = combined_mapping.qual
    seq = combined_mapping.seq
    cigar = combined_mapping.cigar
    before = left_clipped['from_start']
    after = right_clipped['from_end']
    combined_mapping.cigar = before['cigar'] + cigar + after['cigar']
    combined_mapping.seq = before['seq'] + seq + after['seq']
    combined_mapping.qual = before['qual'] + qual + after['qual']

    return combined_mapping

def realigned_mismatches(seq, start, realigned_cigar, ref_dict):
    realigned_pairs = sam.cigar_to_aligned_pairs(realigned_cigar, start)
    mismatches = []
    for read_position, ref_position in realigned_pairs:
        if read_position != None and read_position != 's' and ref_position != None and ref_position != 'S':
            read_base = seq[read_position]
            ref_base = ref_dict[ref_position]
            if read_base != ref_base:
                mismatches.append((read_position, ref_position, read_base, ref_base))

    return mismatches

def realigned_mismatches_backwards(seq, end, realigned_cigar, ref_dict):
    realigned_pairs = sam.cigar_to_aligned_pairs_backwards(realigned_cigar, end, len(seq))
    mismatches = []
    for read_position, ref_position in realigned_pairs:
        if read_position != None and read_position != 's' and ref_position != None and ref_position != 'S':
            read_base = seq[read_position]
            ref_base = ref_dict[ref_position]
            if read_base != ref_base:
                mismatches.append((read_position, ref_position, read_base, ref_base))

    return mismatches

def find_skip_index_in_combined(combined_mapping):
    # There should be exactly one BAM_CREF_SKIP, possibly of length 0,
    # separating R1 from R2.
    skip_indices = [i for i, (op, length) in enumerate(combined_mapping.cigar)
                    if op == sam.BAM_CREF_SKIP]
    if len(skip_indices) != 1:
        raise ValueError(str(combined_mapping))
    skip_index = skip_indices.pop()
    return skip_index

def remove_length_zero_skip(combined_mapping):
    skip_index = find_skip_index_in_combined(combined_mapping)
    _, skip_length = combined_mapping.cigar[skip_index]
    if skip_length == 0:
        removed_cigar = remove_cigar_op(combined_mapping.cigar, skip_index)
    else:
        removed_cigar = combined_mapping.cigar
    return removed_cigar

def extract_seqs_from_combined(combined_mapping,
                               include_overlap=True,
                               remove_soft_clipped=True,
                               flip_if_reverse=True,
                              ):
    ''' Separates out the R1 and R2 seq and quals that went into a
    combined_mapping.
    '''
    strand = sam.get_strand(combined_mapping)
    tags = dict(combined_mapping.tags)
    if 'Xs' not in tags:
        tags['Xs'] = ''
        tags['Xq'] = ''
        tags['Xw'] = 'left'

    skip_index = find_skip_index_in_combined(combined_mapping)

    left_cigar = combined_mapping.cigar[:skip_index]
    right_cigar = combined_mapping.cigar[skip_index + 1:]

    left_length = sam.total_read_nucs(left_cigar)

    left_seq = combined_mapping.seq[:left_length]
    left_qual = combined_mapping.qual[:left_length]

    right_seq = combined_mapping.seq[left_length:]
    right_qual = combined_mapping.qual[left_length:]

    if remove_soft_clipped:
        first_left_op, first_left_length = left_cigar[0]
        if first_left_op == sam.BAM_CSOFT_CLIP:
            left_seq = left_seq[first_left_length:]
            left_qual = left_qual[first_left_length:]
        
        last_right_op, last_right_length = right_cigar[-1]
        if last_right_op == sam.BAM_CSOFT_CLIP:
            right_seq = right_seq[:-last_right_length]
            right_qual = right_qual[:-last_right_length]

    if include_overlap:
        if tags['Xw'] == 'left':
            # Overlapping sequence in the combined read reflects that from the 
            # left mapping, so the overlap from the right was stored in the Xs
            # and Xq tags.
            right_seq = tags['Xs'] + right_seq
            right_qual = tags['Xq'] + right_qual
        elif tags['Xw'] == 'right':
            # Overlapping sequence in the combined read reflects that from the 
            # right mapping, so the overlap from the left was stored in the Xs
            # and Xq tags.
            left_seq = left_seq + tags['Xs']
            left_qual = left_qual + tags['Xq']

    if strand == '+':
        R1_seq, R1_qual = left_seq, left_qual
        R2_seq, R2_qual = right_seq, right_qual

        if flip_if_reverse:
            R2_seq = utilities.reverse_complement(R2_seq)
            R2_qual = R2_qual[::-1]

    elif strand == '-':
        R1_seq, R1_qual = right_seq, right_qual
        R2_seq, R2_qual = left_seq, left_qual
        
        if flip_if_reverse:
            R1_seq = utilities.reverse_complement(R1_seq)
            R1_qual = R1_qual[::-1]

    return R1_seq, R1_qual, R2_seq, R2_qual

def extract_reads_from_combined(combined_mapping):
    R1_seq, R1_qual, R2_seq, R2_qual = extract_seqs_from_combined(combined_mapping, remove_soft_clipped=False)
    R1 = fastq.Read(combined_mapping.qname, R1_seq, R1_qual)
    R2 = fastq.Read(combined_mapping.qname, R2_seq, R2_qual)
    return R1, R2

def split_combined_mapping(combined_mapping, remove_soft_clipped=True):
    ''' Split a combined_mapping into non-overlapping mappings. '''
    R1_mapping = pysam.AlignedRead()
    R1_mapping.is_read1 = True
    R1_mapping.tid = combined_mapping.tid
    R1_mapping.qname = combined_mapping.qname

    R2_mapping = pysam.AlignedRead()
    R2_mapping.is_read2 = True
    R2_mapping.tid = combined_mapping.tid
    R2_mapping.qname = combined_mapping.qname

    skip_index = find_skip_index_in_combined(combined_mapping)

    left_cigar = combined_mapping.cigar[:skip_index]
    right_cigar = combined_mapping.cigar[skip_index + 1:]

    if remove_soft_clipped:
        first_left_op, first_left_length = left_cigar[0]
        if first_left_op == sam.BAM_CSOFT_CLIP:
            left_cigar = left_cigar[1:]
        
        last_right_op, last_right_length = right_cigar[-1]
        if last_right_op == sam.BAM_CSOFT_CLIP:
            right_cigar = right_cigar[:-1]
    
    combined_md = dict(combined_mapping.tags)['MD']
    left_ref_bases = sam.total_reference_nucs(left_cigar)
    right_ref_bases = sam.total_reference_nucs(right_cigar)

    _, gap = combined_mapping.cigar[skip_index]
    left_pos = combined_mapping.pos
    right_pos = left_pos + left_ref_bases + gap

    left_md = sam.truncate_md_string_up_to(combined_md, left_ref_bases)
    right_md = sam.truncate_md_string_from_beginning(combined_md, right_ref_bases)
    
    strand = sam.get_strand(combined_mapping)
    if strand == '+':
        R1_mapping.cigar = left_cigar
        R1_mapping.setTag('MD', left_md)
        
        R2_mapping.cigar = right_cigar
        R2_mapping.setTag('MD', right_md)

        R1_mapping.pos = left_pos
        R2_mapping.pos = right_pos

        R2_mapping.is_reverse = True
    elif strand == '-':
        R1_mapping.cigar = right_cigar
        R1_mapping.setTag('MD', right_md)
        
        R2_mapping.cigar = left_cigar
        R2_mapping.setTag('MD', left_md)

        R1_mapping.pos = right_pos
        R2_mapping.pos = left_pos
        R1_mapping.is_reverse = True

    R1_seq, R1_qual, R2_seq, R2_qual = extract_seqs_from_combined(combined_mapping,
                                                                  include_overlap=False,
                                                                  remove_soft_clipped=remove_soft_clipped,
                                                                  flip_if_reverse=False,
                                                                 )
    if R1_seq != '':
        R1_mapping.seq = R1_seq
        R1_mapping.qual = R1_qual
    if R2_seq != '':
        R2_mapping.seq = R2_seq
        R2_mapping.qual = R2_qual
    
    return R1_mapping, R2_mapping

def remove_cigar_op(cigar, index):
    op, length = cigar[index]
    if length != 0:
        raise ValueError('Attempted to remove cigar op with nonzero length')

    if index == 0:
        removed = cigar[1:]
    elif index == len(cigar) - 1:
        removed = ciger[:-1]
    else:
        before_op, before_length = cigar[index - 1]
        after_op, after_length = cigar[index + 1]
        if before_op == after_op:
            interface = [(before_op, before_length + after_length)]
        else:
            interface = [(before_op, before_length), (after_op, after_length)]
        removed = cigar[:index - 1] + interface + cigar[index + 2:]

    return removed

def mapping_from_line(line):
    combined_mapping = pysam.AlignedRead()
    parsed_line = sam.parse_line(line)
    if parsed_line['strand'] == '-':
        combined_mapping.is_reverse = True
    combined_mapping.seq = parsed_line['SEQ']
    combined_mapping.qual = parsed_line['QUAL']
    combined_mapping.cigarstring = parsed_line['CIGAR']
    
    # This should obviously be made more general.
    for tag_name in ['Xs', 'Xq', 'Xw']:
        if tag_name in parsed_line:
            tag = [(tag_name, parsed_line[tag_name])]
            combined_mapping.tags = combined_mapping.tags + tag 

    return combined_mapping

def filter_mappings(mappings,
                    minimum_mapq=42,
                    max_insert_length=1000,
                    counts_dict=None,
                    verbose=False,
                    unmapped_fns=None,
                   ):
    ''' Filters out unmapped, nonuniquely mapped, or discordantly mapped
        reads.
    '''
    pair_counts = {'total': 0,
                   'unmapped': 0,
                   'indel': 0,
                   'nonunique': 0,
                   'discordant': 0,
                   'disoriented': 0,
                   'unique': Counter(),
                   'mapqs': Counter(),
                   'fragment_lengths': Counter(),
                   'tids': Counter(),
                  }

    if unmapped_fns:
        R1_unmapped_fn, R2_unmapped_fn = unmapped_fns
        R1_unmapped_fh = open(R1_unmapped_fn, 'w')
        R2_unmapped_fh = open(R2_unmapped_fn, 'w')

    for _, aligned_pair in utilities.group_by(mappings, key=lambda m: m.qname):
        if len(aligned_pair) != 2:
            raise ValueError(len(aligned_pair))

        pair_counts['total'] += 1
        
        R1_aligned, R2_aligned = aligned_pair
        # If R2 is mapped but R1 isn't, R2 gets reported first.
        if not R1_aligned.is_read1:
            R1_aligned, R2_aligned = R2_aligned, R1_aligned

        if (not R1_aligned.is_read1) or (not R2_aligned.is_read2):
            raise ValueError(R1_aligned, R2_aligned)
        
        pair_counts['mapqs'][R1_aligned.mapq] += 1
        pair_counts['mapqs'][R2_aligned.mapq] += 1

        if R1_aligned.is_unmapped or R2_aligned.is_unmapped:
            pair_counts['unmapped'] += 1
            
            if verbose:
                logging.info('{0} was unmapped'.format(R1_aligned.qname))
            
            if unmapped_fns:
                R1_read = sam.mapping_to_Read(R1_aligned)
                R2_read = sam.mapping_to_Read(R2_aligned)
                R1_unmapped_fh.write(str(R1_read))
                R2_unmapped_fh.write(str(R2_read))
        
        elif is_discordant(R1_aligned, R2_aligned, max_insert_length):
            pair_counts['discordant'] += 1
        
        else:
            pair_counts['tids'][R1_aligned.tid] += 1

            if is_disoriented(R1_aligned, R2_aligned):
                pair_counts['disoriented'] += 1
            elif R1_aligned.mapq < minimum_mapq or R2_aligned.mapq < minimum_mapq:
                pair_counts['nonunique'] += 1
                if verbose:
                    logging.info('{0} was nonunique, {1}, {2}'.format(R1_aligned.qname, R1_aligned.mapq, R2_aligned.mapq))
            else:
                pair_counts['unique'][R1_aligned.tid] += 1

                fragment_length = abs(R1_aligned.tlen)
                pair_counts['fragment_lengths'][fragment_length] += 1
                
                if sam.contains_indel_pysam(R1_aligned) or sam.contains_indel_pysam(R2_aligned):
                    pair_counts['indel'] += 1
                
                yield R1_aligned, R2_aligned

    if counts_dict != None:
        counts_dict.update(pair_counts)

def extract_fragment_lengths(bam_file_name):
    def concordantly_mapped(mapping):
        return not mapping.mate_is_unmapped and \
               not mapping.is_unmapped and \
               mapping.rnext == mapping.tid and \
               abs(mapping.tlen) < 10000

    bam_fh = pysam.Samfile(bam_file_name)
    TLENs = (abs(mapping.tlen) for mapping in bam_fh if mapping.is_read1 and concordantly_mapped(mapping))
    fragment_lengths = Counter(TLENs)
    
    # Note that counts_to_array implicitly discards negative key values.
    fragment_lengths = utilities.counts_to_array(fragment_lengths)

    return fragment_lengths

def filter_long_TLENs(bam_file_name, filtered_bam_file_name, max_TLEN):
    bam_file = pysam.Samfile(bam_file_name)
    filtered_bam_file = pysam.Samfile(filtered_bam_file_name, 'wb', template=bam_file)
    for aligned_read in bam_file:
        if aligned_read.tlen < max_TLEN:
            filtered_bam_file.write(aligned_read)

def get_concordant_pairs(R1_group, R2_group, max_insert_length):
    ''' Results are sorted by reference extent length. '''
    pairs = []
    for R1_m in R1_group:
        for R2_m in R2_group:
            if not is_discordant(R1_m, R2_m, max_insert_length) and not is_disoriented(R1_m, R2_m):
                pairs.append((R1_m, R2_m))
    pairs = sorted(pairs, key=lambda p: get_reference_extent(*p))
    return pairs

def group_mapping_pairs(mappings):
    groups = utilities.group_by(mappings, lambda r: r.query_name)
    for query_name, query_mappings in groups:
        R1_group = [m for m in query_mappings if m.is_read1]
        R2_group = [m for m in query_mappings if m.is_read2]
        yield query_name, (R1_group, R2_group)
