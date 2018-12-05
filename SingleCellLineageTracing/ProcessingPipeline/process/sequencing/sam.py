''' Utilities for dealing with sam files. '''

import re
import subprocess
import os
import shutil
import logging
import heapq
import contextlib
import copy

from collections import Counter
from itertools import chain
from pathlib import Path

import pysam

from . import utilities
from . import external_sort
from . import fastq
from . import mapping_tools

BAM_CMATCH = 0     # M
BAM_CINS = 1       # I
BAM_CDEL = 2       # D
BAM_CREF_SKIP = 3  # N
BAM_CSOFT_CLIP = 4 # S
BAM_CHARD_CLIP = 5 # H
BAM_CPAD = 6       # P
BAM_CEQUAL = 7     # =
BAM_CDIFF = 8      # X

op_to_char = {
    BAM_CMATCH:     'M',
    BAM_CINS:       'I',
    BAM_CDEL:       'D',
    BAM_CREF_SKIP:  'N',
    BAM_CSOFT_CLIP: 'S',
    BAM_CHARD_CLIP: 'H',
    BAM_CPAD:       'P',
    BAM_CEQUAL:     '=',
    BAM_CDIFF:      'X',
}
# Want to be able to lookup with int or char keys, so make every relevant char
# return itself.
for v in list(op_to_char.values()):
    op_to_char[v] = v

read_consuming_ops = {
    BAM_CMATCH,
    BAM_CINS,
    BAM_CSOFT_CLIP,
    BAM_CEQUAL,
    BAM_CDIFF,
}

ref_consuming_ops = {
    BAM_CMATCH,
    BAM_CDEL,
    BAM_CEQUAL,
    BAM_CDIFF,
    BAM_CREF_SKIP,
}

_unmapped_template = '{0}\t4\t*\t0\t0\t*\t*\t0\t0\t*\t*\n'.format

def get_strand(mapping):
    if mapping.is_reverse:
        strand = '-'
    else:
        strand = '+'
    return strand

def unmapped_aligned_read(qname):
    aligned_read = pysam.AlignedRead()
    aligned_read.qname = qname
    aligned_read.flag = 0x4
    aligned_read.rname = -1
    aligned_read.pos = -1
    aligned_read.mapq = 0
    aligned_read.cigar = None
    aligned_read.rnext = -1
    aligned_read.pnext = -1
    aligned_read.tlen = 0
    aligned_read.seq = '*'
    aligned_read.qual = '*'
    return aligned_read

def splice_in_name(line, new_name):
    return '\t'.join([new_name] + line.split('\t')[1:])

def get_header_lines(sam_file_name):                                                                  
    ''' Returns the header lines in sam_file_name. '''                                  
    
    header_lines = []
    
    with open(sam_file_name) as sam_file:                                                               
        for line in sam_file:                                                                           
            if line.startswith('@'):                                                                    
                header_lines.append(line)
            else:                                                                                       
                break                                                                                   
    
    return header_lines

def count_header_lines(sam_file_name):                                                                  
    ''' Returns the total number of header lines in sam_file_name. '''                                  
    return len(get_header_lines(sam_file_name))

def open_to_reads(sam_file_name):                                                                       
    ''' Returns an open file that has been advanced to the first read line in                           
        sam_file_name (i.e. past all header lines.)                                                     
    '''
    header_line_count = count_header_lines(sam_file_name)
    sam_file = open(sam_file_name)
    for i in range(header_line_count):
        sam_file.readline()
    return sam_file

def separate_header_and_read_lines(sam_line_source):
    ''' Returns a list of header lines and an iterator over read lines. '''
    if type(sam_line_source) == str:
        all_lines = open(sam_line_source)
    else:
        all_lines = iter(sam_line_source)

    header_lines = []
    for line in all_lines:
        if line.startswith('@'):
            header_lines.append(line)
        else:
            first_read_line = [line]
            break

    read_lines = chain(first_read_line, all_lines)

    return header_lines, read_lines

def mapped_reads(sam_file_name):
    ''' Returns an iterator over the lines of sam_file_name that correspond to
        mapped reads.
    '''
    def is_mapped(line):
        parsed = parse_line(line)
        return parsed['mapped']
    
    lines = open_to_reads(sam_file_name)
    mapped_lines = (line for line in lines if is_mapped(line))
    return mapped_lines
    
def parse_line(line):
    ''' Returns a dictionary of the information in a SAM line.
    '''
    fields = line.rstrip().split('\t')
    
    flags = int(fields[1])
    parsed_line = {'mapped':    not (flags & 0x4),
                   'QNAME':     fields[0],
                   'strand':    '-' if (flags & 0x10) else '+',
                   'RNAME':     fields[2],
                   'POS':       int(fields[3]) - 1, # SAM is 1-indexed
                   'MAPQ':      int(fields[4]),
                   'CIGAR':     fields[5],
                   'TLEN':      int(fields[8]),
                   'SEQ':       fields[9],
                   'QUAL':      fields[10],  
                  }
    
    # There are 11 mandatory fields, followed by optional tags.
    for field in fields[11:]:
        name, data_type = field.split(':')[:2]
        value = ':'.join(field.split(':')[2:])
        
        # BWA-specific XA field
        if name == 'XA':
            entries = value.rstrip(';').split(';')
            parsed_line['XA'] = []
            for entry in entries:
                ref_seq_name, strand_and_position, cigar, nm = entry.split(',')
                # first character of strand_and_position is [+-], rest is position
                strand = strand_and_position[0]
                position = int(strand_and_position[1:]) - 1 # SAM is 1-indexed
                xa_dict = {'QNAME':     fields[0],
                           'RNAME':     ref_seq_name,
                           'strand':    strand,
                           'POS':       position,
                           'CIGAR':     cigar,
                           'NM':        int(nm),
                          }
                parsed_line['XA'].append(xa_dict)
        else:
            if data_type == 'i':
                value = int(value)
            elif data_type == 'f':
                value = float(value)
            parsed_line[name] = value

    return parsed_line

cigar_block = re.compile(r'(\d+)([MIDNSHP=X])')

def cigar_string_to_blocks(cigar_string):
    """ Decomposes a CIGAR string into a list of its operations. """
    return [(int(l), k) for l, k in cigar_block.findall(cigar_string)]

def total_reference_nucs(cigar):
    return sum(length for op, length in cigar if op in ref_consuming_ops)

def total_read_nucs(cigar):
    return sum(length for op, length in cigar if op in read_consuming_ops)

def contains_indel(parsed_line):
    cigar_blocks = cigar_string_to_blocks(parsed_line['CIGAR'])
    kinds = [k for l, k in cigar_blocks]
    return ('I' in kinds or 'D' in kinds)

def contains_indel_pysam(read):
    kinds = [k for k, l in read.cigar]
    return (BAM_CINS in kinds or BAM_CDEL in kinds)

def indel_distance_from_edge(cigar):
    indel_indices = [i for i, (k, l) in enumerate(cigar) if k == BAM_CINS or k == BAM_CDEL]
    first_indel_index = min(indel_indices)
    ref_nucs_before = total_reference_nucs(cigar[:first_indel_index])
    last_indel_index = max(indel_indices)
    ref_nucs_after = total_reference_nucs(cigar[last_indel_index + 1:])
    return min(ref_nucs_before, ref_nucs_after)

def contains_splicing(read):
    kinds = [k for k, l in read.cigar]
    return (BAM_CREF_SKIP in kinds)

def contains_soft_clipping(parsed_line):
    cigar_blocks = cigar_string_to_blocks(parsed_line['CIGAR'])
    kinds = [k for k, l in cigar_blocks]
    return ('S' in kinds)

def contains_soft_clipping_pysam(read):
    kinds = [k for k, l in read.cigar]
    return (BAM_CSOFT_CLIP in kinds)

def get_soft_clipped_block(alignment, edge):
    strand = get_strand(alignment)

    if (edge == 5 and strand == '+') or (edge == 3 and strand == '-'):
        op, length = alignment.cigar[0]
        if op == BAM_CSOFT_CLIP:
            sl = slice(None, length)
        else:
            sl = slice(0)
    elif (edge == 5 and strand == '-') or (edge == 3 and strand == '+'):
        op, length = alignment.cigar[-1]
        if op == BAM_CSOFT_CLIP:
            sl = slice(-length, None)
        else:
            sl = slice(0)

    seq = alignment.seq[sl]
    qual = alignment.query_qualities[sl]

    return seq, qual

def get_max_soft_clipped_length(alignment):
    max_soft_clipped = 0
    for edge in [5, 3]:
        seq, qual = get_soft_clipped_block(alignment, edge)
        max_soft_clipped = max(max_soft_clipped, len(seq))
    return max_soft_clipped

def cigar_blocks_to_string(cigar_blocks):
    ''' Builds a CIGAR string out of a corresponding list of operations. '''
    string = ['{0}{1}'.format(length, op_to_char[op])
              for op, length in cigar_blocks
             ]
    return ''.join(string)

def alignment_to_cigar_blocks(ref_aligned, read_aligned):
    """ Builds a list of CIGAR operations from an alignment. """
    expanded_sequence = []
    for ref_char, read_char in zip(ref_aligned, read_aligned):
        if ref_char == '-':
            expanded_sequence.append('I')
        elif read_char == '-':
            expanded_sequence.append('D')
        elif ref_char == read_char:
            #expanded_sequence.append('=')
            expanded_sequence.append('M')
        else:
            #expanded_sequence.append('X')
            expanded_sequence.append('M')
    sequence, counts = utilities.decompose_homopolymer_sequence(expanded_sequence)
    return [[count, char] for char, count in zip(sequence, counts)]

def aligned_pairs_to_cigar(aligned_pairs, guide=None):
    op_sequence = []
    for read, ref in aligned_pairs:
        if read == None or read == '-':
            op_sequence.append(BAM_CDEL)
        elif read == 's':
            op_sequence.append(BAM_CREF_SKIP)
        elif ref == None or ref == '-':
            op_sequence.append(BAM_CINS)
        elif ref == 'S':
            op_sequence.append(BAM_CSOFT_CLIP)
        else:
            op_sequence.append(BAM_CMATCH)

    cigar = [(op, len(times)) for op, times in utilities.group_by(op_sequence)]

    if guide:
        guide_cigar, from_side = guide

        if from_side == 'right':
            cigar = cigar[::-1]
            guide_cigar = guide_cigar[::-1]

        for i in range(min(len(cigar), len(guide_cigar))):
            op, length = cigar[i]
            guide_op, guide_length = guide_cigar[i]
            cigar[i] = (guide_op, length)
        
        if from_side == 'right':
            cigar = cigar[::-1]
            guide_cigar = guide_cigar[::-1]

    return cigar

def cigar_to_aligned_pairs(cigar, start):
    aligned_pairs = []

    ref_pos = start
    read_pos = 0
    for op, length in cigar:
        if op == BAM_CMATCH or op == BAM_CEQUAL or op == BAM_CDIFF:
            for i in range(length):
                aligned_pairs.append((read_pos, ref_pos))

                ref_pos += 1
                read_pos += 1

        elif op == BAM_CDEL:
            # Deletion results in gap in read
            for i in range(length):
                aligned_pairs.append((None, ref_pos))
                
                ref_pos += 1
        
        elif op == BAM_CREF_SKIP:
            # Skip results in gap in read
            for i in range(length):
                aligned_pairs.append(('s', ref_pos))
                
                ref_pos += 1

        elif op == BAM_CINS:
            # Insertion results in gap in ref
            for i in range(length):
                aligned_pairs.append((read_pos, None))

                read_pos += 1
        
        elif BAM_CSOFT_CLIP:
            # Soft-clipping results in gap in ref
            for i in range(length):
                aligned_pairs.append((read_pos, 'S'))

                read_pos += 1

        else:
            raise ValueError('Unsupported op', cigar)

    return aligned_pairs

def cigar_to_aligned_pairs_backwards(cigar, end, read_length):
    aligned_pairs = []

    ref_pos = end
    read_pos = read_length - 1
    for op, length in cigar[::-1]:
        if op == BAM_CMATCH or op == BAM_CEQUAL or op == BAM_CDIFF:
            for i in range(length):
                aligned_pairs.append((read_pos, ref_pos))

                ref_pos -= 1
                read_pos -= 1

        elif op == BAM_CDEL or op == BAM_CREF_SKIP:
            # Deletion results in gap in read
            for i in range(length):
                aligned_pairs.append((None, ref_pos))
                
                ref_pos -= 1

        elif op == BAM_CINS:
            # Insertion results in gap in ref
            for i in range(length):
                aligned_pairs.append((read_pos, None))

                read_pos -= 1
        
        elif op == BAM_CSOFT_CLIP:
            # Soft-clipping results in gap in ref
            for i in range(length):
                aligned_pairs.append((read_pos, 'S'))

                read_pos -= 1

        else:
            raise ValueError('Unsupported op', cigar)

    return aligned_pairs

def truncate_cigar_blocks_up_to(cigar_blocks, truncated_length):
    ''' Given pysam-style cigar_blocks, truncates the blocks to explain
        truncated_length read bases.
    '''
    bases_so_far = 0
    truncated_blocks = []

    for operation, length in cigar_blocks:
        # If the next block wouldn't consume anything, we want to include it.
        if bases_so_far == truncated_length and operation in read_consuming_ops:
            break

        if operation in read_consuming_ops:
            length_to_use = min(truncated_length - bases_so_far, length)
            bases_so_far += length_to_use
        else:
            length_to_use = length

        truncated_blocks.append((operation, length_to_use))

        # If we didn't use the whole block, need to break because the next loop
        # will try to use the next block if it is non-consuming.
        if length_to_use < length:
            break

    return truncated_blocks

def truncate_cigar_blocks_from_beginning(cigar_blocks, truncated_length):
    ''' Removes cigar operations from the beginning of cigar_blocks so that
    truncated_length total read bases remain accounted for.
    '''
    flipped_truncated_blocks = truncate_cigar_blocks_up_to(cigar_blocks[::-1],
                                                           truncated_length,
                                                          )
    truncated_blocks = flipped_truncated_blocks[::-1]
    return truncated_blocks

def collapse_cigar_blocks(cigar_blocks):
    collapsed = []
    for kind, blocks in utilities.group_by(cigar_blocks, lambda t: t[0]):
        collapsed.append((kind, sum(t[1] for t in blocks)))
    return collapsed

def alignment_to_cigar_string(ref_aligned, read_aligned):
    """ Builds a CIGAR string from an alignment. """
    cigar_blocks = alignment_to_cigar_blocks(ref_aligned, read_aligned)
    return cigar_blocks_to_string(cigar_blocks)

md_number = re.compile(r'[0-9]+')
md_text = re.compile(r'[A-Z]+')

def md_string_to_ops_string(md_string):
    ''' Converts an MD string into a list of operations for supplying reference
        characters, either '=' if equal to the read, or any other char if equal
        to that char.
    '''
    # In the presence of a CIGAR string, the '^' character seems extraneous.
    # 94 is unicode ^.
    md_string = md_string.translate({94: None})
    
    match_lengths = [int(s) for s in re.findall(md_number, md_string)]
    text_blocks = re.findall(md_text, md_string)
    
    # The standard calls for a number to start and end, zero if necessary,
    # so after removing the initial number, there must be the same number of 
    # match_lengths and text_blocks.
    if len(text_blocks) != len(match_lengths) - 1:
        raise ValueError(md_string)
    
    ops_string = '='*match_lengths[0]
    for text_block, match_length in zip(text_blocks, match_lengths[1:]):
        ops_string += text_block
        ops_string += '='*match_length
    
    return ops_string

md_item_pattern = re.compile(r'[0-9]+|[TCAGN^]+')

def int_if_possible(string):
    try:
        int_value = int(string)
        return int_value
    except ValueError:
        return string

def md_string_to_items(md_string):
    items = [int_if_possible(item) for item in md_item_pattern.findall(md_string)]
    return items

def md_items_to_md_string(items):
    string = ''.join(str(item) for item in items)
    return string

def reverse_md_items(items):
    reversed_items = []
    for item in items[::-1]:
        if isinstance(item, int):
            reversed_items.append(item)
        else:
            if item.startswith('^'):
                reversed_items.append('^' + item[:0:-1])
            else:
                reversed_items.append(item[::-1])
    return reversed_items

def truncate_md_items(md_items, truncated_length):
    if truncated_length == 0:
        truncated_items = [0]
    else:
        bases_so_far = 0
        truncated_items = []

        for item in md_items:
            if bases_so_far == truncated_length:
                break

            if isinstance(item, int):
                length_to_use = min(truncated_length - bases_so_far, item)
                bases_so_far += length_to_use
                truncated_items.append(length_to_use)
            else:
                if item.startswith('^'):
                    truncated_item = ['^']
                    item = item[1:]
                else:
                    truncated_item = []

                for c in item:
                    if c == '^':
                        raise ValueError
                    
                    if bases_so_far == truncated_length:
                        break

                    truncated_item.append(c)
                    bases_so_far += 1

                truncated_items.append(''.join(truncated_item))

        # Ensure that it starts and ends with a number
        if not isinstance(truncated_items[0], int):
            truncated_items = [0] + truncated_items
        if not isinstance(truncated_items[-1], int):
            truncated_items = truncated_items + [0]

    return truncated_items

def combine_md_strings(first_string, second_string):
    if first_string == '':
        return second_string
    if second_string == '':
        return first_string

    first_items = md_string_to_items(first_string)
    second_items = md_string_to_items(second_string)
    before = first_items[:-1]
    after = second_items[1:]

    if isinstance(first_items[-1], int):
        if isinstance(second_items[0], int):
            interface = [first_items[-1] + second_items[0]]
        else:
            interface = [first_items[-1], second_items[0]]
    else:
        if isinstance(second_items[0], int):
            interface = [first_items[-1], second_items[0]]
        else:
            if first_items[-1].startswith('^'):
                if second_items[0].startswith('^'):
                    interface = [first_items[-1] + second_items[0][1:]]
                else:
                    interface = [first_items[-1], 0, second_items[0]]
            else:
                if second_items[0].startswith('^'):
                    interface = [first_items[-1], 0, second_items[0]]
                else:
                    interface = [first_items[-1] + second_items[0]]

    combined_items = before + interface + after

    combined_string = md_items_to_md_string(combined_items)
    
    return combined_string

def truncate_md_string_up_to(md_string, truncated_length):
    ''' Truncates from the end of md_string so that the result only consumes
    truncated_length ref characters.
    '''
    md_items = md_string_to_items(md_string)
    truncated_items = truncate_md_items(md_items, truncated_length)
    return md_items_to_md_string(truncated_items)

def truncate_md_string_from_beginning(md_string, truncated_length):
    ''' Truncates from the beginning of md_string so that the result only
    consumes truncated_length ref characters.
    '''
    md_items = md_string_to_items(md_string)
    reversed_items = reverse_md_items(md_items)
    reversed_truncated_items = truncate_md_items(reversed_items, truncated_length)
    truncated_items = reverse_md_items(reversed_truncated_items)
    return md_items_to_md_string(truncated_items)
    
def produce_alignment(mapping):
    ''' Returns a list of (ref_char, read_char, qual_char, ref_pos, read_pos)
        tuples.
    '''
    read_seq = mapping.seq
    if read_seq == None:
        read_seq = ''
    
    read_quals = mapping.query_qualities
    if read_quals == None:
        read_quals = []
    
    MD_string = dict(mapping.tags)['MD']
    
    ref_ops = iter(md_string_to_ops_string(MD_string))
    
    columns = []

    ref_pos = mapping.pos
    read_pos = 0
    for op, length in mapping.cigar:
        if op == BAM_CMATCH or op == BAM_CEQUAL or op == BAM_CDIFF:
            for i in range(length):
                read_char = read_seq[read_pos]
                
                ref_op = next(ref_ops)
                if ref_op == '=':
                    ref_char = read_char
                else:
                    ref_char = ref_op
                
                qual = read_quals[read_pos]
                
                column = (ref_char, read_char, qual, ref_pos, read_pos)
                columns.append(column)

                ref_pos += 1
                read_pos += 1

        elif op == BAM_CDEL:
            # Deletion results in gap in read
            for i in range(length):
                read_char = '-'
                ref_char = next(ref_ops)
                qual = 0 
                
                column = (ref_char, read_char, qual, ref_pos, read_pos)
                columns.append(column)
                
                ref_pos += 1

        elif op == BAM_CINS:
            # Insertion results in gap in ref
            for i in range(length):
                read_char = read_seq[read_pos]
                ref_char = '-'
                qual = read_quals[read_pos]
                column = (ref_char, read_char, qual, ref_pos, read_pos)
                columns.append(column)

                read_pos += 1

        elif op == BAM_CREF_SKIP:
            ref_pos += length

        elif op == BAM_CSOFT_CLIP:
            read_pos += length
    
    return columns

def ref_dict_from_mapping(mapping):
    ''' Build a dictionary mapping reference positions to base identities from
    the cigar and MD tag of a mapping.
    '''
    alignment = produce_alignment(mapping)
    ref_dict = {}
    for ref_char, _, _, ref_position, _ in alignment:
        if ref_char == '-':
            continue

        if ref_position in ref_dict:
            # A ref_position shouldn't appear more than once
            raise ValueError(mapping)

        ref_dict[ref_position] = ref_char

    return ref_dict

def merge_ref_dicts(first_dict, second_dict):
    ''' Merge dictionaries mapping reference positions to base identities. '''
    merged_dict = {}
    merged_dict.update(first_dict)
    for position, base in second_dict.items():
        if position in merged_dict:
            if merged_dict[position] != base:
                # contradiction
                raise ValueError(first_dict, second_dict)
        else:
            merged_dict[position] = base

    return merged_dict

def alignment_to_MD_string(ref_aligned, read_aligned):
    ''' Produce an MD string from an alignment. '''
    # Mark all blocks of matching with numbers, all deletion bases with '^*0', and all mismatch bases.
    MD_list = []
    current_match_length = 0
    for ref_char, read_char in zip(ref_aligned, read_aligned):
        if ref_char == read_char:
            current_match_length += 1
        elif ref_char != '-':
            if current_match_length > 0:
                MD_list.append(current_match_length)
                current_match_length = 0

            if read_char == '-':
                MD_list.append('^{0}'.format(ref_char))
                MD_list.append(0)
            else:
                MD_list.append(ref_char)
    
    if current_match_length > 0:
        MD_list.append(current_match_length)

    # Remove all zeros that aren't a deletion followed by a mismatch
    reduced_MD_list = []
    for i in range(len(MD_list)):
        if isinstance(MD_list[i], int):
            if MD_list[i] > 0:
                reduced_MD_list.append(MD_list[i])
            elif 0 < i < len(MD_list) - 1:
                if isinstance(MD_list[i - 1], str) and isinstance(MD_list[i + 1], str) and MD_list[i - 1][0] == '^' and MD_list[i + 1][0] != '^':
                    reduced_MD_list.append(MD_list[i])
        else:
            reduced_MD_list.append(MD_list[i])
    
    # Collapse all deletions.
    collapsed_MD_list = [reduced_MD_list[0]]
    for i in range(1, len(reduced_MD_list)):
        if isinstance(collapsed_MD_list[-1], str) and collapsed_MD_list[-1][0] == '^' and \
           isinstance(reduced_MD_list[i], str) and reduced_MD_list[i][0] == '^':
            
            collapsed_MD_list[-1] += reduced_MD_list[i][1]
        else:
            collapsed_MD_list.append(reduced_MD_list[i])

    # The standard calls for a number to start and to end, zero if necessary.
    if isinstance(collapsed_MD_list[0], str):
        collapsed_MD_list.insert(0, 0)
    if isinstance(collapsed_MD_list[-1], str):
        collapsed_MD_list.append(0)
    
    MD_string = ''.join(map(str, collapsed_MD_list))
    return ''.join(map(str, collapsed_MD_list))

def line_groups(sam_file_name, key):
    ''' Yields (key value, list of consecutive lines from sam_file_name
        that are all transormed to key value).
    '''
    sam_file = open_to_reads(sam_file_name)
    groups = utilities.group_by(sam_file, key)
    return groups

def sort(input_file_name, output_file_name):
    header_lines, read_lines = separate_header_and_read_lines(input_file_name)
    with open(output_file_name, 'w') as output_file:
        for header_line in header_lines:
            output_file.write(header_line)
        
        external_sort.external_sort(read_lines, output_file)
    
def sort_bam(input_file_name, output_file_name, by_name=False, num_threads=1):
    samtools_command = ['samtools', 'sort']
    if by_name:
        samtools_command.append('-n')
    samtools_command.extend(['-@', str(num_threads),
                             '-o', output_file_name,
                             input_file_name,
                            ])

    subprocess.check_call(samtools_command)

    if not by_name:
        pysam.index(output_file_name)

def merge_sorted_bam_files(input_file_names, merged_file_name, by_name=False):
    if len(input_file_names) == 1:
        shutil.copy(input_file_names[0], merged_file_name)
    else:
        merge_command = ['samtools', 'merge', '-f']
        if by_name:
            merge_command.append('-n')
        merge_command.extend([merged_file_name] + input_file_names)
        subprocess.check_call(merge_command)
    
    if not by_name:
        pysam.index(str(merged_file_name))

def bam_to_sam(bam_file_name, sam_file_name):
    view_command = ['samtools', 'view', '-h', '-o', sam_file_name, bam_file_name]
    subprocess.check_call(view_command)

def get_length_counts(bam_file_name, only_primary=True, only_unique=False):
    bam_file = pysam.AlignmentFile(bam_file_name)
    if only_unique:
        qlen_counts = Counter(ar.qlen for ar in bam_file if ar.mapping_quality == 50)
    elif only_primary:
        qlen_counts = Counter(ar.qlen for ar in bam_file if not ar.is_unmapped and not ar.is_secondary)
    else:
        qlen_counts = Counter(ar.qlen for ar in bam_file)
    
    return qlen_counts

def get_tlen_counts(bam_file_name, only_primary=True, only_unique=False):
    bam_file = pysam.AlignmentFile(bam_file_name)
    if only_unique:
        tlen_counts = Counter(ar.tlen for ar in bam_file if ar.mapping_quality == 50)
    elif only_primary:
        tlen_counts = Counter(ar.tlen for ar in bam_file if not ar.is_unmapped and not ar.is_secondary)
    else:
        tlen_counts = Counter(ar.tlen for ar in bam_file)
    
    return tlen_counts

def get_mapq_counts(bam_file_name):
    bam_file = pysam.AlignmentFile(bam_file_name)
    mapq_counts = Counter(ar.mapq for ar in bam_file)
    return mapq_counts

def mapping_to_Read(mapping):
    seq = mapping.get_forward_sequence()
    qual = fastq.encode_sanger(mapping.get_forward_qualities())

    read = fastq.Read(mapping.query_name, seq, qual)
    return read

def sam_to_fastq(sam_file_name):
    sam_file = pysam.AlignmentFile(str(sam_file_name))
    for mapping in sam_file:
        yield mapping_to_Read(mapping)

bam_to_fastq = sam_to_fastq

class AlignmentSorter(object):
    ''' Context manager that handles writing AlignedSegments into a samtools
    sort process.
    '''
    def __init__(self, output_file_name, header, by_name=False):
        self.header = header
        self.output_file_name = str(output_file_name)
        self.by_name = by_name
        self.fifo = mapping_tools.TemporaryFifo(name='unsorted_fifo.bam')

    def __enter__(self):
        self.fifo.__enter__()
        sort_command = ['samtools', 'sort']
        if self.by_name:
            sort_command.append('-n')

        self.dev_null = open(os.devnull, 'w')
        sort_command.extend(['-T', self.output_file_name,
                             '-o', self.output_file_name,
                             self.fifo.file_name,
                            ])
        self.sort_process = subprocess.Popen(sort_command,
                                             stderr=subprocess.PIPE,
                                            )

        self.sam_file = pysam.AlignmentFile(self.fifo.file_name, 'wbu', header=self.header)

        return self

    def __exit__(self, exception_type, exception_value, exception_traceback):
        self.sam_file.close()
        _, err_output = self.sort_process.communicate()
        self.dev_null.close()
        self.fifo.__exit__(exception_type, exception_value, exception_traceback)
        
        if self.sort_process.returncode:
            raise RuntimeError(err_output)

        if not self.by_name:
            pysam.index(self.output_file_name)

    def write(self, alignment):
        self.sam_file.write(alignment)

@contextlib.contextmanager
def multiple_AlignmentSorters(handlers):
    ''' from http://hoardedhomelyhints.dietbuddha.com/2014/02/python-aggregating-multiple-context.html '''
    for handler in handlers:
        handler.__enter__()
  
    err = None
    exc_info = (None, None, None)
    try:
        yield
    except Exception as err:
        exc_info = sys.exc_info()
 
    # exc_info get's passed to each subsequent handler.__exit__
    # unless one of them suppresses the exception by returning True
    for handler in reversed(handlers):
        if handler.__exit__(*exc_info):
            err = False
            exc_info = (None, None, None)
    if err:
        raise err

class AlignedSegmentByName(object):
    def __init__(self, aligned_segment):
        self.aligned_segment = aligned_segment
        
    def __lt__(self, other):
        return self.aligned_segment.query_name < other.aligned_segment.query_name

def merge_by_name(*mapping_iterators):
    ''' Merges iterators over mappings that are sorted by name.
    '''
    wrapped_iterators = [(AlignedSegmentByName(m) for m in mappings) for mappings in mapping_iterators]
    merged_wrapped = heapq.merge(*wrapped_iterators)
    last_qname = None
    for al_by_name in merged_wrapped:
        qname = al_by_name.aligned_segment.query_name
        if last_qname is not None and qname < last_qname:
            raise ValueError('Attempted to merge unsorted mapping iterators')

        last_qname = qname
        yield al_by_name.aligned_segment

def aligned_pairs_exclude_soft_clipping(mapping):
    cigar = mapping.cigartuples
    aligned_pairs = mapping.aligned_pairs
    first_op, first_length = cigar[0]
    if first_op == BAM_CSOFT_CLIP:
        aligned_pairs = aligned_pairs[first_length:]
    if len(cigar) > 1:
        last_op, last_length = cigar[-1]
        if last_op == BAM_CSOFT_CLIP:
            aligned_pairs = aligned_pairs[:-last_length]
    return aligned_pairs

def parse_idxstats(bam_fn):
    lines = pysam.idxstats(bam_fn).splitlines()
    fields = [line.split('\t') for line in lines]
    parsed = {rname: int(count) for rname, _, count, _ in fields}
    return parsed

def get_num_alignments(bam_fn):
    return sum(parse_idxstats(bam_fn).values())

def crop_al_to_query_int(alignment, start, end):
    ''' Replace any parts of alignment that involve query bases not in the
    interval [start, end] with soft clipping.
    query coords are given relative to the original read (and are therefore
    transformed if alignment is reversed.)
    '''
    alignment = copy.deepcopy(alignment)

    if alignment.is_unmapped:
        return alignment

    if alignment.is_reverse:
        start, end = alignment.query_length - 1 - end, alignment.query_length - 1 - start

    aligned_pairs = cigar_to_aligned_pairs(alignment.cigar, alignment.reference_start)

    start_i = 0
    while aligned_pairs[start_i][0] == None or aligned_pairs[start_i][0] < start:
        start_i += 1
        
    end_i = len(aligned_pairs) - 1
    while aligned_pairs[end_i][0] is None or aligned_pairs[end_i][0] > end:
        end_i -= 1

    restricted_pairs = aligned_pairs[start_i:end_i + 1]
    cigar = aligned_pairs_to_cigar(restricted_pairs) 

    before_soft = start
    if before_soft > 0:
        cigar = [(BAM_CSOFT_CLIP, before_soft)] + cigar

    after_soft = alignment.query_length - end - 1
    if after_soft > 0:
        cigar = cigar + [(BAM_CSOFT_CLIP, after_soft)]

    restricted_rs = [r for q, r in restricted_pairs if r != None and r != 'S']
    if not restricted_rs:
        alignment.is_unmapped = True
        alignment.cigar = []
    else:
        alignment.reference_start = min(restricted_rs)
        alignment.cigar = cigar

    return alignment

def crop_al_to_ref_int(alignment, start, end):
    ''' Returns a copy of alignment in which any query bases that align
    outside the interval [start, end] are soft-clipped. If no bases are left,
    sets alignment.is_unmapped to true.
    '''
    alignment = copy.deepcopy(alignment)

    if alignment.reference_start > end or alignment.reference_end - 1 < start:
        # alignment doesn't overlap the ref interval at all
        alignment.is_unmapped = True
        return alignment

    if alignment.reference_start >= start and alignment.reference_end - 1 <= end:
        # alignment is entirely contained in the ref_interval
        return alignment

    query_length = alignment.query_length
    aligned_pairs = cigar_to_aligned_pairs(alignment.cigar, alignment.reference_start)

    start_i = 0
    while (
        aligned_pairs[start_i][1] == 'S' or
        aligned_pairs[start_i][0] is None or
        aligned_pairs[start_i][1] is None or
        aligned_pairs[start_i][1] < start
    ):
        start_i += 1
        
    end_i = len(aligned_pairs) - 1
    while (
        aligned_pairs[end_i][1] == 'S' or
        aligned_pairs[end_i][0] is None or
        aligned_pairs[end_i][1] is None or
        aligned_pairs[end_i][1] > end
    ):
        end_i -= 1

    remaining = aligned_pairs[start_i:end_i + 1]
    if remaining:
        cigar = aligned_pairs_to_cigar(remaining)
        before_soft = remaining[0][0]
        if before_soft > 0:
            cigar = [(BAM_CSOFT_CLIP, before_soft)] + cigar

        after_soft = query_length - remaining[-1][0] - 1
        if after_soft > 0:
            cigar = cigar + [(BAM_CSOFT_CLIP, after_soft)]

        alignment.cigar = cigar

        alignment.reference_start = aligned_pairs[start_i][1]
    else:
        alignment.is_unmapped = True
        alignment.cigar = []
    

    return alignment

def disallow_query_positions_from_other(alignment, other):
    start, end = query_interval(alignment)
    other_start, other_end = query_interval(other)
    if other_start <= end or other_end >= start:
        if other_start > start and other_end < end:
            raise ValueError
        elif other_start <= start:
            crop_al_to_query_int(alignment, other_end + 1, alignment.query_length - 1)
        elif other_end >= end:
            crop_al_to_query_int(alignment, 0, other_start - 1)

    return alignment

def query_interval(alignment):
    start = alignment.query_alignment_start
    end = alignment.query_alignment_end - 1
    if alignment.is_reverse:
        start, end = true_query_position(end, alignment), true_query_position(start, alignment)
    return start, end

def merge_adjacent_alignments(first, second):
    ''' Takes two alignments to the same reference name and strand that
    are adjacent or partially overlap on the query and merges them by
    assigning the overlapping query bases to the rightmost alignment
    and adding an appropriately sized deletion.
    '''
    if first.reference_name != second.reference_name:
        raise ValueError
    if get_strand(first) != get_strand(second):
        raise ValueError
    
    (q_left_start, q_left_end), (q_right_start, q_right_end) = sorted(map(query_interval, (first, second)))
    if q_left_start == q_right_start and q_left_end == q_right_end:
        # query intervals are identical
        raise ValueError

    if not (q_right_start - 1 <= q_left_end <= q_right_end - 1):
        # query intervals are not partially overlapping or adjacent
        raise ValueError

    left, right = sorted((first, second), key=lambda al: al.reference_start)

    disallow_query_positions_from_other(left, right)

    # Since they were adjacent or overlapping before disallowing in left,
    # they are now adjacent and right's cigar starts with a soft clip
    # block of size left.query_alignment_length

    deletion_length = right.reference_start - left.reference_end
    if deletion_length < 0:
        # These shouldn't actually be merged.
        raise ValueError(deletion_length)
    else:
        cigar = left.cigar[:-1] + [(BAM_CDEL, deletion_length)] + right.cigar[1:]
        left.cigar = cigar

    return left

def true_query_position(p, alignment):
    if alignment.is_reverse:
        p = alignment.query_length - 1 - p
    return p

def closest_query_position(r, alignment, which_side='either'):
    ''' Return query position paired with r (or with the closest to r) '''
    r_to_q = {r: true_query_position(q, alignment)
              for q, r in alignment.aligned_pairs
              if r is not None and q is not None
             }
    if r in r_to_q:
        q = r_to_q[r]
    else:
        if which_side == 'either':
            rs = r_to_q
        elif which_side == 'before':
            rs = (other_r for other_r in r_to_q if other_r < r)
        elif which_side == 'after':
            rs = (other_r for other_r in r_to_q if other_r > r)

        closest_r = min(rs, key=lambda other_r: abs(other_r - r))
        q = r_to_q[closest_r]

    return q

def max_block_length(alignment, block_types):
    if alignment.is_unmapped:
        return 0
    else:
        block_lengths = [l for k, l in alignment.cigar if k in block_types]
        if len(block_lengths) == 0:
            return 0
        else:
            return max(block_lengths)

def total_indel_lengths(alignment):
    if alignment.is_unmapped:
        return 0
    else:
        return sum(l for k, l in alignment.cigar if k == BAM_CDEL or k == BAM_CINS)

def get_ref_pos_to_block(alignment):
    ref_pos_to_block = {}
    reference_start = alignment.reference_start
    for kind, length in alignment.cigar:
        if kind in ref_consuming_ops:
            for r in range(reference_start, reference_start + length):
                ref_pos_to_block[r] = (kind, length)
            reference_start += length

    return ref_pos_to_block

def split_at_first_large_insertion(alignment, min_length=20):
    q = 0

    # Using cigar blocks, march from beginning of the read to the (possible)
    # insertion point to determine the query interval to crop to.
    # If the alignment is reversed, alignment.cigar is reversed relative to
    # true query positions.
    cigar = alignment.cigar
    if alignment.is_reverse:
        cigar = cigar[::-1]

    for kind, length in cigar:
        if kind == BAM_CINS and length >= min_length:
            before = crop_al_to_query_int(alignment, 0, q - 1)
            after = crop_al_to_query_int(alignment, q + length, alignment.query_length)
            return [before, after]
        else:
            if kind in read_consuming_ops:
                q += length

    return [alignment]

def split_at_large_insertions(alignment):
    all_split = []
    
    to_split = [alignment]

    while len(to_split) > 0:
        candidate = to_split.pop()
        split = split_at_first_large_insertion(candidate)
        if len(split) > 1:
            to_split.extend(split)
        else:
            all_split.extend(split)

    return all_split

def grouped_by_name(als):
    if isinstance(als, (str, Path)):
        als = pysam.AlignmentFile(str(als))

    grouped = utilities.group_by(als, lambda al: al.query_name)
    return grouped

def header_from_STAR_index(index):
    names = [l.strip() for l in (index / 'chrName.txt').open()]
    lengths = [int(l.strip()) for l in (index / 'chrLength.txt').open()]
    header = pysam.AlignmentHeader.from_references(names, lengths)
    return header
