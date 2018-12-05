import numpy as np
from . import utilities
from .adapters_cython import *

primers = {
    'truseq': {
        'R1':         'TCTTTCCCTACACGACGCTCTTCCGATCT',
        'R2':    'GTGACTGGAGTTCAGACGTGTGCTCTTCCGATCT',
    },
    'PE': {
        'R1':         'TCTTTCCCTACACGACGCTCTTCCGATCT', # Note: same as truseq R1
        'R2': 'CGGTCTCGGCATTCCTGCTGAACCGCTCTTCCGATCT',
    },
    'nextera': {
        'R1':     'TCGTCGGCAGCGTCAGATGTGTATAAGAGACAG',
        'R2':    'GTCTCGTGGGCTCGGAGATGTGTATAAGAGACAG',
    },
}

primers['mix_and_match'] = {
    'R1': primers['nextera']['R1'],
    'R2': primers['truseq']['R2'],
}

flow_cell = {
    'P5': 'AATGATACGGCGACCACCGAGATCTACAC',
    'P7': 'CAAGCAGAAGACGGCATACGAGAT',
}

A_tail = 'A' * 10

# For backwards compatibility
truseq_R1_rc = 'AGATCGGAAGAGCGTCGTGTAGGGAAAGA'
truseq_R2_rc = 'AGATCGGAAGAGCACACGTCTGAACTCCAGTCAC'
P5_rc = 'TCTCGGTGGTCGCCGTATCATT'
P7_rc = 'ATCTCGTATGCCGTCTTCTGCTTG'

def build_before_adapters(I7_sequence='', primer_type='truseq', just_primers=False, R1_index='', R2_index=''):
    if just_primers:
        before_R1 = primers[primer_type]['R1'] + R1_index
        before_R2 = primers[primer_type]['R2'] + R2_index
    else:
        before_R1 = flow_cell['P5'] + primers[primer_type]['R1'] + R1_index
        before_R2 = flow_cell['P7'] + utilities.reverse_complement(I7_sequence) + primers[primer_type]['R2'] + R2_index

    return before_R1, before_R2

def build_adapters(**kwargs):
    before_R1, before_R2 = build_before_adapters(**kwargs)

    adapter_in_R1 = utilities.reverse_complement(before_R2) + A_tail
    adapter_in_R2 = utilities.reverse_complement(before_R1) + A_tail

    return adapter_in_R1, adapter_in_R2

def build_adapter_ranges(index_sequence, primer_type='truseq'):
    def make_ranges(construct, names):
        cumulative_lengths = list(np.cumsum(map(len, construct)))
        bounds = zip([0] + cumulative_lengths, cumulative_lengths)
        ranges = zip(names, bounds)
        return ranges
    
    primer_in_R1 = utilities.reverse_complement(primers[primer_type]['R2'])
    primer_in_R2 = utilities.reverse_complement(primers[primer_type]['R1'])

    R1_construct = [primer_in_R1,
                    index_sequence,
                    P7_rc,
                    A_tail,
                   ]
    R1_names = ['R2 primer',
                'I7',
                'P7',
                'A tail',
               ]

    chemistry_only_cycles = 7
    I5_length = 8
    R2_construct = [primer_in_R2[:-(I5_length + chemistry_only_cycles)],
                    primer_in_R2[-(I5_length + chemistry_only_cycles):-(chemistry_only_cycles)],
                    primer_in_R2[-chemistry_only_cycles:],
                    P5_rc,
                    A_tail,
                   ]
    R2_names = ['R1 primer',
                'I5',
                'Chemistry',
                'P7',
                'A tail',
               ]
    
    R1_ranges = make_ranges(R1_construct, R1_names)
    R2_ranges = make_ranges(R2_construct, R2_names)
    return R1_ranges, R2_ranges

def consistent_paired_position(R1_seq,
                               R2_seq,
                               adapter_in_R1,
                               adapter_in_R2,
                               min_comparison_length,
                               max_distance,
                               allow_prefix=True,
                              ):
    R1_positions = find_adapter_positions(R1_seq, adapter_in_R1, min_comparison_length, max_distance)
    R2_positions = find_adapter_positions(R2_seq, adapter_in_R2, min_comparison_length, max_distance)

    if allow_prefix:
        R1_prefix_position = find_adapter(adapter_in_R1, max_distance, R1_seq)
        if R1_prefix_position != len(R1_seq):
            R1_positions.append(R1_prefix_position)
        
        R2_prefix_position = find_adapter(adapter_in_R2, max_distance, R2_seq)
        if R2_prefix_position != len(R2_seq):
            R2_positions.append(R2_prefix_position)

    R1_positions = set(R1_positions)
    R2_positions = set(R2_positions)
    common_positions = R1_positions & R2_positions
    if common_positions:
        return min(common_positions)
    else:
        return None
