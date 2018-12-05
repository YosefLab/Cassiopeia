cpdef int adapter_hamming_distance(char *seq,
                                   char *adapter,
                                   int seq_length,
                                   int adapter_length,
                                   int start,
                                 ):
    ''' Returns the hamming distance between the overlap of seq[start:] and
        adapter.
    '''
    cdef int compare_length = min(adapter_length, seq_length - start)
    cdef int mismatches = 0
    cdef int i

    for i in range(compare_length):
        if seq[start + i] != adapter[i]:
            mismatches += 1

    return mismatches

cpdef simple_hamming_distance(char *first_seq, char *second_seq):
    return adapter_hamming_distance(first_seq, second_seq, len(first_seq), len(second_seq), 0)

def find_adapter(char* adapter, int max_distance, char *seq):
    ''' Returns the leftmost position in seq for which either:
            - seq[position:position + len(adapter)] is within hamming distance
              max_distance of adapter
            - seq[position:] is at least 10 bases long and is within hamming
              distance one of a prefix of adapter
            - seq[position:] exactly matches a prefix of adapter.
    '''
    cdef int seq_length = len(seq)
    cdef int adapter_length = len(adapter)
    cdef int distance, start
    cdef int max_long_prefix_distance = min(max_distance, 1)
        
    for start in range(seq_length - adapter_length + 1):
        distance = adapter_hamming_distance(seq,
                                            adapter,
                                            seq_length,
                                            adapter_length,
                                            start,
                                           )
        if distance <= max_distance:
            return start
    
    for start in range(seq_length - adapter_length + 1, seq_length):
        distance = adapter_hamming_distance(seq,
                                            adapter,
                                            seq_length,
                                            adapter_length,
                                            start,
                                           )
        if distance == 0:
            return start
        elif seq_length - start >= 10 and distance <= max_long_prefix_distance:
            return start
    
    # Convention: position of seq_length means no position was found
    return seq_length

def find_adapter_positions(read, adapter, int min_comparison_length, int max_distance):
    ''' Temporary for backwards compatibility. '''
    cdef int read_length = len(read)
    cdef int adapter_length = len(adapter)
    cdef int max_start = len(read) - min_comparison_length
    cdef int distance, start
        
    positions = [] 
    for start in range(max_start + 1):
        distance = adapter_hamming_distance(read,
                                            adapter,
                                            read_length,
                                            adapter_length,
                                            start,
                                           )
        if distance <= max_distance:
            positions.append(start)
    return positions
