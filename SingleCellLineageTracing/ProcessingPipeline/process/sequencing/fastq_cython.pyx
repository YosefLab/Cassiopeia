import numpy as np
cimport cython

cdef int SANGER_OFFSET_typed = 33
SANGER_OFFSET = SANGER_OFFSET_typed

@cython.boundscheck(False)
def process_read(char* seq,
                 char* qual,
                 long[:, ::1] q_array,
                 long[:, ::1] average_q_array,
                 long[:, ::1] c_array,
                 long[:, ::1] c_above_min_q_array,
                 int min_q=0,
                ):
    cdef unsigned int i, q, b, seq_length

    seq_length = len(seq)
    for i in range(seq_length):
        # Automatic type conversion means ord() is unneccesary
        q = qual[i] - SANGER_OFFSET_typed
        q_array[i, q] += 1

        b = seq[i]

        average_q_array[i, b] += q

        c_array[i, b] += 1
        
        if q >= min_q:
            c_above_min_q_array[i, b] += 1

@cython.boundscheck(False)
def process_Alignment(char* seq,
                      char[:] qual,
                      long[:, ::1] q_array,
                      long[:, ::1] average_q_array,
                      long[:, ::1] c_array,
                      long[:, ::1] c_above_min_q_array,
                      int min_q=0,
                     ):
    cdef unsigned int i, q, b, seq_length

    seq_length = len(seq)
    for i in range(seq_length):
        q = qual[i]
        q_array[i, q] += 1

        b = seq[i]
        
        average_q_array[i, b] += q

        c_array[i, b] += 1
        
        if q >= min_q:
            c_above_min_q_array[i, b] += 1

def RSQCI_length(char *quals, int length):
    cdef int i
    for i in range(length):
        if quals[length - 1 - i] - SANGER_OFFSET_typed != 2:
            return i

    return length
