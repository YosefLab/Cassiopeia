import numpy as np

extension = 'txt'

def write_file(array, file_name):
    ''' Writes a 2D array into a text file.
    '''
    if array.ndim < 2:
        array = np.atleast_2d(array)
    elif array.ndim == 2:
        np.savetxt(file_name, array, fmt='%i', delimiter='\t')
    else:
        raise RuntimeError('Array has more than 2 dimensions.')

def read_file(file_name):
    ''' Reads the format written by array_file.write into an array.
    '''
    array = np.loadtxt(file_name, dtype=np.int, ndmin=2)
    return array

def combine_data(first_array, second_array):
    ''' Identically shaped arrays are added.
        Arrays of different sizes are treated as the smallest upper-left
        corner of an eventually-zero-in-both-directions infinite array.
    '''
    first_rows, first_cols = first_array.shape
    second_rows, second_cols = second_array.shape
    
    max_rows = max(first_rows, second_rows)
    max_cols = max(first_cols, second_cols)

    combined = np.zeros((max_rows, max_cols), int)
    combined[:first_rows, :first_cols] += first_array
    combined[:second_rows, :second_cols] += second_array
    return combined
