import numpy as np

extension = 'txt'

def write_file(array, file_name):
    ''' Writes a 1D array into a text file.
    '''
    if array.ndim < 1:
        array = np.atleast_1d(array)
    elif array.ndim == 1:
        np.savetxt(file_name, array, fmt='%i', delimiter='\t')
    else:
        raise RuntimeError('Array has more than 2 dimensions.')

def read_file(file_name):
    ''' Reads the format written by array_file.write into an array.
    '''
    array = np.loadtxt(file_name, dtype=np.int, ndmin=1)
    return array

def combine_data(first_array, second_array):
    ''' Identically shaped arrays are added. Arrays of different shapes are
        zero-padded to the size of the biggest before adding.
    '''
    first_length = len(first_array)
    second_length = len(second_array)
    max_length = max(first_length, second_length)
    combined = np.zeros(max_length, int)
    combined[:first_length] += first_array
    combined[:second_length] += second_array
    return combined
