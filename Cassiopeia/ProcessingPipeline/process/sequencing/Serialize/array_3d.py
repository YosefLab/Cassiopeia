import numpy as np

extension = 'npy'

def write_file(array, file_name):
    ''' Writes a 3D array into a text file.
    '''
    np.save(file_name, array)

def read_file(file_name):
    ''' Reads the format written by array_file.write into an array.
    '''
    array = np.load(file_name)
    return array

def combine_data(first_array, second_array):
    ''' Identically shaped arrays are added.
    '''
    if first_array.shape != second_array.shape:
        raise ValueError
    
    combined = first_array + second_array

    return combined
