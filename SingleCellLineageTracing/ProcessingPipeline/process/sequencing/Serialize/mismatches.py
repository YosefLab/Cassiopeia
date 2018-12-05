import numpy as np

extension = 'npy'

def write_file(type_counts, mismatch_file_name):
    if not str(mismatch_file_name).endswith('.npy'):
        raise ValueError('{0} does not end in .npy'.format(mismatch_file_name))
    np.save(mismatch_file_name, type_counts)

def read_file(mismatch_file_name):
    type_counts = np.load(mismatch_file_name)
    return type_counts

def combine_data(first_type_counts, second_type_counts):
    assert first_type_counts.shape == second_type_counts.shape
    return first_type_counts + second_type_counts
