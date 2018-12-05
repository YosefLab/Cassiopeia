import itertools
import collections
from six.moves import zip_longest

extension = 'txt'

def _write_pairs(pairs, log_file):
    ''' Writes pairs to a scalar log file. Each line has the form
        description:value
        where value is a comma-separated list of ints or floats.
    '''
    for description, value in pairs:
        if not isinstance(value, collections.Iterable):
            value = [value]
        if isinstance(value[0], int):
            to_string = str
        else:
            to_string = '{:.2f}'.format
        value_string = ','.join(map(to_string, value))
        log_file.write('{0}: {1}\n'.format(description, value_string))

def write_file(pairs, log_file_name):
    with open(log_file_name, 'w') as log_file:
        _write_pairs(pairs, log_file)

def append(pairs, log_file_name):
    with open(log_file_name, 'a') as log_file:
        _write_pairs(pairs, log_file)

def read_file(log_file_name):
    ''' Reads the format written by write_file into a list of tuples.
    '''
    def process_line(line):
        description, value = line.strip().split(':')
        strings = value.split(',')
        try:
            int(strings[0])
            dtype = int
        except ValueError:
            dtype = float
        value = [dtype(s) for s in strings]
        return description, value
    
    with open(log_file_name) as log_file:
        pairs = [process_line(l) for l in log_file]

    return pairs

def combine_data(first_pairs, second_pairs):
    combined_pairs = []
    zipped = zip_longest(first_pairs, second_pairs, fillvalue=(None, None))
    for (first_d, first_v), (second_d, second_v) in zipped:
        if first_d != second_d:
            raise RuntimeError('descriptions do not match')
        
        # Note: the use of '+' means these can't be np.arrays.
        combined_v = first_v + second_v 
        combined_pairs.append((first_d, combined_v))
    
    return combined_pairs

def collapse_pairs(pairs):
    collapsed_pairs = [(d, sum(v)) for d, v in pairs]
    return collapsed_pairs

def consolidate_stages(file_names, consolidated_file_name):
    pairs_list = [read_file(file_name) for file_name in file_names]
    collapsed_pairs_list = [collapse_pairs(pairs) for pairs in pairs_list]
    consolidated_pairs = itertools.chain.from_iterable(collapsed_pairs_list)
    write_file(consolidated_pairs, consolidated_file_name)
