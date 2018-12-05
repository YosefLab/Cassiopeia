import itertools
import log

write_file = log.write_file

def read_file(file_name):
    def process_line(line):
        description, value = line.strip().split(':')
        value = map(float, value.split(','))
        return description, value
    
    with open(file_name) as fh:
        pairs = map(process_line, fh)

    return pairs

def combine_data(first_pairs, second_pairs):
    combined_pairs = []
    zipped = itertools.izip_longest(first_pairs, second_pairs, fillvalue=(None, None))
    for (first_d, first_v), (second_d, second_v) in zipped:
        if first_d != second_d:
            raise RuntimeError, 'descriptions do not match'
        sum_v = first_v + second_v
        combined_pairs.append((first_d, sum_v))
    return combined_pairs
