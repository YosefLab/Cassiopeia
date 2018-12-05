from collections import Counter

extension = 'txt'

def write_file(dict_of_counters, file_name):
    with open(file_name, 'w') as fh:
        for name, counter in sorted(dict_of_counters.items()):
            if '\t' in name:
                raise ValueError('Name of counter can\'t contain a tab')
            fh.write('{0}\n'.format(name))

            for key, count in counter.most_common():
                if isinstance(key, tuple):
                    key = ' '.join(key)
                fh.write('{0}\t{1}\n'.format(key, count))

def read_file(file_name):
    dict_of_counters = {}
    counter = Counter()
    for line in open(file_name):
        line = line.strip()
        if '\t' not in line:
            name = line
            counter = Counter()
            dict_of_counters[name] = counter
        else:
            key, count = line.split('\t')
            counter[key] = int(count)

    return dict_of_counters

def combine_data(first, second):
    for name in second:
        first[name].update(second[name])
    return first
