from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import string
from itertools import islice, groupby, cycle, product, chain
from six.moves import zip
import contextlib
import sys
import functools

import numpy as np
import Bio.Data.IUPACData
import scipy.optimize

identity = lambda x: x

mapping = Bio.Data.IUPACData.ambiguous_dna_complement
for uppercase in list(mapping):
    mapping[uppercase.lower()] = mapping[uppercase].lower()
order = ''.join(mapping)
from_bytes = order.encode()
to_bytes = ''.join(mapping[f] for f in order).encode()
complement_table = bytes.maketrans(from_bytes, to_bytes)

def complement(seq):
    return seq.translate(complement_table)

def reverse_complement(seq):
    return complement(seq)[::-1]

base_order = 'ACGTN-'
base_to_index = {b: i for i, b in enumerate(base_order)}
base_to_complement_index = {b: i for i, b in enumerate(complement(base_order))}

complement_index = {i: base_to_complement_index[b] for i, b in enumerate(base_order)}

IUPAC = {
    'A': {'A'},
    'C': {'C'},
    'G': {'G'},
    'T': {'T'},
    'M': {'A', 'C'},
    'R': {'A', 'G'},
    'W': {'A', 'T'},
    'S': {'C', 'G'},
    'Y': {'C', 'T'},
    'K': {'G', 'T'},
    'V': {'A', 'C', 'G'},
    'H': {'A', 'C', 'T'},
    'D': {'A', 'G', 'T'},
    'B': {'C', 'G', 'T'},
    'N': {'G', 'A', 'T', 'C'},
}

def counts_to_array(counts, dim=1):
    ''' Converts a dictionary with integer keys into an array. '''
    if dim == 1:
        if len(counts) > 0:
            biggest = max(counts)
            array = np.array([counts[i] for i in range(biggest + 1)])
        else:
            array = np.array([0])
    elif dim == 2:
        if len(counts) > 0:
            larger_of_two = map(max, counts)
            biggest = max(larger_of_two)
            array = np.zeros(shape=(biggest + 1, biggest + 1), dtype=np.int)
            for key, count in counts.items():
                array[key] = count
        else:
            array = np.array([[0]])
    return array

def mean_from_histogram(histogram):
    if histogram.sum() == 0:
        return 0
    else:
        weighted_sum = np.dot(histogram, np.arange(len(histogram)))
        num_items = float(histogram.sum())
        mean = weighted_sum / num_items
    return mean

def group_by(iterable, key=None, sort=False):
    ''' Groups iterable into lists of consecutive elements that are transformed
        into the same value key.
    '''
    if sort:
        iterable = sorted(iterable, key=key)

    groups = groupby(iterable, key)
    group_lists = ((value, list(iterator)) for value, iterator in groups)
    return group_lists

def round_robin(iterables):
    ''' Modified from recipe on itertools doc page credited to George Sakkis.
    '''
    pending = len(iterables)
    nexts = cycle(iter(it).next for it in iterables)
    while pending:
        try:
            for next in nexts:
                yield next()
        except StopIteration:
            pending -= 1
            # The next pending yields from nexts will be the remaining
            # non-empty iterators.
            nexts = cycle(islice(nexts, None, pending))

def line_count(file_name):
    ''' Return the number of lines in file_name by calling wc. '''
    wc_output = subprocess.check_output(['wc', '-l', file_name])
    count, _ = wc_output.strip().split()
    count = int(count)
    return count

def pairwise(s):
    """ Returns the elements of s in overlapping pairs. """
    return [(s[i - 1], s[i]) for i in range(1, len(s))]

def all_consecutive(s):
    """ Returns True if the elements of s when sorted are consecutive integers. """
    for (x, y) in pairwise(sorted(s)):
        if y - x != 1:
            return False
    return True

def empirical_cdf(values):
    ''' From stackoverflow. '''
    sorted_values = np.sort(values)
    cumulative = np.true_divide(np.arange(len(sorted_values)), len(sorted_values))
    return sorted_values, cumulative

def mers(k, include_N=False):
    if include_N:
        chars = 'TCAGN'
    else:
        chars = 'TCAG'
    return (''.join(mer) for mer in product(chars, repeat=k))

def smooth(ys, either_side):
    smoothed = pd.Series(ys).rolling(window=2 * either_side + 1, center=True, min_periods=1).mean()
    return smoothed

def reverse_dictionary(d):
    r = {v: k for k, v in d.iteritems()}
    return r

def split_nonempty(string, delim):
    return [f for f in string.split(delim) if f]

def normalize_rows(array):
    return np.true_divide(array, array.sum(axis=1)[:, np.newaxis])

def reverse_enumerate(list_like):
    reversed_range = range(len(list_like) - 1, -1, -1)
    return zip(reversed_range, list_like[::-1])

def memoized_property(f):
    @property
    @functools.wraps(f)
    def memoized_f(self):
        attr_name = '_' + (f.__name__)
        
        if not hasattr(self, attr_name):
            setattr(self, attr_name, f(self))
        
        return getattr(self, attr_name)
    
    return memoized_f

def reservoir_sample(iterable, n):
    sample = []
    for i, item in enumerate(iterable):
        if i < n:
            sample.append(item)
        else:
            j = np.random.randint(i + 1)
            if j < n:
                sample[j] = item
    
    return sample

def chunks(iterable, n):
    '''from https://stackoverflow.com/a/29524877
    Note: this only works if you actually consume each chunk before getting the
    next one.
    '''
    iterable = iter(iterable)
    while True:
        first = next(iterable)
        rest = islice(iterable, n - 1)
        yield chain([first], rest)

@contextlib.contextmanager
def possibly_fn(fn=None):
    # from https://stackoverflow.com/a/22264583
    if fn is not None:
        writer = open(str(fn), 'w')
    else:
        writer = sys.stdout

    yield writer

    if fn != None: writer.close()


def clopper_pearson(x, n, alpha=0.05):
    if n == 0:
        return 0., 0.

    elif x == 0:
        cdf = lambda t: scipy.stats.binom.cdf(x, n, t) - alpha
        u = scipy.optimize.brentq(cdf, 0, 1)
        l = 0.
        mle = 0.

    elif x == n:
        sf = lambda t: scipy.stats.binom.sf(x - 1, n, t) - alpha
        u = 1.
        l = scipy.optimize.brentq(sf, 0, 1)
        mle = 1.

    else:
        cdf = lambda t: scipy.stats.binom.cdf(x, n, t) - (alpha / 2)
        sf = lambda t: scipy.stats.binom.sf(x - 1, n, t) - (alpha / 2)
        u = scipy.optimize.brentq(cdf, 0, 1)
        l = scipy.optimize.brentq(sf, 0, 1)
        mle = float(x) / n

    return mle - l, u - mle
