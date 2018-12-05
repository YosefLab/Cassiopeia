from collections import defaultdict
from . import sam

def are_disjoint(first, second):
    return first.start > second.end or second.start > first.end

def are_adjacent(first, second):
    return first.start == second.end + 1 or second.start == first.end + 1

class Interval(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end
        
    def __or__(self, other):
        if are_disjoint(self, other):
            left, right = sorted([self, other])
            if are_adjacent(self, other):
                intervals = [Interval(left.start, right.end)]
            else:
                intervals = [left, right]
        else:
            intervals = [Interval(min(self.start, other.start), max(self.end, other.end))]
            
        return DisjointIntervals(intervals)
    
    def __and__(self, other):
        if are_disjoint(self, other):
            return []
        else:
            return Interval(max(self.start, other.start), min(self.end, other.end))
        
    def __contains__(self, other):
        ''' is a strict sub-interval of '''
        return (other.start >= self.start and other.end <= self.end) and (self != other)
        
    @property    
    def comparison_key(self):
        return self.start, self.end
    
    def __lt__(self, other):
        return self.comparison_key < other.comparison_key
    
    def __repr__(self):
        return '[{0:,} - {1:,}]'.format(self.start, self.end)
    
    def __key(self):
        return (self.start, self.end)

    def __eq__(self, other):
        return self.__key() == other.__key()

    def __hash__(self):
        return hash(self.__key())
    
    def __ne__(self, other):
        return not self == other

    def __len__(self):
        return self.end - self.start + 1
    
class DisjointIntervals(object):
    def __init__(self, intervals):
        self.intervals = intervals
        
    def __len__(self):
        return len(self.intervals)
    
    @property
    def start(self):
        return min(interval.start for interval in self.intervals)
    
    @property
    def end(self):
        return max(interval.end for interval in self.intervals)
    
    def __repr__(self):
        return '{{{}}}'.format(', '.join(map(str, self.intervals)))
    
    def __getitem__(self, sl):
        return self.intervals[sl]
    
    def __or__(self, other_interval):
        disjoints = []
        
        for interval in self.intervals:
            union = interval | other_interval
            if len(union) > 1:
                disjoints.append(interval)
            else:
                other_interval = union[0]
                
        disjoints.append(other_interval)
        
        return DisjointIntervals(sorted(disjoints))
    
    def __and__(self, other_interval):
        intersections = []
        for interval in self.intervals:
            intersection = interval & other_interval
            if intersection:
                intersections.append(intersection)
                
        return DisjointIntervals(intersections)
    
    def __eq__(self, other):
        return self.intervals == other.intervals

    def __hash__(self):
        return hash(self.intervals)
    
    def __ne__(self, other):
        return not self == other
    
def get_covered(alignment):
    return Interval(*sam.query_interval(alignment))

def make_disjoint(intervals):
    disjoint = DisjointIntervals([])
    for interval in intervals:
        disjoint = disjoint | interval
    return disjoint

def get_disjoint_covered(alignments):
    intervals = [get_covered(al) for al in alignments]
    covered = make_disjoint(intervals)
    return covered

def remove_nested(alignments):
    unnecessary = set()
    covered_list = [get_covered(al) for al in alignments]
    for i, left in enumerate(covered_list):
        for j, right in enumerate(covered_list):
            if i == j:
                continue
            if left in right:
                unnecessary.add(i)
    necessary = [al for i, al in enumerate(alignments) if i not in unnecessary]
    return necessary

def make_parsimoninous(alignments):
    initial_covered = get_disjoint_covered(alignments)
    
    no_nested = remove_nested(alignments)
    interval_to_als = defaultdict(list)
    for al in no_nested:
        interval_to_als[get_covered(al)].append(al)
        
    unique_intervals = sorted(interval_to_als, key=len, reverse=True)
    remaining = unique_intervals
        
    contributes = []
    for possibly_exclude in unique_intervals:
        exclude_one = [intvl for intvl in remaining if intvl != possibly_exclude]
        now_covered = make_disjoint(exclude_one)
        if initial_covered != now_covered:
            contributes.append(possibly_exclude)
        else:
            remaining = exclude_one
            
    parsimonious = []
    for interval in contributes:
        parsimonious.extend(interval_to_als[interval])
    
    if get_disjoint_covered(parsimonious) != initial_covered:
        raise ValueError
        
    return parsimonious
