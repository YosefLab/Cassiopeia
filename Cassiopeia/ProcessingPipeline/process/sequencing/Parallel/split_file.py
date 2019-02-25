import os
import numpy as np
from itertools import chain
from functools import partial
from collections import deque
from sequencing import fastq

def generate_suffix(num_pieces, which_piece):
    ''' Suffix to append to piece number which_piece out of num_pieces.
    '''
    if num_pieces == 1:
        suffix = ''
    else:
        # Want a fixed-width format for aesthetic reasons. 
        digits = len(str(num_pieces - 1))
        suffix = '.{0:d}.{1:0{width}d}'.format(num_pieces,
                                               which_piece,
                                               width=digits,
                                              )
    return suffix

def piece(file_name, num_pieces, which_piece, file_format, key=None):
    ''' An iterator over the lines in piece number which_piece out of
        num_pieces in file_name, with chunks defined by file_format. 
    '''
    if file_format == 'fastq':
        find_next_chunk = _find_next_fastq_read
    elif file_format == 'sam':
        find_next_chunk = partial(_find_next_sam_chunk, key=key)

    if which_piece == -1:
        # Sentinel value indicating merged experimnt
        which_piece = 0

    this_start = _find_start(file_name,
                             num_pieces,
                             which_piece,
                             find_next_chunk,
                            )
    next_start = _find_start(file_name,
                             num_pieces,
                             which_piece + 1,
                             find_next_chunk,
                            )
    
    fh = open(file_name, 'rb')
    fh.seek(this_start)
    
    while fh.tell() < next_start:
        yield fh.readline().decode()

    fh.close()

def interleaved_piece(file_name, num_pieces, which_piece):
    ''' An iterator over the lines of fastq chunks which are equivalent to 
        which_piece mod num_pieces.
    '''
    if which_piece == -1:
        # Sentinel value indicating merged experimnt
        which_piece = 0

    for i, chunk in enumerate(fastq.get_line_groups(file_name)):
        if i % num_pieces == which_piece:
            for line in chunk:
                yield line

def compromise_piece(file_name, num_pieces, which_piece, factor):
    indices = [num_pieces * i + which_piece for i in range(factor)]
    pieces = [piece(file_name, num_pieces * factor, i, 'fastq')
              for i in indices]
    return chain(*pieces)

def _find_start(file_name, num_pieces, which_piece, find_next_chunk):
    ''' Finds the offset into file_name at which piece number which_piece out
        out of num_pieces starts, given that file_name consists of chunks which
        should not be broken up. Chunks are defined by find_next_chunk, a
        function that takes an open file and finds the start
        of the next chunk.
        Nasty gotcha - this doesn't necessarily return the same starting read
        in R1 and R2 files if the read lengths for R1 and R2 are different.
    '''
    file_size = os.path.getsize(file_name)
    if which_piece >= num_pieces:
        return file_size

    nominal_piece_size = int(np.ceil(float(file_size) / num_pieces))
    nominal_start = nominal_piece_size * which_piece
    
    with open(file_name, 'rb') as fh:
        fh.seek(nominal_start)
        next_start = find_next_chunk(fh)
    
    return next_start 

def preserve_position(func):
    ''' A decorator that takes a function that takes a filehandle and makes the
        function preserve the position of the filehandle by seeking back to
        the initial position after calling the fucntion.
    '''
    def decorated_func(fh, *args, **kwargs):
        initial_position = fh.tell()
        r = func(fh, *args, **kwargs)
        fh.seek(initial_position)
        return r

    return decorated_func

@preserve_position
def _find_next_fastq_read(fh):
    ''' Returns the position in fh of the start of the next fastq read.
        Fastq reads are sets of four lines, the first of which begins with '@'
        and the third of which begins with '+'.
    '''
    # If we are at the beginning of the file, we are at the beginning of a read.
    if fh.tell() == 0:
        return 0

    # Now we are guaranteed to not be at the beginning of the file.
    # Move to the next beginning of a line, which may be the current position.
    fh.seek(-1, os.SEEK_CUR)
    fh.readline().decode()
    
    # Find the next beginning of a set of four lines for which the first
    # starts with '@' and the third starts with '+'.
    positions = deque()
    lines = deque()
    for i in range(4):
        position = fh.tell()
        positions.append(position)
        line = fh.readline().decode()
        if not line:
            return fh.tell()
        lines.append(line)

    while not(lines[0].startswith('@') and lines[2].startswith('+')):
        position = fh.tell()
        line = fh.readline().decode()
        if not line:
            return fh.tell()
        positions.popleft()
        positions.append(position)
        lines.popleft()
        lines.append(line)
    
    return positions[0]

@preserve_position
def _find_next_sam_chunk(fh, key):
    ''' Returns the position in fh of the start of the next line that is
        transformed to a different value by key than the line before it.
        If key == None, use the line itself as the value.
    '''
    if key == None:
        key = lambda x: x

    _advance_to_next_line(fh)
    
    previous_line = _get_previous_line(fh)
    if not previous_line:
        # If there is no previous line, then the start of the next fragment
        # is the start of the reads.
        return _find_start_of_reads(fh)
    
    previous_value = key(previous_line)
    while True:
        start_of_line = fh.tell()
        current_line = fh.readline().decode()
        if not current_line:
            # Reached the end of the file.
            break
        current_value = key(current_line)
        if current_value != previous_value:
            fh.seek(start_of_line)
            break

    return fh.tell()
    
@preserve_position
def _find_start_of_reads(fh):
    ''' Find the position in fh immediately following the last SAM header line.
    '''
    fh.seek(0)
    while True:
        start_of_line = fh.tell()
        line = fh.readline().decode()
        if not line:
            start_of_reads = start_of_line
            break
        elif not line.startswith('@'):
            start_of_reads = start_of_line
            break

    return start_of_reads

def _advance_to_next_line(fh):
    ''' Advances fh to the first position that is greater than or equal to the
        current position, greater than or equal to the start of reads, and
        follows a newline.
    '''
    start_of_reads = _find_start_of_reads(fh)
    
    if fh.tell() <= start_of_reads:
        fh.seek(start_of_reads)
        return
    fh.seek(-1, os.SEEK_CUR)
    fh.readline().decode()

@preserve_position
def _get_previous_line(fh):
    ''' Given fh in a state such that the current position immediately follows
        a newline, returns the line preceding the current position
        if it exists or None if it doesn't. The start of the file
        is considered to follow a newline. The SAM header has be ignored.
    '''
    start_of_reads = _find_start_of_reads(fh)
    new_lines = 0
    # Move backwards one char at a time until hitting start_of_reads or seeing
    # two newlines.
    while fh.tell() > start_of_reads:
        fh.seek(-1, os.SEEK_CUR)
        char = fh.read(1)
        if char == '\n':
            new_lines += 1
            if new_lines == 2:
                break
        if fh.tell() == start_of_reads + 1:
            # Potentially at the beginning of the file, which is treated as if
            # it follows a newline.
            new_lines += 1
            break
        fh.seek(-1, os.SEEK_CUR)
    if new_lines == 2:
        return fh.readline().decode()
    else:
        # Reached the beginning of the reads without hitting 2 newlines, so
        # there is no previous line.
        return None
