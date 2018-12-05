import heapq
import tempfile
import os

def merge(input_files):
    ''' Given input_files, all sorted, returns a generator of the merged
        lines.
    '''
    return heapq.merge(*input_files)

def _sort_chunk(chunk, chunk_file_names):
    ''' Sorts the lines in chunk, writes the sorted lines to a temporary
        file, and appends the temporary file name to chunk_file_names.
    '''
    chunk.sort()
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as chunk_file:
        chunk_file_names.append(chunk_file.name)
        for line in chunk:
            chunk_file.write(line)

def external_sort(input_file, sorted_file, chunk_size=3e6):
    ''' Writes the lines in file_handle in sorted order to sorted_file.
        Never loads more than chunk_size lines into memory at the same time.
    '''
    chunk_file_names = []
    
    chunk = []
    for line in input_file:
        chunk.append(line)

        if len(chunk) == chunk_size:
            # When memory is "full", sort and dump to a file.
            _sort_chunk(chunk, chunk_file_names)
            chunk = []
    
    # Sort and dump the last partial fill-up of memory.
    if len(chunk) > 0:
        _sort_chunk(chunk, chunk_file_names)

    chunk_files = [open(fn) for fn in chunk_file_names]
    for line in merge(chunk_files):
        sorted_file.write(line)

    for chunk_file in chunk_files:
        chunk_file.close()

    for chunk_file_name in chunk_file_names:
        os.remove(chunk_file_name)

def sort_simple(file_handle, sorted_file_name):
    ''' Writes the lines in file_handle in sorted order to sorted_file_name.
    '''
    lines = file_handle.readlines()
    lines.sort()
    with open(sorted_file_name, 'w') as sorted_fh:
        for line in lines:
            sorted_fh.write(line)
