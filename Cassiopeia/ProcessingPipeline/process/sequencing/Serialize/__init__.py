import shutil
from functools import partial, reduce
from .. import external_sort, sam
from . import log

def _concatenate(input_file_names, output_file_name):
    with open(output_file_name, 'w') as output_file:
        for input_file_name in input_file_names:
            shutil.copyfileobj(open(input_file_name), output_file)

def _merge_sam_files(input_file_names, merged_file_name, are_sorted=False):
    ''' Merges a list of sam files.
        Requires all input files to have the same @SQ lines.
    '''
    sq_lines = None
    for file_name in input_file_names:
        these_sq_lines = sam.get_sq_lines(file_name)
        if sq_lines == None:
            sq_lines = these_sq_lines
        else:
            if sq_lines != these_sq_lines:
                raise ValueError('@SQ lines do not agree')

    with open(merged_file_name, 'w') as merged_file:
        for sq_line in sq_lines:
            merged_file.write(sq_line)

        input_files = [sam.open_to_reads(fn) for fn in input_file_names]
        if are_sorted:
            for line in external_sort.merge(input_files):
                merged_file.write(line)
        else:
            for input_file in input_files:
                shutil.copyfileobj(input_file, merged_file)

special_mergers = {
    'bam': sam.merge_sorted_bam_files,
    'bam_by_name': partial(sam.merge_sorted_bam_files, by_name=True),
    'sam_unsorted':  partial(_merge_sam_files, are_sorted=False),
    'sam_sorted': partial(_merge_sam_files, are_sorted=True),
    'fastq': _concatenate,
    'fasta': _concatenate,
    'concatenate': _concatenate,
}

def merge_files(input_file_names, output_file_name, file_format, fast=False):
    try:
        if fast:
            file_format.fast_merge(input_file_names, output_file_name)
        elif isinstance(file_format, str):
            special_mergers[file_format](input_file_names, output_file_name)
        else:
            processed_inputs = (file_format.read_file(fn) for fn in input_file_names)
            merged_data = reduce(file_format.combine_data, processed_inputs)
            file_format.write_file(merged_data, output_file_name)
    except IOError:
        print(output_file_name)
        raise
