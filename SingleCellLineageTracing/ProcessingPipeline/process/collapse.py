#!/usr/bin/env python3

import argparse
import array
import heapq
import subprocess
from collections import namedtuple, Counter
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import bokeh.palettes
import yaml
import sys
import tqdm
import pysam

import networkx as nx

from sequencing import fastq, utilities, sw, sam
from sequencing import annotation as annotation_module

from collapse_cython import hq_mismatches_from_seed, hq_hamming_distance, hamming_distance_matrix, register_corrections

progress = tqdm.tqdm

CELL_BC_TAG = 'CB'
UMI_TAG = 'UR'
NUM_READS_TAG = 'ZR'
CLUSTER_ID_TAG = 'ZC'
LOC_TAG = "BC"
CO_TAG = "CO"

HIGH_Q = 31
LOW_Q = 10
N_Q = 2

cluster_fields = [
    ('cell_BC', 's'),
    ('UMI', 's'),
    ('num_reads', '06d'),
    ('cluster_id', 's'),
]
cluster_Annotation = annotation_module.Annotation_factory(cluster_fields)

def call_consensus(als, max_read_length):
    statistics = fastq.quality_and_complexity(als, max_read_length, alignments=True, min_q=30)
    shape = statistics['c'].shape

    rl_range = np.arange(max_read_length)
    
    fields = [
        ('c_above_min_q', int),
        ('c', int),
        ('average_q', float),
    ]

    stat_tuples = np.zeros(shape, dtype=fields)
    for k in ['c_above_min_q', 'c', 'average_q']:
        stat_tuples[k] = statistics[k]

    argsorted = stat_tuples.argsort()
    second_best_idxs, best_idxs = argsorted[:, -2:].T
    
    best_stats = stat_tuples[rl_range, best_idxs]

    majority = (best_stats['c'] / len(als)) > 0.5
    at_least_one_hq = best_stats['c_above_min_q'] > 0
    
    qs = np.full(max_read_length, LOW_Q, dtype=int)
    qs[majority & at_least_one_hq] = HIGH_Q
    
    ties = (best_stats == stat_tuples[rl_range, second_best_idxs])

    best_idxs[ties] = utilities.base_to_index['N']
    qs[ties] = N_Q

    consensus = pysam.AlignedSegment()
    consensus.query_sequence = ''.join(utilities.base_order[i] for i in best_idxs)
    consensus.query_qualities = array.array('B', qs)
    consensus.set_tag(NUM_READS_TAG, len(als), 'i')

    return consensus

def within_radius_of_seed(seed, als, max_hq_mismatches):
    seed_b = seed.encode()
    ds = [hq_mismatches_from_seed(seed_b, al.query_sequence.encode(), al.query_qualities, 20)
          for al in als]
    
    near_seed = []
    remaining = []
    
    for i, (d, al) in enumerate(zip(ds, als)):
        if d <= max_hq_mismatches:
            near_seed.append(al)
        else:
            remaining.append(al)
    
    return near_seed, remaining

def propose_seed(als, max_read_length):
    seq, count = Counter(al.query_sequence for al in als).most_common(1)[0]
    
    if count > 1:
        seed = seq
    else:
        seed = call_consensus(als, max_read_length).query_sequence
        
    return seed

def make_singleton_cluster(al):
    singleton = pysam.AlignedSegment()
    singleton.query_sequence = al.query_sequence
    singleton.query_qualities = al.query_qualities
    singleton.set_tag(NUM_READS_TAG, 1, 'i')
    return singleton

def form_clusters(als, max_read_length, max_hq_mismatches):
    if len(als) == 0:
        clusters = []
    
    elif len(als) == 1:
        clusters = [make_singleton_cluster(al) for al in als]
    
    else:
        seed = propose_seed(als, max_read_length)
        near_seed, remaining = within_radius_of_seed(seed, als, max_hq_mismatches)
        
        if len(near_seed) == 0:
            # didn't make progress, so give up
            clusters = [make_singleton_cluster(al) for al in als]
        
        else:
            clusters = [call_consensus(near_seed, max_read_length)] + form_clusters(remaining, max_read_length, max_hq_mismatches)
            
    return clusters

def align_clusters(first, second):
    al = sw.global_alignment(first.query_sequence, second.query_sequence)
    
    num_hq_mismatches = 0
    for q_i, t_i in al['mismatches']:
        if (first.query_qualities[q_i] > 20) and (second.query_qualities[t_i] > 20):
            num_hq_mismatches += 1
            
    return al['XO'], num_hq_mismatches

cell_key = lambda al: al.get_tag(CELL_BC_TAG)
UMI_key = lambda al: al.get_tag(UMI_TAG)
loc_key = lambda al: (al.get_tag(LOC_TAG))
empty_header = pysam.AlignmentHeader()

def sort_cellranger_bam(bam_fn, sorted_fn, sort_key, filter_func, show_progress=False):
    Path(sorted_fn).parent.mkdir(exist_ok=True)

    bam_fh = pysam.AlignmentFile(str(bam_fn))

    als = bam_fh

    relevant = list(filter(filter_func, als))

    max_read_length = 0
    total_reads_out = 0
    
    chunk_fns = []
        
    for i, chunk in enumerate(utilities.chunks(relevant, 10000000)):
        suffix = '.{:06d}.bam'.format(i)
        chunk_fn = Path(sorted_fn).with_suffix(suffix)
        sorted_chunk = sorted(chunk, key=sort_key)
    
        with pysam.AlignmentFile(str(chunk_fn), 'wb', template=bam_fh) as fh:
            for al in sorted_chunk:
                max_read_length = max(max_read_length, al.query_length)
                total_reads_out += 1
                fh.write(al)

        chunk_fns.append(chunk_fn)

    chunk_fhs = [pysam.AlignmentFile(str(fn), check_header=False, check_sq=False) for fn in chunk_fns]

    with pysam.AlignmentFile(str(sorted_fn), 'wb', template=bam_fh) as fh:
        merged_chunks = heapq.merge(*chunk_fhs, key=sort_key)

        if show_progress:
            merged_chunks = progress(merged_chunks, total=total_reads_out, desc='Merging sorted chunks')

        for al in merged_chunks:
            fh.write(al)

    for fh in chunk_fhs:
        fh.close()

    for fn in chunk_fns:
        fn.unlink()
    
    yaml_fn = sorted_fn.with_suffix('.yaml')
    stats = {
        'total_reads': total_reads_out,
        'max_read_length': max_read_length,
    }
    yaml_fn.write_text(yaml.dump(stats, default_flow_style=False))

def error_correct_UMIs(cell_group, sampleID, max_UMI_distance=1):

    UMIs = [al.get_tag(UMI_TAG) for al in cell_group]

    ds = hamming_distance_matrix(UMIs)

    corrections = register_corrections(ds, max_UMI_distance, UMIs)

    num_corrections = 0
    corrected_group = []
    ec_string = ""
    total = 0;
    corrected_names = []
    for al in cell_group:
        al_umi = al.get_tag(UMI_TAG)
        for al2 in cell_group:
            al2_umi = al2.get_tag(UMI_TAG)
            # correction keys are 'from' and values are 'to'
            # so correct al2 to al
            if al2_umi in corrections.keys() and corrections[al2_umi] == al_umi:

                bad_qname = al2.query_name
                bad_nr = bad_qname.split("_")[-1]
                qname = al.query_name
                split_qname = qname.split("_")

                prev_nr = split_qname[-1]

                split_qname[-1] = str(int(split_qname[-1]) + int(bad_nr))
                n_qname = '_'.join(split_qname)


                al.query_name = n_qname

                ec_string += al2.get_tag(UMI_TAG) + "\t" + al.get_tag(UMI_TAG) + "\t" + al.get_tag(LOC_TAG) + "\t" + al.get_tag(CO_TAG) + "\t" + str(bad_nr) + "\t" + str(prev_nr) + "\t" + str(split_qname[-1]) + "\t" + sampleID + "\n"


                # update alignment if already seen
                if al.get_tag(UMI_TAG) in list(map(lambda x: x.get_tag(UMI_TAG), corrected_group)):
                    corrected_group.remove(al)

                corrected_group.append(al)

                num_corrections += 1
                corrected_names.append(al2.get_tag(UMI_TAG))
                corrected_names.append(al.get_tag(UMI_TAG))


    for al in cell_group:
        
        # add alignments not touched during error correction back into the group to be written to file
        if al.get_tag(UMI_TAG) not in corrected_names:
            corrected_group.append(al)

    total = len(cell_group)

    return corrected_group, num_corrections, total, ec_string

def merge_annotated_clusters(biggest, other):
    merged_id = biggest.get_tag(CLUSTER_ID_TAG)
    if not merged_id.endswith('+'):
        merged_id = merged_id + '+'
    biggest.set_tag(CLUSTER_ID_TAG, merged_id, 'Z')

    total_reads = biggest.get_tag(NUM_READS_TAG) + other.get_tag(NUM_READS_TAG)
    biggest.set_tag(NUM_READS_TAG, total_reads, 'i')

    return biggest

def form_collapsed_clusters(sorted_fn,
                            max_hq_mismatches,
                            max_indels,
                            max_UMI_distance,
                            show_progress=True):

    collapsed_fn = sorted_fn.with_name(sorted_fn.stem + '_collapsed.bam')

    yaml_fn = sorted_fn.with_suffix('.yaml')
    stats = yaml.load(yaml_fn.read_text())
    max_read_length = stats['max_read_length']
    total_reads = stats['total_reads']

    sorted_als = pysam.AlignmentFile(str(sorted_fn), check_sq=False)
    if progress:
        sorted_als = progress(sorted_als, total=total_reads, desc='Collapsing')
    
    cell_groups = utilities.group_by(sorted_als, cell_key)
    
    with pysam.AlignmentFile(str(collapsed_fn), 'wb', header=empty_header) as collapsed_fh:
        for cell_BC, cell_group in cell_groups:

            for UMI, UMI_group in utilities.group_by(cell_group, UMI_key):
                clusters = form_clusters(UMI_group, max_read_length, max_hq_mismatches)
                clusters = sorted(clusters, key=lambda c: c.get_tag(NUM_READS_TAG), reverse=True)

                for i, cluster in enumerate(clusters):
                    cluster.set_tag(CELL_BC_TAG, cell_BC, 'Z')
                    cluster.set_tag(UMI_TAG, UMI, 'Z')
                    cluster.set_tag(CLUSTER_ID_TAG, str(i), 'Z')

                biggest = clusters[0]
                rest = clusters[1:]

                not_collapsed = []

                for other in rest:
                    if other.get_tag(NUM_READS_TAG) == biggest.get_tag(NUM_READS_TAG):
                        not_collapsed.append(other)
                    else:
                        indels, hq_mismatches = align_clusters(biggest, other)

                        if indels <= max_indels and hq_mismatches <= max_hq_mismatches:
                            biggest = merge_annotated_clusters(biggest, other)
                        else:
                            not_collapsed.append(other)
                
                for cluster in [biggest] + not_collapsed:
                    annotation = cluster_Annotation(cell_BC=cluster.get_tag(CELL_BC_TAG),
                                                    UMI=cluster.get_tag(UMI_TAG),
                                                    num_reads=cluster.get_tag(NUM_READS_TAG),
                                                    cluster_id=cluster.get_tag(CLUSTER_ID_TAG),
                                                   )

                    cluster.query_name = str(annotation)
                    collapsed_fh.write(cluster)


def error_correct_allUMIs(sorted_fn,
                          yaml_fn,
                            max_hq_mismatches,
                            max_indels,
                            max_UMI_distance,
                            sampleID,
                            show_progress=True):

    collapsed_fn = sorted_fn.with_name(sorted_fn.stem + '_ec.bam')
    log_fn = sorted_fn.with_name(sorted_fn.stem + '_umi_ec_log.txt')
    
    sorted_als = pysam.AlignmentFile(str(sorted_fn), check_sq=False)

    # group_by only works if sorted_als is already sorted by loc_key
    allele_groups = utilities.group_by(sorted_als, loc_key)
    
    num_corrected = 0
    total = 0

    with pysam.AlignmentFile(str(collapsed_fn), 'wb', header=sorted_als.header) as collapsed_fh:
        for allele_bc, allele_group in allele_groups:
            if max_UMI_distance > 0:
                allele_group, num_corr, tot, erstring = error_correct_UMIs(allele_group, sampleID,  max_UMI_distance)

            for a in allele_group:
                collapsed_fh.write(a)

            #log_fh.write(error_corrections)
            print(erstring, end=' ', flush=True)
            num_corrected += num_corr
            total += tot

    print(str(num_corrected) + " UMIs Corrected of " + str(total) + " (" + str(round( float(num_corrected) / total, 5)*100) + "%)", file=sys.stderr)


def split_into_guide_fastqs(collapsed_fn, cell_BC_to_guide, gemgroup, group_dir):
    clusters = pysam.AlignmentFile(str(collapsed_fn), check_sq=False)

    guide_fhs = {}

    for cluster in clusters:
        cell_BC = cluster.get_tag(CELL_BC_TAG)
        cell_BC = '{0}-{1}'.format(cell_BC.split('-')[0], gemgroup)
        guide = cell_BC_to_guide.get(cell_BC, 'unknown')
        if guide == '*':
            guide = 'unknown'

        if guide not in guide_fhs:
            guide_fn = (Path(group_dir) / guide).with_suffix('.fastq')
            guide_fhs[guide] = guide_fn.open('w')

        read = sam.mapping_to_Read(cluster)

        # temporary hack
        read.name = '{0}_{1}'.format(cell_BC, read.name.split('_', 1)[1])

        guide_fhs[guide].write(str(read))

    for guide, fh in guide_fhs.items():
        fh.close()

    guides = sorted(guide_fhs)
    return guides

def make_sample_sheet(group_dir, target, guides):
    color_list = bokeh.palettes.Category20c_20[:16] #+ bokeh.palettes.Category20b_20
    color_groups = itertools.cycle(list(zip(*[iter(color_list)]*4)))

    sample_sheet = {}

    grouped_guides = utilities.group_by(sorted(guides), lambda n: n.split('-')[0])
    for (group_name, group), color_group in zip(grouped_guides, color_groups):
        for name, color in zip(group, color_group[1:]):
            sample_sheet[name] = {
                'fastq_fns': name + '.fastq',
                'target_info': target,
                'project': 'screen',
                'color': color,
            }

    sample_sheet_fn = group_dir / 'sample_sheet.yaml'
    sample_sheet_fn.write_text(yaml.dump(sample_sheet, default_flow_style=False))

def make_cluster_fastqs(collapsed_fn, target, gemgroup, notebook=True):
    group_dir = Path(collapsed_fn).parent
    df = pd.read_csv('/home/jah/projects/britt/data/cell_identities.csv', index_col='cell_barcode') 
    cell_BC_to_guide = df['guide_identity']
    guides = split_into_guide_fastqs(collapsed_fn, cell_BC_to_guide, gemgroup, group_dir)
    make_sample_sheet(group_dir, target, guides)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--base_dir', required=True)
    parser.add_argument('--force_sort', action='store_true')
    parser.add_argument('--no_progress', action='store_true')

    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--parallel', metavar='MAX_PROCS')
    mode_group.add_argument("--correct_parallel", metavar='MAX_PROCS')
    mode_group.add_argument('--collapse', metavar='NAME')
    mode_group.add_argument("--correct", metavar="NAME")

    args = parser.parse_args()

    base_dir = Path(args.base_dir)

    if args.parallel is not None:

        sample_sheet_fn = base_dir / 'data' / 'sample_sheet.yaml'
        sample_sheet = yaml.load(sample_sheet_fn.read_text())

        max_procs = args.parallel

        if args.force_sort:
            possibly_force_sort = ['--force_sort']
        else:
            possibly_force_sort = []

        parallel_command = [
            'parallel',
            '-n', '1', 
            '--verbose',
            '--max-procs', max_procs,
            './collapse.py',
            '--no_progress',
        ] + possibly_force_sort + [
            '--base_dir', str(base_dir),
            '--collapse', ':::'
        ] + sorted(sample_sheet)

        subprocess.check_call(parallel_command)

    if args.correct_parallel is not None:
        max_procs = args.correct_parallel

        sample_sheet_fn = base_dir / 'data' / 'sample_sheet_umicorr.yaml'
        sample_sheet = yaml.load(sample_sheet_fn.read_text())

        if args.force_sort:
            possibly_force_sort = ['--force_sort']
        else:
            possibly_force_sort = []


        parallel_command = [
            'parallel',
            '-n', '1', 
            '--verbose',
            '--max-procs', max_procs,
            './collapse.py',
            '--no_progress',
        ] + possibly_force_sort + [
            '--base_dir', str(base_dir),
            '--correct', ':::'
        ] + sorted(sample_sheet)

        subprocess.check_call(parallel_command)

    elif args.collapse is not None:

        sample_sheet_fn = base_dir / 'data' / 'sample_sheet.yaml'
        sample_sheet = yaml.load(sample_sheet_fn.read_text())

        name = args.collapse
        info = sample_sheet[name]

        max_hq_mismatches = info.get('max_hq_mismatches', 10)
        max_indels = info.get('max_indels', 2)
        max_UMI_distance = info.get('max_UMI_distance', 1)
        
        show_progress = not args.no_progress

        input_fn = Path(info['cellranger_dir']) / 'outs' / 'possorted_genome_bam.bam'
        sorted_fn = (base_dir / 'data' / name / name).with_suffix('.bam')

        sort_key = lambda al: (al.get_tag(CELL_BC_TAG), al.get_tag(UMI_TAG))
        filter_func = lambda al: al.has_tag(CELL_BC_TAG)

        if not sorted_fn.exists() or args.force_sort:
            sort_cellranger_bam(input_fn, sorted_fn, sort_key, filter_func, show_progress=show_progress)

        form_collapsed_clusters(sorted_fn,
                                max_hq_mismatches,
                                max_indels,
                                max_UMI_distance,
                                show_progress=show_progress
                               )
            
        #make_cluster_fastqs(collapsed_fn, target, gemgroup)

    elif args.correct is not None:

        sample_sheet_fn = base_dir / 'data' / 'sample_sheet_umicorr.yaml'
        sample_sheet = yaml.load(sample_sheet_fn.read_text())

        name = args.correct
        info = sample_sheet[name]

        bam_dir = info.get('cellranger_dir')
        max_hq_mismatches = info.get("max_hq_mismatches", 10)
        max_indels = info.get('max_indels', 2)
        max_UMI_distance = info.get("max_UMI_distance", 1)

        sort_key = lambda al: (al.get_tag(LOC_TAG), -1*int(al.query_name.split("_")[-1]))

        show_progress = not args.no_progress

        input_fn = (base_dir / bam_dir / name).with_suffix('.moltable.bam')
        sorted_fn = (base_dir / bam_dir/ name).with_suffix('.moltable_sorted.bam')
        yaml_fn = (base_dir / 'data' / 'sample_sheet.yaml')

        filter_func = lambda al: al.has_tag(LOC_TAG) or al.has_tag(CELL_BC_TAG)

        #if not sorted_fn.exists() or args.force_sort:
        sort_cellranger_bam(input_fn, sorted_fn, sort_key, filter_func, show_progress = show_progress)

        error_correct_allUMIs(sorted_fn, 
                              yaml_fn,
                              max_hq_mismatches,
                              max_indels,
                              max_UMI_distance,
                              info.get("id"), 
                            show_progress = show_progress)
