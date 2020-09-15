"""
This file contains all functions pertaining to UMI collapsing and preprocessing.
Invoked through pipeline.py and supports the collapseUMIs and errorCorrectUMIs functions. 
"""

##TODO: richardyz98: make configuration files that specify bam labels as an optional input
##TODO: environment files

from pathlib import Path
import subprocess
import os

from typing import Callable
import numpy as np
import array
import heapq
import pysam
import yaml
from tqdm.auto import tqdm
from collections import namedtuple, Counter

from hits import fastq, utilities, sw, sam
from hits import annotation as annotation_module

from .collapse_cython import (
    hq_mismatches_from_seed,
    hq_hamming_distance,
    hamming_distance_matrix,
    register_corrections,
)

####################Utils for Collapsing UMIs#####################

progress = tqdm
empty_header = pysam.AlignmentHeader()
cluster_fields = [
    ("cell_BC", "s"),
    ("UMI", "s"),
    ("num_reads", "06d"),
    ("cluster_id", "s"),
]
cluster_Annotation = annotation_module.Annotation_factory(cluster_fields)

CELL_BC_TAG = "CB"
UMI_TAG = "UR"
NUM_READS_TAG = "ZR"
CLUSTER_ID_TAG = "ZC"
LOC_TAG = "BC"

cell_key = lambda al: al.get_tag(CELL_BC_TAG)
UMI_key = lambda al: al.get_tag(UMI_TAG)
loc_key = lambda al: (al.get_tag(LOC_TAG))

N_Q = 2
HIGH_Q = 31
LOW_Q = 10


def sort_cellranger_bam(
    bam_fn: str,
    sorted_fn: str,
    sort_key: Callable[[pysam.AlignedSegment], str],
    filter_func: Callable[[pysam.AlignedSegment], str],
    show_progress: bool = False,
) -> (int, int):
    """Sorts aligned segments (representing a read) in the BAM file according to a specified key.

    Splits the BAM file into chunks and sorts relevant aligned sequences within chunks, filtering relevant aligned sequences with a
    specified key. Exports the total number of reads sorted and the max read length to a .yaml file to be used in downstream analyses
    (specifically in form_collapsed_clusters).

    Args:
        bam_fn: The file name of the BAM to be sorted.
        sorted_fn: The file name of the output BAM after sorting.
        sort_key: A function specifying the key by which to sort the aligned sequences.
        filter_func: A function specifying the key by which to filter out irrelevant sequences.
        show_progress: Allow progress bar to be shown.

    Returns:
        max_read_length: The max read length and the
        total_reads_out: The total number of relevant reads sorted.
    """
    Path(sorted_fn).parent.mkdir(exist_ok=True)

    bam_fh = pysam.AlignmentFile(str(bam_fn))

    relevant = filter(filter_func, bam_fh)

    max_read_length = 0
    total_reads_out = 0

    chunk_fns = []

    for i, chunk in enumerate(utilities.chunks(relevant, 10000000)):
        suffix = ".{:06d}.bam".format(i)
        chunk_fn = Path(sorted_fn).with_suffix(suffix)
        sorted_chunk = sorted(chunk, key=sort_key)

        with pysam.AlignmentFile(str(chunk_fn), "wb", template=bam_fh) as fh:
            for al in sorted_chunk:
                max_read_length = max(max_read_length, al.query_length)
                total_reads_out += 1
                fh.write(al)

        chunk_fns.append(chunk_fn)

    chunk_fhs = [
        pysam.AlignmentFile(str(fn), check_header=False, check_sq=False)
        for fn in chunk_fns
    ]

    with pysam.AlignmentFile(str(sorted_fn), "wb", template=bam_fh) as fh:
        merged_chunks = heapq.merge(*chunk_fhs, key=sort_key)

        if show_progress:
            merged_chunks = progress(
                merged_chunks, total=total_reads_out, desc="Merging sorted chunks"
            )

        for al in merged_chunks:
            fh.write(al)

    for fh in chunk_fhs:
        fh.close()

    for fn in chunk_fns:
        fn.unlink()

    yaml_fn = Path(sorted_fn).with_suffix(".yaml")
    stats = {
        "total_reads": total_reads_out,
        "max_read_length": max_read_length,
    }

    with open(yaml_fn, "w") as f:
        f.write(yaml.dump(stats, default_flow_style=False))

    return max_read_length, total_reads_out


def form_collapsed_clusters(
    sorted_fn: Callable[[pysam.AlignedSegment], str],
    max_hq_mismatches: int,
    max_indels: int,
    show_progress=False,
):
    """Aggregates together aligned segments (reads) by their UMI and also collapses similar UMIs.

    Clusters aligned segments with similar UMIs, with clusters represented by a single annotated aligned segment that records the UMI,
    Cell_BC, and number of reads. Then saves these annotated aligned segments representing clusters to a BAM file.

    Args:
        sorted_fn: The file name of the sorted BAM.
        max_hq_mismatches: A threshold specifying the maximum number of high quality mismatches between the seqeunces of 2 aligned segments to be collapsed.
        max_indels: A threshold specifying the maximum number of differing indels allowed between the sequences of 2 aligned segments to be collapsed.
        show_progress: Allow progress bar to be shown.
    """

    collapsed_fn = ".".join(sorted_fn.split(".")[:-1]) + ".collapsed.bam"

    yaml_fn = ".".join(sorted_fn.split(".")[:-1]) + ".yaml"
    stats = yaml.load(open(yaml_fn, "r"))
    max_read_length = stats["max_read_length"]
    total_reads = stats["total_reads"]

    sorted_als = pysam.AlignmentFile(str(sorted_fn), check_sq=False)
    if show_progress:
        sorted_als = progress(sorted_als, total=total_reads, desc="Collapsing")

    cell_groups = utilities.group_by(sorted_als, cell_key)

    with pysam.AlignmentFile(
        str(collapsed_fn), "wb", header=empty_header
    ) as collapsed_fh:
        for cell_BC, cell_group in cell_groups:

            for UMI, UMI_group in utilities.group_by(cell_group, UMI_key):
                clusters = form_clusters(UMI_group, max_read_length, max_hq_mismatches)
                clusters = sorted(
                    clusters, key=lambda c: c.get_tag(NUM_READS_TAG), reverse=True
                )

                for i, cluster in enumerate(clusters):
                    cluster.set_tag(CELL_BC_TAG, cell_BC, "Z")
                    cluster.set_tag(UMI_TAG, UMI, "Z")
                    cluster.set_tag(CLUSTER_ID_TAG, str(i), "Z")

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
                    annotation = cluster_Annotation(
                        cell_BC=cluster.get_tag(CELL_BC_TAG),
                        UMI=cluster.get_tag(UMI_TAG),
                        num_reads=cluster.get_tag(NUM_READS_TAG),
                        cluster_id=cluster.get_tag(CLUSTER_ID_TAG),
                    )

                    cluster.query_name = str(annotation)
                    collapsed_fh.write(cluster)


def form_clusters(als: list, max_read_length: int, max_hq_mismatches: int) -> list:
    """Forms clusters from a list of aligned segments (reads).

    Forms clusters of aligned segments that have similar UMIs, creating an aligned segment that represents the consensus of a clsuter.
    Goes through the list, finding the most frequent sequences and aggregating all close sequences in that cluster. Then clusters on
    the remaining aligned segments.

    Args:
        als: A list of aligned segments.
        max_read_length: The maximimum read length in the dataset.
        max_hq_mismatches: A threshold specifying the maximum number of high quality mismatches between the seqeunces of 2 aligned segments to be collapsed.

    Returns:
        clusters: A list of annotated aligned segments representing the consensus of each cluster.
    """

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
            clusters = [call_consensus(near_seed, max_read_length)] + form_clusters(
                remaining, max_read_length, max_hq_mismatches
            )

    return clusters


def align_clusters(
    first: pysam.AlignedSegment, second: pysam.AlignedSegment
) -> (int, int):
    """Calculates the number of indel mismatches and high quality mismatches between the sequences of 2 aligned segments (reads).

    Args:
        first: The first aligned segment.
        second: The second aligned segment.

    Returns:
        indels: The number of differing indels.
        num_hq_mismatches: The number of high quality mismatches.
    """

    al = sw.global_alignment(first.query_sequence, second.query_sequence)

    num_hq_mismatches = 0
    for q_i, t_i in al["mismatches"]:
        if (first.query_qualities[q_i] > 20) and (second.query_qualities[t_i] > 20):
            num_hq_mismatches += 1

    indels = al["XO"]
    return indels, num_hq_mismatches


def within_radius_of_seed(seed: str, als: list, max_hq_mismatches: int) -> (list, list):
    """Returns the aligned segments (reads) in a list with sequences that are close enough to specified seed

    Finds the aligned segments in the input list that have fewer than a specified threshold of high quality mismatches
    between their sequence and the seed sequence.

    Args:
        seed: A specified sequence.
        als: A list of aligned segments.
        max_hq_mismatches: A threshold specifying the maximum number of high quality mismatches between the seqeunces of 2 aligned segments to be collapsed.

    Returns:
        near_seed: List of aligned segments with sequences that are within radius of seed.
        remaining: List of aligned segments with sequences that are not within radius of seed.
    """
    seed_b = seed.encode()
    ds = [
        hq_mismatches_from_seed(
            seed_b, al.query_sequence.encode(), al.query_qualities, 20
        )
        for al in als
    ]

    near_seed = []
    remaining = []

    for i, (d, al) in enumerate(zip(ds, als)):
        if d <= max_hq_mismatches:
            near_seed.append(al)
        else:
            remaining.append(al)

    return near_seed, remaining


def propose_seed(als: list, max_read_length: int) -> str:
    """Proposeses a 'seed' around which a cluster is formed.

    Generates a seed around which clusters of aligned segments (reads) are formed in within_radius_of_seed.
    The method of clustering is defined in the documentation of that function. The seed is defined as the
    most frequent sequence in the list of aligned segments, or the consensus sequence of the aligned segments
    if there is no most frequent sequence.

    Args:
        als: A list of aligned segments.
        max_read_length: the max read length in the sorted BAM.

    Returns:
        seed: A sequence as the seed.
    """
    seq, count = Counter(al.query_sequence for al in als).most_common(1)[0]

    if count > 1:
        seed = seq
    else:
        seed = call_consensus(als, max_read_length).query_sequence

    return seed


def make_singleton_cluster(al: pysam.AlignedSegment) -> pysam.AlignedSegment:
    """Defines a cluster for a single aligned segment.

    In the forming of clusters, if a cluster only contains one aligned segment (read), then an annotated
    aligned segment is generated for this singleton aligned segment for comptability reasons.

    Args:
        al: The single aligned segment.

    Returns:
        singleton: A single annotated aligned segment representing the singleton cluster.
    """
    singleton = pysam.AlignedSegment()
    singleton.query_sequence = al.query_sequence
    singleton.query_qualities = al.query_qualities
    singleton.set_tag(NUM_READS_TAG, 1, "i")
    return singleton


def call_consensus(als: list, max_read_length: int) -> pysam.AlignedSegment:
    """Generates a consensus annotated aligned segment for a list of aligned segments (reads).

    For a list of aligned segments, creates a consensus sequence by taking the most frequent high quality base for each index in the sequence. Then it constructs an
    annotated aligned segment recording this sequence, as well as the qualities and the total number of reads. The qualities of each base in the final sequence is
    recorded as high only if a majority of the original aligned segments had had that base at that index and at least one of them was high quality.

    Args:
        als: A list of aligned segments.
        max_read_length: The max read length in the dataset.

    Returns:
        consensus: A consensus annotated aligned segment.
    """
    statistics = fastq.quality_and_complexity(
        als, max_read_length, alignments=True, min_q=30
    )

    shape = statistics["c"].shape

    rl_range = np.arange(max_read_length)

    fields = [
        ("c_above_min_q", int),
        ("c", int),
        ("average_q", float),
    ]

    stat_tuples = np.zeros(shape, dtype=fields)
    for k in ["c_above_min_q", "c", "average_q"]:
        stat_tuples[k] = statistics[k]

    argsorted = stat_tuples.argsort()
    second_best_idxs, best_idxs = argsorted[:, -2:].T

    best_stats = stat_tuples[rl_range, best_idxs]

    majority = (best_stats["c"] / len(als)) > 0.5
    at_least_one_hq = best_stats["c_above_min_q"] > 0

    qs = np.full(max_read_length, LOW_Q, dtype=int)
    qs[majority & at_least_one_hq] = HIGH_Q

    ties = best_stats == stat_tuples[rl_range, second_best_idxs]

    best_idxs[ties] = utilities.base_to_index["N"]
    qs[ties] = N_Q

    consensus = pysam.AlignedSegment()
    consensus.query_sequence = "".join(utilities.base_order[i] for i in best_idxs)
    consensus.query_qualities = array.array("B", qs)
    consensus.set_tag(NUM_READS_TAG, len(als), "i")

    return consensus


def merge_annotated_clusters(
    biggest: pysam.AlignedSegment, other: pysam.AlignedSegment
) -> pysam.AlignedSegment:
    """Merges 2 annotated clusters together.

    Merges 2 annotated aligned segments, each representing a cluster. Merges the smaller into the larger.
    Adds the read number of the 2nd cluster to the first.

    Args:
        biggest: The larger of the 2 clusters, with a higher read number.
        other: The smaller of the 2 clusters, with a lower read number.

    Returns:
        biggest: The annotated aligned segment representing the merged cluster.
    """

    merged_id = biggest.get_tag(CLUSTER_ID_TAG)
    if not merged_id.endswith("+"):
        merged_id = merged_id + "+"
    biggest.set_tag(CLUSTER_ID_TAG, merged_id, "Z")

    total_reads = biggest.get_tag(NUM_READS_TAG) + other.get_tag(NUM_READS_TAG)
    biggest.set_tag(NUM_READS_TAG, total_reads, "i")

    return biggest


####################Utils for Error Correcting UMIs#####################


def error_correct_allUMIs(
    sorted_fn, max_UMI_distance, sampleID, log_fh=None, show_progress=True
):

    collapsed_fn = sorted_fn.with_name(sorted_fn.stem + "_ec.bam")
    log_fn = sorted_fn.with_name(sorted_fn.stem + "_umi_ec_log.txt")

    sorted_als = pysam.AlignmentFile(str(sorted_fn), check_sq=False)

    # group_by only works if sorted_als is already sorted by loc_key
    allele_groups = utilities.group_by(sorted_als, loc_key)

    num_corrected = 0
    total = 0

    with pysam.AlignmentFile(
        str(collapsed_fn), "wb", header=sorted_als.header
    ) as collapsed_fh:
        for allele_bc, allele_group in allele_groups:
            if max_UMI_distance > 0:
                allele_group, num_corr, tot, erstring = error_correct_UMIs(
                    allele_group, sampleID, max_UMI_distance
                )

            for a in allele_group:
                collapsed_fh.write(a)

            # log_fh.write(error_corrections)
            if log_fh is None:
                print(erstring, end=" ")
                sys.stdout.flush()
            else:
                with open(log_fh, "a") as f:
                    f.write(erstring)

            num_corrected += num_corr
            total += tot

    print(
        str(num_corrected)
        + " UMIs Corrected of "
        + str(total)
        + " ("
        + str(round(float(num_corrected) / total, 5) * 100)
        + "%)",
        file=sys.stderr,
    )


def error_correct_UMIs(cell_group, sampleID, max_UMI_distance=1):

    UMIs = [al.get_tag(UMI_TAG) for al in cell_group]

    ds = hamming_distance_matrix(UMIs)

    corrections = register_corrections(ds, max_UMI_distance, UMIs)

    num_corrections = 0
    corrected_group = []
    ec_string = ""
    total = 0
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
                n_qname = "_".join(split_qname)

                al.query_name = n_qname

                ec_string += (
                    al2.get_tag(UMI_TAG)
                    + "\t"
                    + al.get_tag(UMI_TAG)
                    + "\t"
                    + al.get_tag(LOC_TAG)
                    + "\t"
                    + al.get_tag(CO_TAG)
                    + "\t"
                    + str(bad_nr)
                    + "\t"
                    + str(prev_nr)
                    + "\t"
                    + str(split_qname[-1])
                    + "\t"
                    + sampleID
                    + "\n"
                )

                # update alignment if already seen
                if al.get_tag(UMI_TAG) in list(
                    map(lambda x: x.get_tag(UMI_TAG), corrected_group)
                ):
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
