"""
This file contains all functions pertaining to UMI collapsing and preprocessing.
Invoked through pipeline.py and supports the collapseUMIs and 
errorCorrectUMIs functions. 
"""

import array
import heapq
import logging
import numpy as np
import pandas as pd
import pysam
import os
import subprocess
import yaml

from collections import Counter, defaultdict, namedtuple
from pathlib import Path
from tqdm.auto import tqdm
from typing import Callable, List

from hits import annotation as annotation_module
from hits import fastq, utilities, sw, sam

from .collapse_cython import (
    hq_mismatches_from_seed,
    hq_hamming_distance,
    hamming_distance_matrix,
    register_corrections,
)

progress = tqdm
empty_header = pysam.AlignmentHeader()
cluster_fields = [
    ("cell_BC", "s"),
    ("UMI", "s"),
    ("num_reads", "06d"),
    ("cluster_id", "s"),
]
cluster_Annotation = annotation_module.Annotation_factory(cluster_fields)

# TODO(richardyz98@): add user specified bam file tags as an option to config files

global CELL_BC_TAG, UMI_TAG, NUM_READS_TAG, CLUSTER_ID_TAG
global LOC_TAG, N_Q, HIGH_Q, LOW_Q

CELL_BC_TAG = "CB"  # The tag denoting the field that records the cellBC for
# each aligned segment.
UMI_TAG = "UR"  # The tag denothing the field that records the UMI for each
# aligned segment.
NUM_READS_TAG = "ZR"  # The tag denothing the field that records the UMI for
# each aligned segment.
CLUSTER_ID_TAG = "ZC"  # The tag denothing the field that records the cluster ID
# for each annotated aligned segment represnting a cluster
# of aligned segments.
# LOC_TAG = "BC"

N_Q = 2  # The default quality value indicating a consensus could not be reached
# for a base in the sequence for the consensus aligned segment.
HIGH_Q = 31  # The default high quality value when annotating the qualities in a
# consensus aligned segment.
LOW_Q = 10  # The default low quality value when annotating the qualities in a
# consensus aligned segment.

cell_key = lambda al: al.get_tag(CELL_BC_TAG)
UMI_key = lambda al: al.get_tag(UMI_TAG)
# loc_key = lambda al: (al.get_tag(LOC_TAG))

sort_key = lambda al: (al.get_tag(CELL_BC_TAG), al.get_tag(UMI_TAG))
filter_func = lambda al: al.has_tag(CELL_BC_TAG)


####################Utils for Collapsing UMIs#####################


def sort_cellranger_bam(
    bam_fp: str,
    sorted_fn: str,
    sort_key: Callable[[pysam.AlignedSegment], str] = sort_key,
    filter_func: Callable[[pysam.AlignedSegment], str] = filter_func,
    show_progress: bool = False,
) -> (int, int):
    """Sorts aligned segments (representing a read) in the BAM file according
    to a specified key.

    Splits the BAM file into chunks and sorts relevant aligned sequences within
    chunks, filtering relevant aligned sequences with a specified key.

    Args:
      bam_fp: The file path of the BAM to be sorted.
      sorted_fn: The file name of the output BAM after sorting.
      sort_key: A function specifying the key by which to sort the aligned sequences.
      filter_func: A function specifying the key by which to filter out
        irrelevant sequences.
      show_progress: Allow progress bar to be shown.

    Returns:
      max_read_length: The max read length and the
      total_reads_out: The total number of relevant reads sorted.
    """
    Path(sorted_fn).parent.mkdir(exist_ok=True)

    bam_fh = pysam.AlignmentFile(str(bam_fp))

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
                merged_chunks,
                total=total_reads_out,
                desc="Merging sorted chunks",
            )

        for al in merged_chunks:
            fh.write(al)

    for fh in chunk_fhs:
        fh.close()

    for fn in chunk_fns:
        fn.unlink()

    return max_read_length, total_reads_out


def form_collapsed_clusters(
    sorted_fn: Callable[[pysam.AlignedSegment], str],
    max_hq_mismatches: int,
    max_indels: int,
    show_progress: bool = False,
):
    """Aggregates together aligned segments (reads) that share UMIs if their
    sequences are close.

    Clusters aligned segments within a UMI and with sequences are less than a
    threshold of mismatches from one another, effectively determining the
    sequence and number of reads for each UMI while accounting for sequencing
    errors. If ties do not exist, the most frequent sequence in reads with a
    certain UMI is chosen as the 'true' sequence for that UMI, otherwise a
    consensus is created between sequences. Then, it further attempts to
    cluster clusters with similar sequences after the consensuses are created.
    Clusters are represented by a single annotated aligned segment that records
    the UMI, cellBC, and number of reads. Then saves these annotated aligned
    segments representing clusters to a BAM file. Multiple clusters can be
    created for a given UMI, if there are multiple clusters with significantly
    different sequences that could not be resolved at this step.

    Args:
      sorted_fn: The file name of the sorted BAM.
      max_hq_mismatches: A threshold specifying the maximum number of high
        quality mismatches between the seqeunces of 2 aligned segments to be
        collapsed.
      max_indels: A threshold specifying the maximum number of differing indels
        allowed between the sequences of 2 aligned segments to be collapsed.
      show_progress: Allow progress bar to be shown.
    """

    collapsed_fn = ".".join(sorted_fn.split(".")[:-1]) + ".collapsed.bam"
    sorted_als = pysam.AlignmentFile(str(sorted_fn), check_sq=False)

    total_reads = 0
    max_read_length = 0
    for al in sorted_als:
        total_reads += 1
        max_read_length = max(max_read_length, al.query_length)

    # Read in the AlignmentFile again as iterating over it in the previous for
    # loop has destructively removed all alignments from the file object
    sorted_als = pysam.AlignmentFile(str(sorted_fn), check_sq=False)

    if show_progress:
        sorted_als = progress(sorted_als, total=total_reads, desc="Collapsing")

    cell_groups = utilities.group_by(sorted_als, cell_key)

    with pysam.AlignmentFile(
        str(collapsed_fn), "wb", header=empty_header
    ) as collapsed_fh:
        for cell_BC, cell_group in cell_groups:
            for UMI, UMI_group in utilities.group_by(cell_group, UMI_key):
                clusters = form_clusters(
                    UMI_group, max_read_length, max_hq_mismatches
                )
                clusters = sorted(
                    clusters,
                    key=lambda c: c.get_tag(NUM_READS_TAG),
                    reverse=True,
                )

                for i, cluster in enumerate(clusters):
                    cluster.set_tag(CELL_BC_TAG, cell_BC, "Z")
                    cluster.set_tag(UMI_TAG, UMI, "Z")
                    cluster.set_tag(CLUSTER_ID_TAG, str(i), "Z")

                biggest = clusters[0]
                rest = clusters[1:]

                not_collapsed = []

                for other in rest:
                    if other.get_tag(NUM_READS_TAG) == biggest.get_tag(
                        NUM_READS_TAG
                    ):
                        not_collapsed.append(other)
                    else:
                        indels, hq_mismatches = align_clusters(biggest, other)

                        if (
                            indels <= max_indels
                            and hq_mismatches <= max_hq_mismatches
                        ):
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


def form_clusters(
    als: List[pysam.AlignedSegment],
    max_read_length: int,
    max_hq_mismatches: int,
) -> List[pysam.AlignedSegment]:
    """Forms clusters from a list of aligned segments (reads).

    Forms clusters of aligned segments that have similar sequences, creating
    an aligned segment that represents the consensus of a cluster. Goes through
    the list, finding the most frequent sequences and aggregating all close
    sequences in that cluster. If there are ties, then a consensus is created
    from the sequences. Then clusters on the remaining aligned segments.

    Args:
      als: A list of aligned segments.
      max_read_length: The maximimum read length in the dataset.
      max_hq_mismatches: A threshold specifying the maximum number of high
        quality mismatches between the seqeunces of 2 aligned segments to be
        collapsed.

    Returns:
      clusters: A list of annotated aligned segments representing the consensus
        of each cluster.
    """

    if len(als) == 0:
        clusters = []

    elif len(als) == 1:
        clusters = [make_singleton_cluster(al) for al in als]

    else:
        seed = propose_seed(als, max_read_length)
        near_seed, remaining = within_radius_of_seed(
            seed, als, max_hq_mismatches
        )

        if len(near_seed) == 0:
            # didn't make progress, so give up
            clusters = [make_singleton_cluster(al) for al in als]

        else:
            clusters = [
                call_consensus(near_seed, max_read_length)
            ] + form_clusters(remaining, max_read_length, max_hq_mismatches)

    return clusters


def align_clusters(
    first: pysam.AlignedSegment, second: pysam.AlignedSegment
) -> (int, int):
    """Calculates the number of indel mismatches and high quality mismatches
      between the sequences of 2 aligned segments (reads).

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
        if (first.query_qualities[q_i] > 20) and (
            second.query_qualities[t_i] > 20
        ):
            num_hq_mismatches += 1

    indels = al["XO"]
    return indels, num_hq_mismatches


def within_radius_of_seed(
    seed: str, als: List[pysam.AlignedSegment], max_hq_mismatches: int
) -> (List[pysam.AlignedSegment], List[pysam.AlignedSegment]):
    """Returns the aligned segments (reads) in a list with sequences that are
    close enough to specified seed

    Finds the aligned segments in the input list that have fewer than a specified
    threshold of high quality mismatches between their sequence and the seed
    sequence (Hamming Distance).

    Args:
      seed: A specified sequence.
      als: A list of aligned segments.
      max_hq_mismatches: A threshold specifying the maximum number of high quality
        mismatches between the seqeunces of 2 aligned segments to be collapsed.

    Returns:
      near_seed: List of aligned segments with sequences that are within
        radius of seed.
      remaining: List of aligned segments with sequences that are not within
        radius of seed.
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


def propose_seed(als: List[pysam.AlignedSegment], max_read_length: int) -> str:
    """Proposes a 'seed' around which a cluster is formed.

    Generates a seed around which clusters of aligned segments (reads) are formed
    in within_radius_of_seed. The method of clustering is defined in the
    documentation of that function. The seed is defined as the most frequent
    sequence in the list of aligned segments, or the consensus sequence of the
    aligned segments if there is no most frequent sequence.

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

    In the forming of clusters, if a cluster only contains one aligned segment
    (read), then an annotated aligned segment is generated for this singleton
    aligned segment.

    Args:
      al: The single aligned segment.

    Returns:
      singleton: A single annotated aligned segment representing the singleton
        cluster.
    """
    singleton = pysam.AlignedSegment()
    singleton.query_sequence = al.query_sequence
    singleton.query_qualities = al.query_qualities
    singleton.set_tag(NUM_READS_TAG, 1, "i")
    return singleton


def call_consensus(
    als: List[pysam.AlignedSegment], max_read_length: int
) -> pysam.AlignedSegment:
    """Generates a consensus annotated aligned segment for a list of aligned
    segments (reads).

    For a list of aligned segments, creates a consensus sequence by taking the
    most frequent high quality base for each index in the sequence. Then it
    constructs an annotated aligned segment recording this sequence, as well
    as the qualities and the total number of reads. The qualities of each base
    in the final sequence is recorded as high only if a majority of the original
    aligned segments had had that base at that index and at least one of them was
    high quality. If there is a tie in the frequency, then a placeholder base is
    placed in the final sequence indicating a consensus could not be reached for
    that base.

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
    consensus.query_sequence = "".join(
        utilities.base_order[i] for i in best_idxs
    )
    consensus.query_qualities = array.array("B", qs)
    consensus.set_tag(NUM_READS_TAG, len(als), "i")

    return consensus


def merge_annotated_clusters(
    biggest: pysam.AlignedSegment, other: pysam.AlignedSegment
) -> pysam.AlignedSegment:
    """Merges 2 annotated clusters together.

    Merges 2 annotated aligned segments, each representing a cluster. Merges the
    smaller into the larger. Adds the read number of the 2nd cluster to the first.

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


def correct_umis_in_group(
    cell_group: pd.DataFrame, sampleID: str, max_UMI_distance: int = 2, verbose = False
) -> pd.DataFrame:
    """
    Given a group of alignments, collapses UMIs that have close sequences.

    Given a group of alignments (that share a cellBC and intBC if from
    errorCorrectUMIs), determines which UMIs are to be merged into which.
    For a given UMI, merges it into the UMI with the highest read count
    that has a Hamming Distance <= max_UMI_distance. For a merge, removes the
    less abundant UMI and adds its read count to the more abundant UMI.
    If a UMI is to be merged into a UMI that itself will be merged, the
    correction is propogated through.

    TODO: We have noticed that UMIs with ties in their read counts are
    corrected and merged rather arbitrarily. We are looking into this.

    Args:
        input_df: Input DataFrame of alignments.
        _id: Identification of sample.
        max_UMI_distance: Maximum Hamming distance between UMIs
            for error correction.

    Returns:
        A DataFrame of error corrected UMIs within the grouping.

    """

    UMIs = list(cell_group["UMI"])

    ds = hamming_distance_matrix(UMIs)

    corrections = register_corrections(ds, max_UMI_distance, UMIs)

    num_corrections = 0
    corrected_group = pd.DataFrame()
    corrected_names = []

    if len(corrections) == 0:
        return cell_group, 0, cell_group.shape[0], ""

    for _, al in cell_group.iterrows():
        al_umi = al["UMI"]
        for _, al2 in cell_group.iterrows():
            al2_umi = al2["UMI"]
            # Keys are 'from' and values are 'to', so correct al2 to al
            if al2_umi in corrections.keys() and corrections[al2_umi] == al_umi:

                bad_nr = al2["ReadCount"]
                prev_nr = al["ReadCount"]
                al["ReadCount"] = bad_nr + prev_nr

                if verbose:
                    logging.info(
                        f"{bad_nr} reads merged from {al2_umi} to {al_umi}"
                        + f"for a total of {bad_nr + prev_nr} reads."
                    )

                # update alignment if already seen
                if al["UMI"] in corrected_names:
                    corrected_group = corrected_group[
                        corrected_group["UMI"] != al_umi
                    ]
                corrected_group = corrected_group.append(al)

                num_corrections += 1
                corrected_names.append(al_umi)
                corrected_names.append(al2_umi)

    for _, al in cell_group.iterrows():

        # Add alignments not touched during error correction back into the group
        # to be written to file.
        if al["UMI"] not in corrected_names:
            corrected_group = corrected_group.append(al)

    total = len(cell_group)

    return corrected_group, num_corrections, total
