"""
This file stores useful functions for dealing with alignments.
Invoked through pipeline.py and supports the align_sequences function.
"""
import re
from typing import Dict, List, Tuple

import ngs_tools as ngs
from pyseq_align import NeedlemanWunsch, SmithWaterman

from cassiopeia.mixins import PreprocessError, UnknownCigarStringError


def align_local(
    ref: str,
    seq: str,
    substitution_matrix: Dict[str, Dict[str, int]],
    gap_open_penalty: int,
    gap_extend_penalty: int,
) -> Tuple[str, int, int, float, str]:
    """Perform local alignment of `seq` to `ref` using Smith-Waterman.

    Todo:
        Deprecate dependency on skbio.

    Args:
        ref: The reference sequence.
        seq: The query sequence.
        substitution_matrix: Nested dictionary encoding the substitution matrix.
        gap_open_penalty: Gap open penalty.
        gap_extend_penalty: Gap extend penalty.

    Returns:
        A tuple containing the CIGAR string, query sequence start position,
            reference sequence start position, alignment score, and query
            sequence

    Raises:
        PreprocessError if skbio could not be imported.
    """
    aligner = SmithWaterman(
        substitution_matrix=substitution_matrix,
        gap_open=-gap_open_penalty + 1,
        gap_extend=-gap_extend_penalty,
    )
    aln = aligner.align(ref, seq, n=1)[0]
    return (
        ngs.sequence.alignment_to_cigar(aln.result_a, aln.result_b),
        aln.pos_b,
        aln.pos_a,
        aln.score,
        seq,
    )


def align_global(
    ref: str,
    seq: str,
    substitution_matrix: Dict[str, Dict[str, int]],
    gap_open_penalty: int,
    gap_extend_penalty: int,
) -> Tuple[str, int, int, float, str]:
    """Perform global alignment of `seq` to `ref` using Needleman-Wunsch.

    Args:
        ref: The reference sequence.
        seq: The query sequence.
        substitution_matrix: Nested dictionary encoding the substitution matrix.
        gap_open_penalty: Gap open penalty.
        gap_extend_penalty: Gap extend penalty.

    Returns:
        A tuple containing the CIGAR string, query sequence start position,
            reference sequence start position, alignment score, and query
            sequence
    """
    aligner = NeedlemanWunsch(
        substitution_matrix=substitution_matrix,
        gap_open=-gap_open_penalty
        + 1,  # Slight difference in score calculation
        gap_extend=-gap_extend_penalty,
        no_end_gap_penalty=True,  # Reads are expected to be shorter
    )
    aln = aligner.align(ref, seq)
    return (
        ngs.sequence.alignment_to_cigar(aln.result_a, aln.result_b),
        aln.pos_b,
        aln.pos_a,
        aln.score,
        seq,
    )


def parse_cigar(
    cigar: str,
    seq: str,
    ref: str,
    ref_start: int,
    query_start: int,
    barcode_interval: Tuple[int, int],
    cutsites: List[int],
    cutsite_window: int = 0,
    context: bool = True,
    context_size: int = 5,
) -> Tuple[str, List[str]]:
    """Parse the cigar string from a TargetSite alignment.

    Parse the cigar string from an alignment of a TargetSite read into the
    indels observed at each of the cutsites. We assume that the construct
    has at least an integration barcode marking reads from the same Target Site.
    It's also likely that there is at least one cut site per Target Site, whose
    location can be specified with the cutsites list, though if this is an
    empty list, no cut sites will be processed. After aligning, the intBC and a
    list storing the indels at each cutsite is returned.

    Args:
        cigar: CIGAR string from the alignment
        seq: Query sequence
        ref: Reference sequence
        ref_start: Position that alignment starts in the reference string
        query_start: Position that alignment starts in the query string
        barcode_interval: Interval in reference sequence that stores the
            integration barcode
        cutsites: A list of cutsite locations in the reference
        cutsite_window: Number of nucleotides to the left and right of the
            cutsite to look for the beginning of an indel
        context: Include nucleotide sequence around indels
        context_size: Number of bases to report for context

    Returns:
        The intBC and a list storing the indel observed at each cutsite.
    """

    cutsite_lims = [
        (site - cutsite_window, site + cutsite_window) for site in cutsites
    ]
    indels = ["" for site in cutsites]
    uncut_indels = ["" for site in cutsites]
    intBC = "NC"

    ref_anchor = ref[barcode_interval[0] - 11 : barcode_interval[0]]
    intBC_length = barcode_interval[1] - barcode_interval[0]

    # set up two pointers to the reference and query, respectively
    ref_pointer = ref_start
    query_pointer = query_start
    query_pad = 0

    cigar_chunks = re.findall(r"(\d+)?([A-Za-z])?", cigar)

    for chunk in cigar_chunks:

        if chunk[1] == "":
            continue

        length, _type = int(chunk[0]), chunk[1]

        if _type == "M":
            pos_start, pos_end = ref_pointer, (ref_pointer + length)

            # check if match goes into barcode interval.
            # this misses instances where the entire barcode isn't matched, as
            # in when there is a 1bp deletion in the intBC region.
            if (
                barcode_interval[0] >= pos_start
                and barcode_interval[1] <= pos_end
            ):
                intBC_offset = barcode_interval[0] - pos_start
                # if the entire intBC is deleted, we report this as not-captured
                if query_pad > (-1 * intBC_length):
                    intbc_end_index = (
                        query_pointer + intBC_offset + intBC_length + query_pad
                    )
                    intBC = seq[
                        (query_pointer + intBC_offset) : intbc_end_index
                    ]

            # check if a match occurs in any of the cutsite windows
            for i, site in zip(range(len(cutsites)), cutsites):

                if site >= pos_start and site <= pos_end and indels[i] == "":
                    dist = site - pos_start
                    loc = query_pointer + dist

                    if context:
                        context_l = seq[(loc - context_size) : loc]
                        context_r = seq[loc : (loc + context_size)]

                        uncut_indels[i] = f"{context_l}[None]{context_r}"
                    else:
                        uncut_indels[i] = "None"

            # account for partial deletions of the intBC
            if (
                pos_start >= barcode_interval[0]
                and pos_start <= barcode_interval[1]
                and pos_end > barcode_interval[1]
            ):
                new_intbc_len = barcode_interval[1] - pos_start + 1
                intbc_end_index = query_pointer + new_intbc_len
                intBC = seq[query_pointer:intbc_end_index]

            if (
                pos_end >= barcode_interval[0]
                and pos_end <= barcode_interval[1]
                and pos_start < barcode_interval[0]
            ):
                new_intbc_len = pos_end - barcode_interval[0] + 1
                intBC_offset = barcode_interval[0] - pos_start
                intbc_end_index = query_pointer + intBC_offset + new_intbc_len
                intBC = seq[(query_pointer + intBC_offset) : intbc_end_index]

            ref_pointer += length
            query_pointer += length

        elif _type == "I":

            if ref_pointer == barcode_interval[0]:
                query_pad = length

            # increment the query string pointer since we're working with an
            # insertion (i.e. characters that appear in the query but not the
            # reference)
            pos_start = query_pointer
            query_pointer += length

            for i, window in zip(range(len(cutsites)), cutsite_lims):

                if ref_pointer >= window[0] and ref_pointer <= window[1]:

                    if context:
                        context_l = seq[(pos_start - context_size) : pos_start]
                        context_r = seq[
                            pos_start : (pos_start + context_size + length)
                        ]

                        # when referencing the actual string, we say convert
                        # to 1-indexing for easier comparison
                        indels[
                            i
                        ] += (
                            f"{context_l}[{ref_pointer+1}:{length}I]{context_r}"
                        )
                    else:

                        # when referencing the actual string, we say convert
                        # to 1-indexing for easier comparison
                        indels[i] += f"{ref_pointer+1}:{length}I"

        elif _type == "D":

            # increment the reference string pointer since we're working with a
            # deletion (i.e. character that appear in the reference but not the
            # query string)
            pos_start = ref_pointer
            ref_pointer += length

            if ref_pointer == barcode_interval[0]:
                query_pad = -1 * length

            for i, window in zip(range(len(cutsites)), cutsite_lims):

                if (
                    (window[0] <= ref_pointer and ref_pointer <= window[1])
                    or (window[0] <= pos_start and pos_start <= window[1])
                    or (pos_start <= window[0] and ref_pointer >= window[1])
                    or (pos_start >= window[0] and ref_pointer <= window[1])
                ):

                    if context:
                        context_l = seq[
                            (query_pointer - context_size) : query_pointer
                        ]
                        context_r = seq[
                            query_pointer : (query_pointer + context_size)
                        ]

                        # when referencing the actual string, we say convert
                        # to 1-indexing for easier comparison
                        indels[
                            i
                        ] += f"{context_l}[{pos_start+1}:{length}D]{context_r}"
                    else:
                        # when referencing the actual string, we say convert
                        # to 1-indexing for easier comparison
                        indels[i] += f"{pos_start+1}:{length}D"

        elif _type == "H":
            # Hard clip! Do nothing.

            query_pointer += length

        else:
            raise UnknownCigarStringError(
                f"Encountered unknown cigar string: {chunk}"
            )

    if intBC == "NC" or len(intBC) < intBC_length:

        anchor = seq[(barcode_interval[0] - 11) : barcode_interval[0]]
        if anchor == ref_anchor:
            intBC = seq[barcode_interval[0] : barcode_interval[1]]

    for i, indel in zip(range(len(indels)), indels):

        if indel == "":
            indels[i] = uncut_indels[i]

    return intBC, indels
