"""
This file contains all high-level functionality for preprocessing sequencing
data into character matrices ready for phylogenetic inference. This file
is mainly invoked by cassiopeia_preprocess.py.
"""
import warnings

from functools import partial
import os
from pathlib import Path
import time
from typing import List, Optional, Tuple, Union

from Bio import SeqIO
from joblib import delayed
import matplotlib.pyplot as plt
import ngs_tools as ngs
import numpy as np
import pandas as pd
import pysam
from tqdm.auto import tqdm
from typing_extensions import Literal

from cassiopeia.mixins import logger, PreprocessError
from cassiopeia.mixins.warnings import PreprocessWarning
from cassiopeia.preprocess import (
    alignment_utilities,
    constants,
    map_utils as m_utils,
    doublet_utils as d_utils,
    lineage_utils as l_utils,
    UMI_utils,
    utilities,
)


DNA_SUBSTITUTION_MATRIX = constants.DNA_SUBSTITUTION_MATRIX
BAM_CONSTANTS = constants.BAM_CONSTANTS
progress = tqdm


@logger.namespaced("convert")
@utilities.log_kwargs
@utilities.log_runtime
def convert_fastqs_to_unmapped_bam(
    fastq_fps: List[str],
    chemistry: Literal["dropseq", "10xv2", "10xv3", "indropsv3", "slideseq2"],
    output_directory: str,
    name: Optional[str] = None,
    n_threads: int = 1,
) -> str:
    """Converts FASTQs into an unmapped BAM based on a chemistry.

    This function converts a set of FASTQs into an unmapped BAM with appropriate
    BAM tags.

    Args:
        fastq_fps: List of paths to FASTQ files. Usually, this argument contains
            two FASTQs, where the first contains the barcode and UMI sequences
            and the second contains cDNA. The FASTQs may be gzipped.
        chemistry: Sample-prep/sequencing chemistry used. The following
            chemistries are supported:
            * dropseq: Droplet-based scRNA-seq chemistry described in
                Macosco et al. 2015
            * 10xv2: 10x Genomics 3' version 2
            * 10xv3: 10x Genomics 3' version 3
            * indropsv3: inDrops version 3 by Zilionis et al. 2017
            * slideseq2: Slide-seq version 2
        output_directory: The output directory where the unmapped BAM will be
            written to. This directory must exist prior to calling this function.
        name: Name of the reads in the FASTQs. This name is set as the read group
            name for the reads in the output BAM, as well as the filename prefix
            of the output BAM. If not provided, a short random UUID is used as
            the read group name, but not as the filename prefix of the BAM.
        n_threads: Number of threads to use. Defaults to 1.

    Returns:
        Path to written BAM

    Raises:
        PreprocessError if the provided chemistry does not exist.
    """
    if chemistry not in constants.CHEMISTRY_BAM_TAGS:
        raise PreprocessError(f"Unknown chemistry {chemistry}")
    tag_map = constants.CHEMISTRY_BAM_TAGS[chemistry]
    chemistry = ngs.chemistry.get_chemistry(chemistry)

    logger.info(f"Using {chemistry} chemistry.")
    bam_fp = os.path.join(
        output_directory, f"{name}_unmapped.bam" if name else "unmapped.bam"
    )
    ngs.fastq.fastqs_to_bam_with_chemistry(
        fastq_fps,
        chemistry,
        tag_map,
        bam_fp,
        name=name,
        show_progress=True,
        n_threads=n_threads,
    )
    return bam_fp


@logger.namespaced("filter_bam")
@utilities.log_kwargs
@utilities.log_runtime
def filter_bam(
    bam_fp: str,
    output_directory: str,
    quality_threshold: int = 10,
    n_threads: int = 1,
) -> str:
    """Filter reads in a BAM that have low quality barcode or UMIs.

    Args:
        bam_fp: Input BAM filepath containing reads to filter.
        output_directory: The output directory where the filtered BAM will be
            written to. This directory must exist prior to calling this function.
        quality_threshold: Every base of the barcode and UMI sequence for a
            given read must have at least this PHRED quality score for it to
            pass the filtering.
        n_threads: Number of threads to use. Defaults to 1.

    Returns:
        Path to filtered BAM
    """
    n_filtered = 0

    def filter_func(aln):
        # False means this read will be filtered out
        filter_bool = all(
            q >= quality_threshold
            for q in pysam.qualitystring_to_array(
                aln.get_tag(BAM_CONSTANTS["RAW_CELL_BC_QUALITY_TAG"])
            )
        ) and all(
            q >= quality_threshold
            for q in pysam.qualitystring_to_array(
                aln.get_tag(BAM_CONSTANTS["UMI_QUALITY_TAG"])
            )
        )
        nonlocal n_filtered
        n_filtered += not filter_bool
        return filter_bool

    prefix, ext = os.path.splitext(os.path.basename(bam_fp))
    filtered_fp = os.path.join(output_directory, f"{prefix}_filtered{ext}")
    ngs.bam.filter_bam(
        bam_fp,
        filter_func,
        filtered_fp,
        show_progress=True,
        n_threads=n_threads,
    )
    logger.info(f"Filtered {n_filtered} reads that didn't pass the filter.")
    return filtered_fp


@logger.namespaced("error_correct_cellbcs_to_whitelist")
@utilities.log_kwargs
@utilities.log_runtime
def error_correct_cellbcs_to_whitelist(
    bam_fp: str,
    whitelist: Union[str, List[str]],
    output_directory: str,
    n_threads: int = 1,
) -> str:
    """Error-correct cell barcodes in the input BAM.

    The barcode correction procedure used in Cell Ranger by 10X Genomics is used.
    https://kb.10xgenomics.com/hc/en-us/articles/115003822406-How-does-Cell-Ranger-correct-barcode-sequencing-errors
    This function can either take a list of whitelisted barcodes or a plaintext
    file containing these barcodes.

    Args:
        bam_fp: Input BAM filepath containing raw barcodes
        whitelist: May be either a single path to a plaintext file containing
            the barcode whitelist, one barcode per line, or a list of
            barcodes.
        output_directory: The output directory where the corrected BAM will be
            written to. This directory must exist prior to calling this function.
        n_threads: Number of threads to use. Defaults to 1.

    Todo:
        Currently, the user must provide their own whitelist, and Cassiopeia
        does not use any of the whitelists provided by the ngs-tools library.
        At some point, we should update the pipeline so that if ngs-tools
        provides a pre-packaged whitelists, it uses that for those chemistries.

    Returns:
        Path to corrected BAM
    """
    if isinstance(whitelist, list):
        whitelist_set = set(whitelist)
    else:
        with open(whitelist, "r") as f:
            whitelist_set = set(
                line.strip() for line in f if not line.isspace()
            )
    whitelist = list(whitelist_set)

    # Extract all raw barcodes and their qualities
    barcodes = []
    qualities = []
    with pysam.AlignmentFile(
        bam_fp, "rb", check_sq=False, threads=n_threads
    ) as f:
        for read in f:
            barcodes.append(read.get_tag(BAM_CONSTANTS["RAW_CELL_BC_TAG"]))
            qualities.append(
                read.get_tag(BAM_CONSTANTS["RAW_CELL_BC_QUALITY_TAG"])
            )
    logger.info(f"Detected {len(set(barcodes))} raw barcodes.")

    # Correct
    corrections = ngs.sequence.correct_sequences_to_whitelist(
        barcodes, qualities, whitelist, show_progress=True, n_threads=n_threads
    )

    # Write corrected BAM
    prefix, ext = os.path.splitext(os.path.basename(bam_fp))
    corrected_fp = os.path.join(output_directory, f"{prefix}_corrected{ext}")
    with pysam.AlignmentFile(
        bam_fp, "rb", check_sq=False, threads=n_threads
    ) as f_in:
        with pysam.AlignmentFile(
            corrected_fp, "wb", template=f_in, threads=n_threads
        ) as f_out:
            for i, read in enumerate(f_in):
                if corrections[i]:
                    read.set_tag(BAM_CONSTANTS["CELL_BC_TAG"], corrections[i])
                f_out.write(read)
    return corrected_fp


@logger.namespaced("collapse")
@utilities.log_kwargs
@utilities.log_runtime
def collapse_umis(
    bam_fp: str,
    output_directory: str,
    max_hq_mismatches: int = 3,
    max_indels: int = 2,
    method: Literal["cutoff", "likelihood"] = "cutoff",
    n_threads: int = 1,
) -> pd.DataFrame:
    """Collapses close UMIs together from a bam file.

    On a basic level, it aggregates together identical or close reads to count
    how many times a UMI was read. Performs basic error correction, allowing
    UMIs to be collapsed together which differ by at most a certain number of
    high quality mismatches and indels in the sequence read itself. Writes out
    a dataframe of the collapsed UMIs table.

    Args:
        bam_file_name: File path of the bam_file. Just the bam file name can be
            specified if the bam already exists in the output directory
        output_directory: The output directory where the sorted bam directory, the
            collapsed bam directory, and the final collapsed table are written to
        max_hq_mismatches: A threshold specifying the max number of high quality
            mismatches between the seqeunces of 2 aligned segments to be collapsed
        max_indels: A threshold specifying the maximum number of differing indels
            allowed between the sequences of 2 aligned segments to be collapsed
        method: Which method to use to form initial sequence clusters. Must be
            one of the following:
            * cutoff: Uses a quality score hard cutoff of 30, and any mismatches
                below this quality are ignored. Initial sequence clusters are
                formed by selecting the most common base at each position (with
                quality at least 30).
            * likelihood: Utilizes the error probability encoded in the quality
                score. Initial sequence clusters are formed by selecting the
                most probable at each position.
        n_threads: Number of threads to use.

    Returns:
        A DataFrame of collapsed reads.
    """
    # pathing written such that the bam file that is being converted does not
    # have to exist currently in the output directory
    if output_directory[-1] == "/":
        output_directory = output_directory[:-1]
    sorted_file_name = Path(
        output_directory
        + "/"
        + ".".join(bam_fp.split("/")[-1].split(".")[:-1])
        + "_sorted.bam"
    )

    cell_bc_tag = UMI_utils.detect_cell_bc_tag(bam_fp)
    logger.info(f"Using BAM tag `{cell_bc_tag}` as cell barcodes")

    max_read_length, total_reads_out = UMI_utils.sort_bam(
        bam_fp,
        str(sorted_file_name),
        sort_key=lambda al: (
            al.get_tag(cell_bc_tag),
            al.get_tag(BAM_CONSTANTS["UMI_TAG"]),
        ),
        filter_func=lambda al: al.has_tag(cell_bc_tag),
    )
    logger.info("Sorted bam directory saved to " + str(sorted_file_name))
    logger.info("Max read length of " + str(max_read_length))
    logger.info("Total reads: " + str(total_reads_out))

    collapsed_file_name = sorted_file_name.with_suffix(".collapsed.bam")
    UMI_utils.form_collapsed_clusters(
        str(sorted_file_name),
        str(collapsed_file_name),
        max_hq_mismatches,
        max_indels,
        cell_key=lambda al: al.get_tag(cell_bc_tag),
        method=method,
        n_threads=n_threads,
    )

    collapsed_df_file_name = sorted_file_name.with_suffix(".collapsed.txt")

    df = utilities.convert_bam_to_df(
        str(collapsed_file_name), str(collapsed_df_file_name), create_pd=True
    )
    logger.info("Collapsed bam directory saved to " + str(collapsed_file_name))
    logger.info("Converted dataframe saved to " + str(collapsed_df_file_name))
    return df


@logger.namespaced("resolve")
@utilities.log_kwargs
@utilities.log_runtime
def resolve_umi_sequence(
    molecule_table: pd.DataFrame,
    output_directory: str,
    min_umi_per_cell: int = 10,
    min_avg_reads_per_umi: float = 2.0,
    plot: bool = True,
) -> pd.DataFrame:
    """Resolve a consensus sequence for each UMI.

    This procedure will perform UMI and cellBC filtering on the basis of reads
    per UMI and UMIs per cell and then assign the most abundant sequence to
    each UMI if there is a set of conflicting sequences per UMI.

    Args:
        molecule_table: molecule table to resolve
        output_directory: Directory to store results
        min_umi_per_cell: The threshold specifying the minimum number of UMIs
            in a cell needed to be retained during filtering
        min_avg_reads_per_umi: The threshold specifying the minimum coverage
            (i.e. average) reads per UMI in a cell needed for that cell to be
            retained during filtering

    Return:
        A molecule table with unique mappings between cellBC-UMI pairs.
    """
    if plot:
        # -------------------- Plot # of sequences per UMI -------------------- #
        equivClass_group = (
            molecule_table.groupby(["cellBC", "UMI"])
            .agg({"grpFlag": "count"})
            .sort_values("grpFlag", ascending=False)
            .reset_index()
        )

        _ = plt.figure(figsize=(8, 5))
        plt.hist(
            equivClass_group["grpFlag"],
            bins=range(1, equivClass_group["grpFlag"].max()),
        )
        plt.title("Unique Seqs per cellBC+UMI")
        plt.yscale("log", basey=10)
        plt.xlabel("Number of Unique Seqs")
        plt.ylabel("Count (Log)")
        plt.savefig(os.path.join(output_directory, "seqs_per_equivClass.png"))
        plt.close()

    # ----------------- Select most abundant sequence ------------------ #

    mt_filter = {}
    total_numReads = {}
    top_reads = {}
    second_reads = {}
    first_reads = {}

    unique_pairs = molecule_table.groupby(["cellBC", "UMI"], sort=False)

    for _, group in tqdm(
        unique_pairs,
        total=len(unique_pairs.size()),
        desc="Resolving UMI sequences",
    ):

        # base case - only one sequence
        if group.shape[0] == 1:
            good_readName = group["readName"].iloc[0]
            mt_filter[good_readName] = False
            total_numReads[good_readName] = group["readCount"]
            top_reads[good_readName] = group["readCount"]

        # more commonly - many sequences for a given UMI
        else:
            group_sort = group.sort_values(
                "readCount", ascending=False, ignore_index=True
            )
            good_readName = group_sort["readName"].iloc[0]

            # keep the first entry (highest readCount)
            mt_filter[good_readName] = False

            total_numReads[good_readName] = group_sort["readCount"].sum()
            top_reads[good_readName] = group_sort["readCount"].iloc[0]
            second_reads[good_readName] = group_sort["readCount"].iloc[1]
            first_reads[good_readName] = group_sort["readCount"].iloc[0]

            # mark remaining UMIs for filtering
            for i in range(1, group.shape[0]):
                bad_readName = group_sort["readName"].iloc[i]
                mt_filter[bad_readName] = True

    # apply the filter using the hash table created above
    molecule_table["filter"] = molecule_table["readName"].map(mt_filter)
    n_filtered = molecule_table[molecule_table["filter"]].shape[0]

    logger.info(f"Filtered out {n_filtered} reads.")

    # filter based on status & reindex
    filt_molecule_table = molecule_table[~molecule_table["filter"]].copy()
    filt_molecule_table.drop(columns=["filter"], inplace=True)

    if plot:
        # ---------------- Plot Diagnositics after Resolving ---------------- #
        h = plt.figure(figsize=(14, 10))
        plt.plot(list(top_reads.values()), list(total_numReads.values()), "r.")
        plt.ylabel("Total Reads")
        plt.xlabel("Number Reads for Picked Sequence")
        plt.title("Total vs. Top Reads for Picked Sequence")
        plt.savefig(
            os.path.join(output_directory, "total_vs_top_reads_pickSeq.png")
        )
        plt.close()

        h = plt.figure(figsize=(14, 10))
        plt.plot(list(first_reads.values()), list(second_reads.values()), "r.")
        plt.ylabel("Number Reads for Second Best Sequence")
        plt.xlabel("Number Reads for Picked Sequence")
        plt.title("Second Best vs. Top Reads for Picked Sequence")
        plt.savefig(
            os.path.join(output_directory, "second_vs_top_reads_pickSeq.png")
        )
        plt.close()

    filt_molecule_table = utilities.filter_cells(
        filt_molecule_table,
        min_umi_per_cell=min_umi_per_cell,
        min_avg_reads_per_umi=min_avg_reads_per_umi,
    )
    return filt_molecule_table


@logger.namespaced("align")
@utilities.log_kwargs
@utilities.log_runtime
def align_sequences(
    queries: pd.DataFrame,
    ref_filepath: Optional[str] = None,
    ref: Optional[str] = None,
    gap_open_penalty: float = 20,
    gap_extend_penalty: float = 1,
    n_threads: int = 1,
) -> pd.DataFrame:
    """Align reads to the TargetSite reference.

    Take in several queries stored in a DataFrame mapping cellBC-UMIs to a
    sequence of interest and align each to a reference sequence. The alignment
    algorithm used is the Smith-Waterman local alignment algorithm. The desired
    output consists of the best alignment score and the CIGAR string storing the
    indel locations in the query sequence.

    Args:
        queries: DataFrame storing a list of sequences to align.
        ref_filepath: Filepath to the reference FASTA.
        ref: Reference sequence.
        gapopen: Gap open penalty
        gapextend: Gap extension penalty
        n_threads: Number of threads to use.

    Returns:
        A DataFrame mapping each sequence name to the CIGAR string, quality,
        and original query sequence.
    """
    try:
        from skbio import alignment
    except ModuleNotFoundError:
        raise PreprocessError(
            "Scikit-bio is not installed. Try pip-installing "
            " first and then re-running this function."
        )

    if (ref is None) == (ref_filepath is None):
        raise PreprocessError(
            "Either `ref_filepath` or `ref` must be provided."
        )

    alignment_dictionary = {}

    if ref_filepath:
        ref = str(list(SeqIO.parse(ref_filepath, "fasta"))[0].seq)

    # Helper function for paralleleization
    def align(seq):
        aligner = alignment.StripedSmithWaterman(
            seq,
            substitution_matrix=DNA_SUBSTITUTION_MATRIX,
            gap_open_penalty=gap_open_penalty,
            gap_extend_penalty=gap_extend_penalty,
        )
        aln = aligner(ref)
        return (
            aln.cigar,
            aln.query_begin,
            aln.target_begin,
            aln.optimal_alignment_score,
            aln.query_sequence,
        )

    for umi, aln in zip(
        queries.index,
        ngs.utils.ParallelWithProgress(
            n_jobs=n_threads,
            total=queries.shape[0],
            desc="Aligning sequences to reference",
        )(delayed(align)(queries.loc[umi].seq) for umi in queries.index),
    ):
        query = queries.loc[umi]
        alignment_dictionary[query.readName] = (
            query.cellBC,
            query.UMI,
            query.readCount,
            *aln,
        )

    final_time = time.time()
    alignment_df = pd.DataFrame.from_dict(alignment_dictionary, orient="index")
    alignment_df.columns = [
        "cellBC",
        "UMI",
        "readCount",
        "CIGAR",
        "QueryBegin",
        "ReferenceBegin",
        "AlignmentScore",
        "Seq",
    ]

    alignment_df.index.name = "readName"
    alignment_df.reset_index(inplace=True)
    return alignment_df


@logger.namespaced("call_alleles")
@utilities.log_kwargs
@utilities.log_runtime
def call_alleles(
    alignments: pd.DataFrame,
    ref_filepath: Optional[str] = None,
    ref: Optional[str] = None,
    barcode_interval: Tuple[int, int] = (20, 34),
    cutsite_locations: List[int] = [112, 166, 220],
    cutsite_width: int = 12,
    context: bool = True,
    context_size: int = 5,
) -> pd.DataFrame:
    """Call indels from CIGAR strings.

    Given many alignments, we extract the indels by comparing the CIGAR strings
    of each alignment to the reference sequence.

    Args:
        alignments: Alignments provided in DataFrame
        ref_filepath: Filepath to the reference sequence
        ref: Nucleotide sequence of the reference
        barcode_interval: Interval in reference corresponding to the integration
            barcode
        cutsite_locations: A list of all cutsite positions in the reference
        cutsite_width: Number of nucleotides left and right of cutsite location
            that indels can appear in.
        context: Include sequence context around indels
        context_size: Number of bases to the right and left to include as
            context

    Returns:
        A DataFrame mapping each sequence alignment to the called indels.
    """
    if (ref is None) == (ref_filepath is None):
        raise PreprocessError(
            "Either `ref_filepath` or `ref` must be provided."
        )

    alignment_to_indel = {}
    alignment_to_intBC = {}

    if ref_filepath:
        ref = str(list(SeqIO.parse(ref_filepath, "fasta"))[0].seq)

    for _, row in tqdm(
        alignments.iterrows(),
        total=alignments.shape[0],
        desc="Parsing CIGAR strings into indels",
    ):

        intBC, indels = alignment_utilities.parse_cigar(
            row.CIGAR,
            row.Seq,
            ref,
            row.ReferenceBegin,
            row.QueryBegin,
            barcode_interval,
            cutsite_locations,
            cutsite_width,
            context=context,
            context_size=context_size,
        )

        alignment_to_indel[row.readName] = indels
        alignment_to_intBC[row.readName] = intBC

    indel_df = pd.DataFrame.from_dict(
        alignment_to_indel,
        orient="index",
        columns=[f"r{i}" for i in range(1, len(cutsite_locations) + 1)],
    )

    indel_df["allele"] = indel_df.apply(
        lambda x: "".join([str(i) for i in x.values]), axis=1
    )
    indel_df["intBC"] = indel_df.index.map(alignment_to_intBC)

    alignments.set_index("readName", inplace=True)

    alignments = alignments.join(indel_df)

    alignments.reset_index(inplace=True)

    # check cut-sites and raise a warning if any missing data is detected
    cutsites = utilities.get_default_cut_site_columns(alignments)
    if np.any((alignments[cutsites] == "").sum(axis=0) > 0):
        warnings.warn(
            "Detected missing data in alleles. You might"
            " consider re-running align_sequences with a"
            " lower gap-open penalty, or using a separate"
            " alignment strategy.",
            PreprocessWarning,
        )

    return alignments


@logger.namespaced("error_correct_intbcs_to_whitelist")
@utilities.log_kwargs
@utilities.log_runtime
def error_correct_intbcs_to_whitelist(
    input_df: pd.DataFrame,
    whitelist: Union[str, List[str]],
    intbc_dist_thresh: int = 1,
) -> pd.DataFrame:
    """Corrects all intBCs to the provided whitelist.

    This function can either take a list of whitelisted intBCs or a plaintext
    file containing these intBCs.

    Args:
        input_df: Input DataFrame of alignments.
        whitelist: May be either a single path to a plaintext file containing
            the barcode whitelist, one barcode per line, or a list of
            barcodes.
        intbc_dist_thresh: The threshold specifying the maximum Levenshtein
            distance between the read sequence and whitelist to be corrected.

    Returns:
        A DataFrame of error corrected intBCs.
    """
    if isinstance(whitelist, list):
        whitelist_set = set(whitelist)
    else:
        with open(whitelist, "r") as f:
            whitelist_set = set(
                line.strip() for line in f if not line.isspace()
            )
    whitelist = list(whitelist_set)
    unique_intbcs = list(input_df["intBC"].unique())
    corrections = {intbc: intbc for intbc in whitelist_set}
    logger.info(f"{len(unique_intbcs)} intBCs detected.")

    for intbc in progress(unique_intbcs, desc="Correcting intBCs to whitelist"):
        min_distance = np.inf
        min_wls = []
        if intbc not in whitelist_set:
            for wl_intbc in whitelist:
                distance = ngs.sequence.levenshtein_distance(intbc, wl_intbc)
                if distance < min_distance:
                    min_distance = distance
                    min_wls = [wl_intbc]
                elif distance == min_distance:
                    min_wls.append(wl_intbc)

        # Correct only if there is one matching whitelist. Discard if there
        # are multiple possible corrections.
        if len(min_wls) == 1 and min_distance <= intbc_dist_thresh:
            corrections[intbc] = min_wls[0]

    input_df["intBC"] = input_df["intBC"].map(corrections)

    return input_df[~input_df["intBC"].isna()]


@logger.namespaced("error_correct_umis")
@utilities.log_kwargs
@utilities.log_runtime
def error_correct_umis(
    input_df: pd.DataFrame,
    max_umi_distance: int = 2,
    allow_allele_conflicts: bool = False,
    n_threads: int = 1,
) -> pd.DataFrame:
    """Within cellBC-intBC pairs, collapses UMIs that have close sequences.

    Error correct UMIs together within cellBC-intBC pairs. UMIs that have a
    Hamming Distance between their sequences less than a threshold are
    corrected towards whichever UMI is more abundant. The `allow_allele_conflicts`
    option may be used to also group on the actual allele.

    Args:
        input_df: Input DataFrame of alignments.
        max_umi_distance: The threshold specifying the Maximum Hamming distance
            between UMIs for one to be corrected to another.
        allow_allele_conflicts: Whether or not to include the allele when
            splitting UMIs into allele groups. When True, UMIs are grouped by
            cellBC-intBC-allele triplets. When False, UMIs are grouped by
            cellBC-intBC pairs. This option is used when it is possible for
            each cellBC-intBC pair to have >1 allele state, such as for
            spatial data.
        n_threads: Number of threads to use.

    Returns:
        A DataFrame of error corrected UMIs.
    """
    if (
        len(
            [
                i
                for i in input_df.groupby(["cellBC", "intBC", "UMI"]).size()
                if i > 1
            ]
        )
        != 0
    ):
        raise PreprocessError(
            "Non-unique cellBC-UMI pair exists, please resolve UMIs."
        )

    sorted_df = input_df.sort_values(
        ["cellBC", "intBC", "readCount", "UMI"],
        ascending=[True, True, False, False],
    )

    if max_umi_distance == 0:
        logger.info(
            "Distance of 0, no correction occurred, all alignments returned"
        )
        return sorted_df

    num_corrected = 0
    total = 0

    alignment_df = pd.DataFrame()

    groupby = ["cellBC", "intBC"]
    if allow_allele_conflicts:
        groupby.append("allele")
    allele_groups = sorted_df.groupby(groupby)

    alignment_dfs = []
    for allele_group, num_corr, tot in ngs.utils.ParallelWithProgress(
        n_jobs=n_threads, total=len(allele_groups), desc="Error-correcting UMIs"
    )(
        delayed(UMI_utils.correct_umis_in_group)(allele_group, max_umi_distance)
        for _, allele_group in allele_groups
    ):
        num_corrected += num_corr
        total += tot

        alignment_dfs.append(allele_group)
    alignment_df = pd.concat(alignment_dfs, sort=True)
    logger.info(
        f"{num_corrected} UMIs Corrected of {total}"
        + f"({round(float(num_corrected) / total, 5) * 100}%)"
    )

    alignment_df["readName"] = alignment_df.apply(
        lambda x: "_".join([x.cellBC, x.UMI, str(int(x.readCount))]), axis=1
    )

    alignment_df.set_index("readName", inplace=True)
    alignment_df.reset_index(inplace=True)

    return alignment_df


@logger.namespaced("filter_molecule_table")
@utilities.log_kwargs
@utilities.log_runtime
def filter_molecule_table(
    input_df: pd.DataFrame,
    output_directory: str,
    min_umi_per_cell: int = 10,
    min_avg_reads_per_umi: float = 2.0,
    umi_read_thresh: int = -1,
    intbc_prop_thresh: float = 0.5,
    intbc_umi_thresh: int = 10,
    intbc_dist_thresh: int = 1,
    doublet_threshold: float = 0.35,
    allow_allele_conflicts: bool = False,
    plot: bool = False,
) -> pd.DataFrame:
    """Filters and corrects a molecule table of cellBC-UMI pairs.

    Performs the following steps on the alignments in a DataFrame:
        1. Filters out cellBCs with less than <= `min_umi_per_cell` unique UMIs
        2. Filters out UMIs with read count less than <= `umi_read_thresh`
        3. Error corrects intBCs by changing intBCs with low UMI counts to intBCs with the same allele and a close sequence
        4. Filters out cellBCs that contain too much conflicting allele information as intra-lineage doublets
        5. Chooses one allele for each cellBC-intBC pair, by selecting the most common

    Args:
        input_df: A molecule table, i.e. cellBC-UMI pairs. Note that
            each cellBC should only contain one instance of each UMI
        output_directory: The output directory path to store plots
        min_umi_per_cell: The threshold specifying the minimum number of UMIs
            in a cell needed to be retained during filtering
        min_avg_reads_per_umi: The threshold specifying the minimum coverage
            (i.e. average) reads per UMI in a cell needed in order for that
            cell to be retained during filtering
        umi_read_thresh: The threshold specifying the minimum read count needed
            for a UMI to be retained during filtering. Set dynamically if value
            is < 0
        intbc_prop_thresh: The threshold specifying the maximum proportion of
            the total UMI counts for a intBC to be corrected to another
        intbc_umi_thresh: The threshold specifying the maximum UMI count for
            an intBC needs to be corrected to another
        intbc_dist_thresh: The threshold specifying the maximum Levenshtein
            Distance between sequences for an intBC to be corrected to another
        doublet_threshold: The threshold specifying the maximum proportion of
            conflicting alleles information allowed to for an intBC to be
            retained in doublet filtering. Set to None to skip doublet filtering
        allow_allele_conflicts: Whether or not to allow multiple alleles to be
            assigned to each cellBC-intBC pair. For fully single-cell data,
            this option should be set to False, since each cell is expected to
            have a single allele state for each intBC. However, this option
            should be set to True for chemistries that may result in multiple
            physical cells being captured for each barcode.
        plot: Indicates whether to plot the change in intBC and cellBC counts
            across filtering stages

    Returns:
        A filtered and corrected allele table of cellBC-UMI-allele groups

    """
    input_df["status"] = "good"
    input_df.sort_values("readCount", ascending=False, inplace=True)
    rc_profile, upi_profile, upc_profile = {}, {}, {}

    logger.info("Logging initial stats...")
    if plot:
        (
            rc_profile["Init"],
            upi_profile["Init"],
            upc_profile["Init"],
        ) = utilities.record_stats(input_df)

    logger.info(
        f"Filtering out cell barcodes with fewer than {min_umi_per_cell} UMIs..."
    )
    filtered_df = utilities.filter_cells(
        input_df,
        min_umi_per_cell=min_umi_per_cell,
        min_avg_reads_per_umi=min_avg_reads_per_umi,
    )
    if plot:
        (
            rc_profile["CellFilter"],
            upi_profile["CellFilter"],
            upc_profile["CellFilter"],
        ) = utilities.record_stats(filtered_df)

    if umi_read_thresh < 0:
        R = filtered_df["readCount"]
        if list(R):
            umi_read_thresh = np.percentile(R, 99) // 10
        else:
            umi_read_thresh = 0
    logger.info(f"Filtering UMIs with read threshold {umi_read_thresh}...")
    filtered_df = utilities.filter_umis(
        filtered_df, readCountThresh=umi_read_thresh
    )

    if plot:
        (
            rc_profile["Filtered_UMI"],
            upi_profile["Filtered_UMI"],
            upc_profile["Filtered_UMI"],
        ) = utilities.record_stats(filtered_df)

    if intbc_dist_thresh > 0:
        logger.info("Error correcting intBCs...")
        filtered_df = utilities.error_correct_intbc(
            filtered_df,
            prop=intbc_prop_thresh,
            umiCountThresh=intbc_umi_thresh,
            bcDistThresh=intbc_dist_thresh,
        )

    if plot:
        (
            rc_profile["Process_intBC"],
            upi_profile["Process_intBC"],
            upc_profile["Process_intBC"],
        ) = utilities.record_stats(filtered_df)

    logger.info("Filtering cell barcodes one more time...")
    filtered_df = utilities.filter_cells(
        filtered_df,
        min_umi_per_cell=min_umi_per_cell,
        min_avg_reads_per_umi=min_avg_reads_per_umi,
    )

    if doublet_threshold and not allow_allele_conflicts:
        logger.info(
            f"Filtering out intra-lineage group doublets with proportion {doublet_threshold}..."
        )
        filtered_df = d_utils.filter_intra_doublets(
            filtered_df, prop=doublet_threshold
        )

    if not allow_allele_conflicts:
        logger.info("Mapping remaining intBC conflicts...")
        filtered_df = m_utils.map_intbcs(filtered_df)
    if plot:
        (
            rc_profile["Final"],
            upi_profile["Final"],
            upc_profile["Final"],
        ) = utilities.record_stats(filtered_df)

    # Count total filtered cellBCs
    cellBC_count = 0
    for name, grp in filtered_df.groupby(["cellBC"]):
        cellBC_count += 1

    if plot:
        stages = [
            "Init",
            "CellFilter",
            "Filtered_UMI",
            "Process_intBC",
            "Final",
        ]

        # Plot Read Per UMI Histogram
        h = plt.figure(figsize=(14, 10))
        for n in stages:
            ax = plt.hist(
                rc_profile[n], label=n, histtype="step", log=True, bins=200
            )
        plt.legend()
        plt.ylabel("Frequency")
        plt.xlabel("Number of Reads")
        plt.title("Reads Per UMI")
        plt.savefig(os.path.join(output_directory, "reads_per_umi.png"))
        plt.close()

        h = plt.figure(figsize=(14, 10))
        for n in stages:
            ax = plt.plot(upc_profile[n], label=n)
        plt.legend()
        plt.ylabel("Number of UMIs")
        plt.xlabel("Rank Order")
        plt.xscale("log", basex=10)
        plt.yscale("log", basey=10)
        plt.title("UMIs per CellBC")
        plt.savefig(os.path.join(output_directory, "umis_per_cellbc.png"))
        plt.close()

        h = plt.figure(figsize=(14, 10))
        for n in stages:
            ax = plt.hist(
                upi_profile[n], label=n, histtype="step", log=True, bins=200
            )
        plt.legend()
        plt.ylabel("Frequency")
        plt.xlabel("Number of UMIs")
        plt.title("UMIs per intBC")
        plt.savefig(os.path.join(output_directory, "umis_per_intbc.png"))
        plt.close()

    logger.info(
        f"Overall, filtered {cellBC_count} cells, with {filtered_df.shape[0]} UMIs."
    )

    filtered_df.set_index("readName", inplace=True)
    filtered_df.reset_index(inplace=True)

    return filtered_df


@logger.namespaced("call_lineages")
@utilities.log_kwargs
@utilities.log_runtime
def call_lineage_groups(
    input_df: pd.DataFrame,
    output_directory: str,
    min_umi_per_cell: int = 10,
    min_avg_reads_per_umi: float = 2.0,
    min_cluster_prop: float = 0.005,
    min_intbc_thresh: float = 0.05,
    inter_doublet_threshold: float = 0.35,
    kinship_thresh: float = 0.25,
    plot: bool = False,
) -> pd.DataFrame:
    """Assigns cells to their clonal populations.

    Performs multiple rounds of filtering and assigning to lineage groups:
        1. Iteratively generates putative lineage groups by forming intBC
        groups for each lineage group and then assigning cells based on how
        many intBCs they share with each intBC group (kinship).

        2. Refines these putative groups by removing non-informative intBCs
        and reassigning cells through kinship.

        3. Removes all inter-lineage doublets, defined as cells that have
        relatively equal kinship scores across multiple lineages and whose
        assignments are therefore ambigious.

        4. Finally, performs one more round of filtering non-informative intBCs
        and cellBCs with low UMI counts before returning a final table of
        lineage assignments, allele information, and read and umi counts for
        each sample.

    Args:
        input_df: The allele table of cellBC-UMI-allele groups to be annotated
            with lineage assignments
        output_directory: The folder to store the final table as well as plots
        min_umi_per_cell: The threshold specifying the minimum number of UMIs a
            cell needs in order to not be filtered during filtering
        min_avg_reads_per_umi: The threshold specifying the minimum coverage
            (i.e. average) reads per UMI in a cell needed in order for that
            cell not to be filtered during filtering
        min_cluster_prop: The minimum cluster size in the putative lineage
            assignment step, as a proportion of the number of cells
        min_intbc_thresh: The threshold specifying the minimum proportion of
            cells in a lineage group that need to have an intBC in order for it
            be retained during filtering. Also specifies the minimum proportion
            of cells that share an intBC with the most frequent intBC in
            forming putative lineage groups
        inter_doublet_threshold: The threshold specifying the minimum proportion
            of kinship a cell shares with its assigned lineage group out of all
            lineage groups for it to be retained during doublet filtering
        kinship_thresh: The threshold specifying the minimum proportion of
            intBCs shared between a cell and the intBC set of a lineage group
            needed to assign that cell to that lineage group in putative
            assignment
        plot: Indicates whether to generate plots

    Returns:
        None, saves output allele table to file.
    """
    logger.info(
        f"{input_df.shape[0]} UMIs (rows), with {input_df.shape[1]} attributes (columns)"
    )
    logger.info(str(len(input_df["cellBC"].unique())) + " Cells")

    # Create a pivot_table
    piv = pd.pivot_table(
        input_df, index="cellBC", columns="intBC", values="UMI", aggfunc="count"
    )
    piv = piv.div(piv.sum(axis=1), axis=0)

    # Reorder piv columns by binarized intBC frequency
    pivbin = piv.copy()
    pivbin[pivbin > 0] = 1
    intBC_sums = pivbin.sum(0)
    ordered_intBCs = intBC_sums.sort_values(ascending=False).index.tolist()
    piv = piv[ordered_intBCs]
    min_clust_size = int(min_cluster_prop * piv.shape[0])

    logger.info("Assigning initial lineage groups...")
    logger.info(f"Clustering with minimum cluster size {min_clust_size}...")
    piv_assigned = l_utils.assign_lineage_groups(
        piv,
        min_clust_size,
        min_intbc_thresh=min_intbc_thresh,
        kinship_thresh=kinship_thresh,
    )

    logger.info("Refining lineage groups...")
    logger.info(
        "Redefining lineage groups by removing low proportion intBCs..."
    )
    master_LGs, master_intBCs = l_utils.filter_intbcs_lg_sets(
        piv_assigned, min_intbc_thresh=min_intbc_thresh
    )

    logger.info("Reassigning cells to refined lineage groups by kinship...")
    kinship_scores = l_utils.score_lineage_kinships(
        piv_assigned, master_LGs, master_intBCs
    )

    logger.info("Annotating alignment table with refined lineage groups...")
    allele_table = l_utils.annotate_lineage_groups(
        input_df, kinship_scores, master_intBCs
    )
    if inter_doublet_threshold:
        logger.info(
            f"Filtering out inter-lineage group doublets with proportion {inter_doublet_threshold}..."
        )
        allele_table = d_utils.filter_inter_doublets(
            allele_table, rule=inter_doublet_threshold
        )

    logger.info(
        "Filtering out low proportion intBCs in finalized lineage groups..."
    )
    filtered_lgs = l_utils.filter_intbcs_final_lineages(
        allele_table, min_intbc_thresh=min_intbc_thresh
    )

    allele_table = l_utils.filtered_lineage_group_to_allele_table(filtered_lgs)

    logger.debug("Final lineage group assignments:")
    for n, g in allele_table.groupby(["lineageGrp"]):
        logger.debug(f"LG {n}: " + str(len(g["cellBC"].unique())) + " cells")

    logger.info("Filtering out low UMI cell barcodes...")
    allele_table = utilities.filter_cells(
        allele_table,
        min_umi_per_cell=int(min_umi_per_cell),
        min_avg_reads_per_umi=min_avg_reads_per_umi,
    )
    allele_table["lineageGrp"] = allele_table["lineageGrp"].astype(int)

    if plot:
        logger.info("Producing Plots...")
        at_pivot_I = pd.pivot_table(
            allele_table,
            index="cellBC",
            columns="intBC",
            values="UMI",
            aggfunc="count",
        )
        at_pivot_I.fillna(value=0, inplace=True)
        at_pivot_I[at_pivot_I > 0] = 1

        logger.info("Producing pivot table heatmap...")
        l_utils.plot_overlap_heatmap(allele_table, at_pivot_I, output_directory)

        logger.info("Plotting filtered lineage group pivot table heatmap...")
        l_utils.plot_overlap_heatmap_lg(
            allele_table, at_pivot_I, output_directory
        )

    return allele_table
