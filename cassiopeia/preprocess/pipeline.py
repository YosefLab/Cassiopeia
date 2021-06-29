"""
This file contains all high-level functionality for preprocessing sequencing
data into character matrices ready for phylogenetic inference. This file
is mainly invoked by cassiopeia_preprocess.py.
"""

from functools import partial
import logging
import os
from pathlib import Path
import time
from typing import List, Optional, Tuple

from Bio import SeqIO
import matplotlib.pyplot as plt
import ngs_tools as ngs
import numpy as np
import pandas as pd
import pysam
from skbio import alignment
from tqdm.auto import tqdm
from typing_extensions import Literal

from cassiopeia.preprocess import alignment_utilities
from cassiopeia.preprocess import constants
from cassiopeia.preprocess import map_utils as m_utils
from cassiopeia.preprocess import doublet_utils as d_utils
from cassiopeia.preprocess import lineage_utils as l_utils
from cassiopeia.preprocess import UMI_utils
from cassiopeia.preprocess import utilities


DNA_SUBSTITUTION_MATRIX = constants.DNA_SUBSTITUTION_MATRIX
BAM_CONSTANTS = constants.BAM_CONSTANTS
progress = tqdm


class PreprocessError(Exception):
    pass


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

    logging.info("Converting FASTQs to unmapped BAM...")
    t0 = time.time()

    tag_map = constants.CHEMISTRY_BAM_TAGS[chemistry]
    bam_fp = os.path.join(
        output_directory, f"{name}_unmapped.bam" if name else "unmapped.bam"
    )
    ngs.fastq.fastqs_to_bam_with_chemistry(
        fastq_fps,
        ngs.chemistry.get_chemistry(chemistry),
        tag_map,
        bam_fp,
        name=name,
        show_progress=True,
        n_threads=n_threads,
    )
    logging.info(f"Finished writing unmapped BAM in {time.time() - t0} s.")
    return bam_fp


def error_correct_barcodes(
    bam_fp: str, output_directory: str, whitelist_fp: str, n_threads: int = 1
) -> str:
    """Error-correct barcodes in the input BAM.

    The barcode correction procedure used in Cell Ranger by 10X Genomics is used.
    https://kb.10xgenomics.com/hc/en-us/articles/115003822406-How-does-Cell-Ranger-correct-barcode-sequencing-errors

    Args:
        bam_fp: Input BAM filepath containing raw barcodes
        output_directory: The output directory where the corrected BAM will be
            written to. This directory must exist prior to calling this function.
        whitelist_fp: Path to plaintext file containing barcode whitelist, one
            barcode per line.
        n_threads: Number of threads to use. Defaults to 1.

    Todo:
        Currently, the user must provide their own whitelist, and Cassiopeia
        does not use any of the whitelists provided by the ngs-tools library.
        At some point, we should update the pipeline so that if ngs-tools
        provides a pre-packaged whitelists, it uses that for those chemistries.

    Returns:
        Path to corrected BAM
    """
    logging.info("Correcting barcodes to whitelist...")
    t0 = time.time()

    # Read whitelist
    with open(whitelist_fp, "r") as f:
        whitelist = [line.strip() for line in f if not line.isspace()]

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
    logging.info(f"Finished correcting barcodes in {time.time() - t0} s.")
    return corrected_fp


def collapse_umis(
    bam_fp: str,
    output_directory: str,
    max_hq_mismatches: int = 3,
    max_indels: int = 2,
    skip_existing: bool = False,
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
        skip_existing: Indicates whether to check if the output files already
            exist in the output directory for the sorting and collapsing steps.
            Skips each step if the respective file already exists

    Returns:
        A DataFrame of collapsed reads.
    """

    logging.info("Collapsing UMI sequences...")

    t0 = time.time()

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
    logging.info(f"Using BAM tag `{cell_bc_tag}` as cell barcodes")

    if not sorted_file_name.exists() and not skip_existing:
        max_read_length, total_reads_out = UMI_utils.sort_bam(
            bam_fp,
            str(sorted_file_name),
            sort_key=lambda al: (
                al.get_tag(cell_bc_tag),
                al.get_tag(BAM_CONSTANTS["UMI_TAG"]),
            ),
            filter_func=lambda al: al.has_tag(cell_bc_tag),
        )
        logging.info("Sorted bam directory saved to " + str(sorted_file_name))
        logging.info("Max read length of " + str(max_read_length))
        logging.info("Total reads: " + str(total_reads_out))

    collapsed_file_name = sorted_file_name.with_suffix(".collapsed.bam")
    if not collapsed_file_name.exists() and not skip_existing:
        UMI_utils.form_collapsed_clusters(
            str(sorted_file_name),
            max_hq_mismatches,
            max_indels,
            cell_key=lambda al: al.get_tag(cell_bc_tag),
        )

    logging.info(f"Finished collapsing UMI sequences in {time.time() - t0} s.")
    collapsed_df_file_name = sorted_file_name.with_suffix(".collapsed.txt")

    df = utilities.convert_bam_to_df(
        str(collapsed_file_name), str(collapsed_df_file_name), create_pd=True
    )
    logging.info("Collapsed bam directory saved to " + str(collapsed_file_name))
    logging.info("Converted dataframe saved to " + str(collapsed_df_file_name))

    return df


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
        verbose: Indicates whether to log the number of cellBCs and UMIs
            remaining after filtering

    Return:
        A molecule table with unique mappings between cellBC-UMI pairs.
    """

    logging.info("Resolving UMI sequences...")

    t0 = time.time()

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

    unique_pairs = molecule_table.groupby(["cellBC", "UMI"])

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
                "readCount", ascending=False
            ).reset_index()
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
    n_filtered = molecule_table[molecule_table["filter"] == True].shape[0]

    logging.info(f"Filtered out {n_filtered} reads.")

    # filter based on status & reindex
    filt_molecule_table = molecule_table[
        molecule_table["filter"] == False
    ].copy()
    filt_molecule_table.drop(columns=["filter"], inplace=True)

    logging.info(f"Finished resolving UMI sequences in {time.time() - t0}s.")

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


def align_sequences(
    queries: pd.DataFrame,
    ref_filepath: Optional[str] = None,
    ref: Optional[str] = None,
    gap_open_penalty: float = 20,
    gap_extend_penalty: float = 1,
) -> pd.DataFrame:
    """Align reads to the TargetSite reference.

    Take in several queries stored in a DataFrame mapping cellBC-UMIs to a
    sequence of interest and align each to a reference sequence. The alignment
    algorithm used is the Smith-Waterman local alignment algorithm. The desired
    output consists of the best alignment score and the CIGAR string storing the
    indel locations in the query sequence.

    TODO(mattjones315): Parallelize?

    Args:
        queries: DataFrame storing a list of sequences to align.
        ref_filepath: Filepath to the reference FASTA.
        ref: Reference sequence.
        gapopen: Gap open penalty
        gapextend: Gap extension penalty

    Returns:
        A DataFrame mapping each sequence name to the CIGAR string, quality,
        and original query sequence.
    """
    assert ref or ref_filepath

    alignment_dictionary = {}

    if ref_filepath:
        ref = str(list(SeqIO.parse(ref_filepath, "fasta"))[0].seq)

    logging.info("Beginning alignment to reference...")
    t0 = time.time()

    for umi in tqdm(
        queries.index,
        total=queries.shape[0],
        desc="Aligning sequences to reference",
    ):

        query = queries.loc[umi]

        aligner = alignment.StripedSmithWaterman(
            query.seq,
            substitution_matrix=DNA_SUBSTITUTION_MATRIX,
            gap_open_penalty=gap_open_penalty,
            gap_extend_penalty=gap_extend_penalty,
        )
        aln = aligner(ref)
        alignment_dictionary[query.readName] = (
            query.cellBC,
            query.UMI,
            query.readCount,
            aln.cigar,
            aln.query_begin,
            aln.target_begin,
            aln.optimal_alignment_score,
            aln.query_sequence,
        )

    final_time = time.time()

    logging.info(f"Finished aligning in {final_time - t0}.")
    logging.info(
        f"Average time to align each sequence: {(final_time - t0) / queries.shape[0]})"
    )

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

    assert ref or ref_filepath

    alignment_to_indel = {}
    alignment_to_intBC = {}

    if ref_filepath:
        ref = str(list(SeqIO.parse(ref_filepath, "fasta"))[0].seq)

    logging.info("Calling indels...")
    t0 = time.time()

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

    final_time = time.time()

    logging.info(f"Finished calling alleles in {final_time - t0}s")
    return alignments


def error_correct_umis(
    input_df: pd.DataFrame,
    max_umi_distance: int = 2,
    verbose: bool = False,
) -> pd.DataFrame:
    """Within cellBC-intBC pairs, collapses UMIs that have close sequences.

    Error correct UMIs together within cellBC-intBC pairs. UMIs that have a
    Hamming Distance between their sequences less than a threshold are
    corrected towards whichever UMI is more abundant.

    Args:
        input_df: Input DataFrame of alignments.
        max_umi_distance: The threshold specifying the Maximum Hamming distance
            between UMIs for one to be corrected to another.
        verbose: Indicates whether to log every UMI correction.

    Returns:
        A DataFrame of error corrected UMIs.
    """

    assert (
        len(
            [
                i
                for i in input_df.groupby(["cellBC", "intBC", "UMI"]).size()
                if i > 1
            ]
        )
        == 0
    ), "Non-unique cellBC-UMI pair exists, please resolve UMIs."

    t0 = time.time()

    logging.info("Beginning error correcting UMIs...")

    sorted_df = input_df.sort_values(
        ["cellBC", "intBC", "readCount", "UMI"],
        ascending=[True, True, False, False],
    )

    if max_umi_distance == 0:
        logging.info(
            "Distance of 0, no correction occurred, all alignments returned"
        )
        return sorted_df

    num_corrected = 0
    total = 0

    alignment_df = pd.DataFrame()

    allele_groups = sorted_df.groupby(["cellBC", "intBC"])

    allele_groups = progress(
        allele_groups, total=len(allele_groups), desc="Error-correcting UMIs"
    )

    for fields, allele_group in allele_groups:
        cellBC, intBC = fields
        if verbose:
            logging.info(f"cellBC: {cellBC}, intBC: {intBC}")
        allele_group, num_corr, tot = UMI_utils.correct_umis_in_group(
            allele_group, max_umi_distance
        )
        num_corrected += num_corr
        total += tot

        alignment_df = alignment_df.append(allele_group, sort=True)

    final_time = time.time()

    logging.info(f"Finished error correcting UMIs in {final_time - t0}.")
    logging.info(
        f"{num_corrected} UMIs Corrected of {total}"
        + f"({round(float(num_corrected) / total, 5) * 100}%)"
    )

    alignment_df["readName"] = alignment_df.apply(
        lambda x: "_".join([x.cellBC, x.UMI, str(int(x.readCount))]), axis=1
    )

    alignment_df.set_index("readName", inplace=True)
    alignment_df.reset_index(inplace=True)

    return alignment_df


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
    plot: bool = False,
    verbose: bool = False,
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
        plot: Indicates whether to plot the change in intBC and cellBC counts
            across filtering stages
        verbose: Indicates whether to log detailed information on each filter
            and correction step

    Returns:
        A filtered and corrected allele table of cellBC-UMI-allele groups

    """

    t0 = time.time()
    logging.info("Begin filtering reads...")
    input_df["status"] = "good"
    input_df.sort_values("readCount", ascending=False, inplace=True)
    rc_profile, upi_profile, upc_profile = {}, {}, {}
    utilities.generate_log_output(input_df, begin=True)

    logging.info("Logging initial stats...")
    if plot:
        (
            rc_profile["Init"],
            upi_profile["Init"],
            upc_profile["Init"],
        ) = utilities.record_stats(input_df)

    logging.info(
        f"Filtering out cell barcodes with fewer than {min_umi_per_cell} UMIs..."
    )
    filtered_df = utilities.filter_cells(
        input_df,
        min_umi_per_cell=min_umi_per_cell,
        min_avg_reads_per_umi=min_avg_reads_per_umi,
        verbose=verbose,
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
    logging.info(f"Filtering UMIs with read threshold {umi_read_thresh}...")
    filtered_df = utilities.filter_umis(
        filtered_df, readCountThresh=umi_read_thresh, verbose=verbose
    )

    if plot:
        (
            rc_profile["Filtered_UMI"],
            upi_profile["Filtered_UMI"],
            upc_profile["Filtered_UMI"],
        ) = utilities.record_stats(filtered_df)

    if intbc_dist_thresh > 0:
        logging.info("Error correcting intBCs...")
        filtered_df = utilities.error_correct_intbc(
            filtered_df,
            prop=intbc_prop_thresh,
            umiCountThresh=intbc_umi_thresh,
            bcDistThresh=intbc_dist_thresh,
            verbose=verbose,
        )

    if plot:
        (
            rc_profile["Process_intBC"],
            upi_profile["Process_intBC"],
            upc_profile["Process_intBC"],
        ) = utilities.record_stats(filtered_df)

    logging.info("Filtering cell barcodes one more time...")
    filtered_df = utilities.filter_cells(
        filtered_df,
        min_umi_per_cell=min_umi_per_cell,
        min_avg_reads_per_umi=min_avg_reads_per_umi,
        verbose=verbose,
    )

    if doublet_threshold:
        logging.info(
            f"Filtering out intra-lineage group doublets with proportion {doublet_threshold}..."
        )
        filtered_df = d_utils.filter_intra_doublets(
            filtered_df, prop=doublet_threshold, verbose=verbose
        )

    logging.info("Mapping remaining intBC conflicts...")
    filtered_df = m_utils.map_intbcs(filtered_df, verbose=verbose)
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

    final_time = time.time()
    logging.info(f"Finished filtering alignments in {final_time - t0}.")
    logging.info(
        f"Overall, filtered {cellBC_count} cells, with {filtered_df.shape[0]} UMIs."
    )

    filtered_df.set_index("readName", inplace=True)
    filtered_df.reset_index(inplace=True)

    return filtered_df


def call_lineage_groups(
    input_df: pd.DataFrame,
    output_directory: str,
    min_umi_per_cell: int = 10,
    min_avg_reads_per_umi: float = 2.0,
    min_cluster_prop: float = 0.005,
    min_intbc_thresh: float = 0.05,
    inter_doublet_threshold: float = 0.35,
    kinship_thresh: float = 0.25,
    verbose: bool = False,
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
        verbose: Indicates whether to log detailed information on filtering
            steps
        plot: Indicates whether to generate plots

    Returns:
        None, saves output allele table to file.
    """

    t0 = time.time()

    logging.info(
        f"{input_df.shape[0]} UMIs (rows), with {input_df.shape[1]} attributes (columns)"
    )
    logging.info(str(len(input_df["cellBC"].unique())) + " Cells")

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

    logging.info("Assigning initial lineage groups...")
    logging.info(f"Clustering with minimum cluster size {min_clust_size}...")
    piv_assigned = l_utils.assign_lineage_groups(
        piv,
        min_clust_size,
        min_intbc_thresh=min_intbc_thresh,
        kinship_thresh=kinship_thresh,
    )

    logging.info("Refining lineage groups...")
    logging.info(
        "Redefining lineage groups by removing low proportion intBCs..."
    )
    master_LGs, master_intBCs = l_utils.filter_intbcs_lg_sets(
        piv_assigned, min_intbc_thresh=min_intbc_thresh
    )

    logging.info("Reassigning cells to refined lineage groups by kinship...")
    kinship_scores = l_utils.score_lineage_kinships(
        piv_assigned, master_LGs, master_intBCs
    )

    logging.info("Annotating alignment table with refined lineage groups...")
    allele_table = l_utils.annotate_lineage_groups(
        input_df, kinship_scores, master_intBCs
    )
    if inter_doublet_threshold:
        logging.info(
            f"Filtering out inter-lineage group doublets with proportion {inter_doublet_threshold}..."
        )
        allele_table = d_utils.filter_inter_doublets(
            allele_table, rule=inter_doublet_threshold, verbose=verbose
        )

    logging.info(
        "Filtering out low proportion intBCs in finalized lineage groups..."
    )
    filtered_lgs = l_utils.filter_intbcs_final_lineages(
        allele_table, min_intbc_thresh=min_intbc_thresh
    )

    allele_table = l_utils.filtered_lineage_group_to_allele_table(filtered_lgs)

    if verbose:
        logging.info("Final lineage group assignments:")
        for n, g in allele_table.groupby(["lineageGrp"]):
            logging.info(
                f"LG {n}: " + str(len(g["cellBC"].unique())) + " cells"
            )

    logging.info("Filtering out low UMI cell barcodes...")
    allele_table = utilities.filter_cells(
        allele_table,
        min_umi_per_cell=int(min_umi_per_cell),
        min_avg_reads_per_umi=min_avg_reads_per_umi,
        verbose=verbose,
    )
    allele_table["lineageGrp"] = allele_table["lineageGrp"].astype(int)

    final_time = time.time()
    logging.info(f"Finished filtering alignments in {final_time - t0}.")

    if plot:
        logging.info("Producing Plots...")
        at_pivot_I = pd.pivot_table(
            allele_table,
            index="cellBC",
            columns="intBC",
            values="UMI",
            aggfunc="count",
        )
        at_pivot_I.fillna(value=0, inplace=True)
        at_pivot_I[at_pivot_I > 0] = 1

        logging.info("Producing pivot table heatmap...")
        l_utils.plot_overlap_heatmap(allele_table, at_pivot_I, output_directory)

        logging.info("Plotting filtered lineage group pivot table heatmap...")
        l_utils.plot_overlap_heatmap_lg(
            allele_table, at_pivot_I, output_directory
        )

    return allele_table
