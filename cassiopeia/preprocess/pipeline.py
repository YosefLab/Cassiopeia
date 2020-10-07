"""
This file contains all high-level functionality for preprocessing sequencing
data into character matrices ready for phylogenetic inference. This file
is mainly invoked by cassiopeia_preprocess.py.
TODO: richardyz98: Standardize logging outputs across all modules
"""

import os
import time

from typing import List, Optional, Tuple

from Bio import SeqIO
from functools import partial
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skbio import alignment

from pathlib import Path
from tqdm.auto import tqdm

from cassiopeia.preprocess import alignment_utilities
from cassiopeia.preprocess import call_lineage_utils as cl_utils
from cassiopeia.preprocess import constants
from cassiopeia.preprocess import filter_utils
from cassiopeia.preprocess import lineageGroup_utils as lg_utils
from cassiopeia.preprocess import UMI_utils
from cassiopeia.preprocess import utilities


DNA_SUBSTITUTION_MATRIX = constants.DNA_SUBSTITUTION_MATRIX
BAM_CONSTANTS = constants.BAM_CONSTANTS
progress = tqdm


def collapse_umis(
    output_directory: str,
    bam_fp: str,
    max_hq_mismatches: int = 3,
    max_indels: int = 2,
    n_threads: int = 1,
    show_progress: bool = True,
    force_sort: bool = True,
) -> pd.DataFrame:
    """Collapses close UMIs together from a bam file.

    On a basic level, it aggregates together identical or close reads to count
    how many times a UMI was read. Performs basic error correction, allowing
    UMIs to be collapsed together which differ by at most a certain number of
    high quality mismatches and indels in the sequence read itself. Writes out
    a dataframe of the collapsed UMIs table.

    Args:
        output_directory: The output directory where the sorted bam directory, the
          collapsed bam directory, and the final collapsed table are written to.
        bam_file_name: File path of the bam_file. Just the bam file name can be
          specified if the bam already exists in the output directory.
        max_hq_mismatches: A threshold specifying the max number of high quality
          mismatches between the seqeunces of 2 aligned segments to be collapsed.
        max_indels: A threshold specifying the maximum number of differing indels
          allowed between the sequences of 2 aligned segments to be collapsed.
        n_threads: Number of threads used. Currently only supports single
          threaded use.
        show_progress: Allow progress bar to be shown.
        force_sort: Specify whether to sort the initial bam directory, regardless
          of if the sorted file already exists.

    Returns:
        None; output table is written to file.
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

    if force_sort or not sorted_file_name.exists():
        max_read_length, total_reads_out = UMI_utils.sort_cellranger_bam(
            bam_fp, str(sorted_file_name), show_progress=show_progress
        )
        logging.info("Sorted bam directory saved to " + str(sorted_file_name))
        logging.info("Max read length of " + str(max_read_length))
        logging.info("Total reads: " + str(total_reads_out))

    collapsed_file_name = sorted_file_name.with_suffix(".collapsed.bam")
    if not collapsed_file_name.exists():
        UMI_utils.form_collapsed_clusters(
            str(sorted_file_name),
            max_hq_mismatches,
            max_indels,
            show_progress=show_progress,
        )

    logging.info(f"Finished collapsing UMI sequences in {time.time() - t0} s.")
    collapsed_df_file_name = sorted_file_name.with_suffix(".collapsed.txt")

    df = utilities.convert_bam_to_df(
        str(collapsed_file_name),
        str(collapsed_df_file_name),
        create_pd=True,
    )
    logging.info("Collapsed bam directory saved to " + str(collapsed_file_name))
    logging.info("Converted dataframe saved to " + str(collapsed_df_file_name))

    return df


def resolve_umi_sequence(
    molecule_table: pd.DataFrame,
    output_directory: str,
    min_avg_reads_per_umi: float = 2.0,
    min_umi_per_cell: int = 10,
    plot: bool = True,
) -> pd.DataFrame:
    """Resolve a consensus sequence for each UMI.

    This procedure will perform UMI and cellBC filtering on the basis of reads per
    UMI and UMIs per cell and then assign the most abundant sequence to each UMI
    if there is a set of conflicting sequences per UMI.

    Args:
      molecule_table: MoleculeTable to resolve
      output_directory: Directory to store results
      min_avg_reads_per_umi: Minimum covarage (i.e. average reads) per UMI allowed
      min_umi_per_cell: Minimum number of UMIs per cell allowed

    Return:
      A MoleculeTable with unique mappings between cellBC-UMI pairs.
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

    # ----------------- Select most abundant sequence ------------------ #

    mt_filter = {}
    total_numReads = {}
    top_reads = {}
    second_reads = {}
    first_reads = {}

    for _, group in tqdm(molecule_table.groupby(["cellBC", "UMI"])):

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
            os.path.join(output_directory + "second_vs_top_reads_pickSeq.png")
        )
        plt.close()

    filt_molecule_table = utilities.filter_cells(
        filt_molecule_table, min_umi_per_cell, min_avg_reads_per_umi
    )
    return filt_molecule_table


def align_sequences(
    queries: pd.DataFrame,
    ref_filepath: Optional[str] = None,
    ref: Optional[str] = None,
    gap_open_penalty: float = 20,
    gap_extend_penalty: float = 1,
) -> pd.DataFrame:
    """Align reads to the TargetSite refernece.

    Take in several queries store in a DataFrame mapping cellBC-UMIs to a
    sequence of interest and align each to a reference sequence. The alignment
    algorithm used is the Smith-Waterman local alignment algorithm. The desired
    output consists of the best alignment score and the CIGAR string storing the
    indel locations in the query sequence.

    TODO(mattjones315): Parallelize?

    Args:
        queries: Dataframe storing a list of sequences to align.
        ref_filepath: Filepath to the reference FASTA.
        ref: Reference sequence.
        gapopen: Gap open penalty
        gapextend: Gap extension penalty

    Returns:
        A dataframe mapping each sequence name to the CIGAR string, quality,
        and original query sequence.
    """

    assert ref or ref_filepath

    alignment_dictionary = {}

    if ref_filepath:
        ref = str(list(SeqIO.parse(ref_filepath, "fasta"))[0].seq)

    logging.info("Beginning alignment to reference...")
    t0 = time.time()

    for umi in queries.index:

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
            query.ReadCount,
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
        "ReadCount",
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
        alignments: Alignments provided in dataframe
        ref_filepath: Filepath to the ference sequence
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
        A dataframe mapping each sequence alignment to the called indels.
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
    _id: str,
    max_UMI_distance: int = 2,
    show_progress: bool = False,
    verbose: bool = False,
) -> pd.DataFrame:
    """Within cellBC-intBC pairs, collapses UMIs that have close sequences.

    Error correct UMIs together within cellBC-intBC pairs. UMIs that have a
    Hamming Distance between their sequences less than a threshold are
    corrected towards whichever UMI is more abundant.

    Args:
        input_df: Input DataFrame of alignments.
        _id: Identification of sample.
        max_UMI_distance: Maximum Hamming distance between UMIs
            for error correction.
        show_progress: Allow a progress bar to be shown.
        verbose: Log every UMI correction.

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
        ["cellBC", "intBC", "ReadCount", "UMI"],
        ascending=[True, True, False, False],
    )

    if max_UMI_distance == 0:
        logging.info(
            "Distance of 0, no correction occured, all alignments returned"
        )
        return sorted_df

    num_corrected = 0
    total = 0

    alignment_df = pd.DataFrame()

    if show_progress:
        sorted_df = progress(sorted_df, total=total, desc="Collapsing")

    allele_groups = sorted_df.groupby(["cellBC", "intBC"])

    for fields, allele_group in allele_groups:
        cellBC, intBC = fields
        if verbose:
            logging.info(f"cellBC: {cellBC}, intBC: {intBC}")
        (allele_group, num_corr, tot) = UMI_utils.correct_umis_in_group(
            allele_group, _id, max_UMI_distance
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
        lambda x: "_".join([x.cellBC, x.UMI, str(int(x.ReadCount))]), axis=1
    )

    alignment_df.set_index("readName", inplace=True)
    alignment_df.reset_index(inplace=True)

    return alignment_df


def filter_alignments(
    input_df: pd.DataFrame,
    output_directory: str,
    cell_umi_thresh: int = 10,
    umi_read_thresh: int = None,
    intbc_prop_thresh: float = 0.5,
    intbc_umi_thresh: int = 10,
    intbc_dist_thresh: int = 1,
    doublet_threshold: float = 0.35,
    plot: bool = False,
    verbose: bool = False,
) -> pd.DataFrame:
    """A wrapper function to perform multiple filtering and correcting steps
    on a DataFrame of alignments (reads).

    Performs the following steps on the alignments in a DataFrame:
        1. Filters out all alignments of a cellBC if the number of unique UMIs
        in alignments with that cellBC <= cell_umi_thresh.
        2. Filters out all alignments with a certain UMI if the read count of
        that UMI <= umi_read_thresh.
        3. Error corrects intBCs by changing intBCs with low UMI counts to
        intBCs with the same allele and a close sequence.
        4. Filters out alignments of a cellBC if alignments in that cellBC
        provide too much conflicting allele information.

    Args:
        input_df: A DataFrame of alignments
        output_directory: The output directory path to store plots
        cell_umi_thresh: A cell needs to have a higher UMI count than this
            value to be kept (not filtered out)
        umi_read_thresh: A UMI needs to have a higher read count than this
            value to be kept (not filtered out)
        intbc_prop_thresh: For a intBC to be corrected to another, its
            proportion of the total UMI counts of both intBCs needs to be less
            than this value
        intbc_umi_thresh: An intBC needs to have less than this value in order
            to be corrected to another
        intbc_dist_thresh: An intBC needs to have the Levenshtein Distance
            between its sequence and the sequence of another intBC to be
            corrected to the other
        doublet_threshold: A cell needs to have aproportion of alignments with
            conflicting allele information less than this value to be kept.
            Set to None to skip doublet detection
        plot: Indicates whether to plot the change in intBC and cellBC counts
            across filtering stages
        verbose: Indicates whether to log detailed information on each filter
            and correction step

    Returns:
        A filtered and corrected DataFrame of alignments.

    """

    t0 = time.time()
    logging.info("Begin filtering reads...")
    input_df["status"] = "good"
    input_df.sort_values("ReadCount", ascending=False)
    rc_profile, upi_profile, upc_profile = {}, {}, {}
    lg_utils.generate_log_output(input_df, begin=True)

    logging.info("Logging initial stats...")
    if plot:
        (
            rc_profile["Init"],
            upi_profile["Init"],
            upc_profile["Init"],
        ) = filter_utils.record_stats(input_df)

    logging.info(
        f"Filtering out cell barcodes with fewer than {cell_umi_thresh} UMIs..."
    )
    filtered_df = filter_utils.filter_cellbcs(
        input_df, umiCountThresh=cell_umi_thresh, verbose=verbose
    )
    if plot:
        (
            rc_profile["CellFilter"],
            upi_profile["CellFilter"],
            upc_profile["CellFilter"],
        ) = filter_utils.record_stats(filtered_df)

    logging.info(f"Filtering UMIs with read threshold {umi_read_thresh}...")
    if umi_read_thresh is None:
        R = filtered_df["ReadCount"]
        if list(R):
            umi_read_thresh = np.percentile(R, 99) // 10
        else:
            umi_read_thresh = 0
    filtered_df = filter_utils.filter_umis(
        filtered_df, readCountThresh=umi_read_thresh, verbose=verbose
    )

    if plot:
        (
            rc_profile["Filtered_UMI"],
            upi_profile["Filtered_UMI"],
            upc_profile["Filtered_UMI"],
        ) = filter_utils.record_stats(filtered_df)

    if intbc_dist_thresh > 0:
        logging.info("Error correcting intBCs...")
        filtered_df = filter_utils.error_correct_intbc(
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
        ) = filter_utils.record_stats(filtered_df)

    logging.info("Filtering cell barcodes one more time...")
    filtered_df = filter_utils.filter_cellbcs(
        filtered_df, umiCountThresh=cell_umi_thresh, verbose=verbose
    )

    if doublet_threshold:
        logging.info(
            f"Filtering out intra-lineage group doublets with proportion {doublet_threshold}..."
        )
        filtered_df = lg_utils.filter_intra_doublets(
            filtered_df, prop=doublet_threshold, verbose=verbose
        )

    logging.info("Mapping remaining intBC conflicts...")
    filtered_df = lg_utils.map_intbcs(filtered_df, verbose=verbose)
    if plot:
        (
            rc_profile["Final"],
            upi_profile["Final"],
            upc_profile["Final"],
        ) = filter_utils.record_stats(filtered_df)

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
    out_fn: str,
    output_directory: str,
    cell_umi_filter: int = 10,
    min_cluster_prop: float = 0.005,
    min_intbc_thresh: float = 0.05,
    inter_doublet_threshold: float = 0.35,
    kinship_thresh: float = 0.25,
    verbose: bool = False,
    plot: bool = False,
):
    """Assigns cells represented as cellBCs to their clonal populations
    (lineage groups) based on the groups of intBCs they share.

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
        input_df: The alignment DataFrame to be annotated with lineage
            assignments
        out_fn: The file name of the final table
        output_directory: The folder to store the final table as well as plots
        cell_umi_filter: The threshold specifying the minimum number of UMIs a
            cell needs in order to not be filtered out
        min_cluster_prop: The minimum cluster size in the putative lineage
            assignment step, as a proportion of the number of cells
        min_intbc_thresh: The threshold specifying the minimum proportion of
            cells in a lineage group that need to have an intBC in order for it
            to not be filtered out. Also specifies the minimum proportion of
            cells that share an intBC with the most frequence intBC in forming
            putative lineage groups
        inter_doublet_threshold: The threshold specifying the minimum proportion
            of kinship a cell shares with its assigned lineage group out of all
            lineage groups for it not to be filtered out as an inter-lineage
            doublet
        kinship_thresh: Specifies the proportion of intBCs that a cell needs to
            share with the intBC set of a lineage group such that it is
            assigned to that lineage group in putative assignment
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
    piv_assigned = cl_utils.assign_lineage_groups(
        piv,
        min_clust_size,
        min_intbc_thresh=min_intbc_thresh,
        kinship_thresh=kinship_thresh,
    )

    logging.info("Refining lineage groups...")
    logging.info(
        "Redefining lineage groups by removing low proportion intBCs..."
    )
    master_LGs, master_intBCs = cl_utils.filter_intbcs_lg_sets(
        piv_assigned, min_intbc_thresh=min_intbc_thresh
    )

    logging.info("Reassigning cells to refined lineage groups by kinship...")
    kinship_scores = cl_utils.score_lineage_kinships(
        piv_assigned, master_LGs, master_intBCs
    )

    logging.info("Annotating alignment table with refined lineage groups...")
    at = cl_utils.annotate_lineage_groups(
        input_df, kinship_scores, master_intBCs
    )
    if inter_doublet_threshold:
        logging.info(
            f"Filtering out inter-lineage group doublets with proportion {inter_doublet_threshold}..."
        )
        at = lg_utils.filter_inter_doublets(
            at, rule=inter_doublet_threshold, verbose=verbose
        )

    logging.info(
        "Filtering out low proportion intBCs in finalized lineage groups..."
    )
    filtered_lgs = cl_utils.filter_intbcs_final_lineages(
        at, min_intbc_thresh=min_intbc_thresh
    )
    at = cl_utils.filtered_lineage_group_to_allele_table(filtered_lgs)

    if verbose:
        logging.info("Final lineage group assignments:")
        for n, g in at.groupby(["lineageGrp"]):
            logging.info(
                f"LG {n}: " + str(len(g["cellBC"].unique())) + " cells"
            )

    logging.info("Filtering out low UMI cell barcodes...")
    at = filter_utils.filter_cellbcs(
        at,
        umiCountThresh=int(cell_umi_filter),
        verbose=verbose,
    )
    at = at.drop(columns=["status"])
    at["lineageGrp"] = at["lineageGrp"].astype(int)
    at.set_index("readName", inplace=True)
    at.reset_index(inplace=True)

    final_time = time.time()
    logging.info(f"Finished filtering alignments in {final_time - t0}.")

    at.to_csv(output_directory + "/" + out_fn, sep="\t", index=False)

    if plot:
        logging.info("Producing Plots...")
        at_pivot_I = pd.pivot_table(
            at, index="cellBC", columns="intBC", values="UMI", aggfunc="count"
        )
        at_pivot_I.fillna(value=0, inplace=True)
        at_pivot_I[at_pivot_I > 0] = 1

        logging.info("Producing pivot table heatmap...")
        cl_utils.plot_overlap_heatmap(at_pivot_I, at, output_directory)

        logging.info("Plotting filtered lineage group pivot table heatmap...")
        cl_utils.plot_overlap_heatmap_lg(at, at_pivot_I, output_directory)
