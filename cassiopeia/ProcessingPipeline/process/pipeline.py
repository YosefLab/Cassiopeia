"""
This file contains all high-level functionality for preprocessing sequencing
data into character matrices ready for phylogenetic inference. This file
is mainly invoked by cassiopeia_preprocess.py.
"""

import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pysam
import time

from tqdm.auto import tqdm
from pathlib import Path

from cassiopeia.ProcessingPipeline.process import UMI_utils


def resolve_UMI_sequence(
    molecule_table: pd.DataFrame,
    output_directory: str,
    min_avg_reads_per_umi: float = 2.0,
    min_umi_per_cell: int = 10,
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

    # ---------------- Plot Diagnositics after Resolving ---------------- #
    h = plt.figure(figsize=(14, 10))
    plt.plot(top_reads.values(), total_numReads.values(), "r.")
    plt.ylabel("Total Reads")
    plt.xlabel("Number Reads for Picked Sequence")
    plt.title("Total vs. Top Reads for Picked Sequence")
    plt.savefig(
        os.path.join(output_directory, "/total_vs_top_reads_pickSeq.png")
    )
    plt.close()

    h = plt.figure(figsize=(14, 10))
    plt.plot(first_reads.values(), second_reads.values(), "r.")
    plt.ylabel("Number Reads for Second Best Sequence")
    plt.xlabel("Number Reads for Picked Sequence")
    plt.title("Second Best vs. Top Reads for Picked Sequence")
    plt.savefig(
        os.path.join(output_directory + "/second_vs_top_reads_pickSeq.png")
    )
    plt.close()

    return filt_molecule_table


def collapseUMIs(
    out_dir: str,
    bam_fp: str,
    max_hq_mismatches: int = 3,
    max_indels: int = 2,
    n_threads: int = 1,
    show_progress: bool = True,
    force_sort: bool = True,
):
    """Collapses close UMIs together from a bam file.

    On a basic level, it aggregates together identical or close reads to count 
    how many times a UMI was read. Performs basic error correction, allowing 
    UMIs to be collapsed together which differ by at most a certain number of 
    high quality mismatches and indels in the sequence read itself. Writes out 
    a dataframe of the collapsed UMIs table.

    Args:
        out_dir: The output directory where the sorted bam directory, the
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
    if out_dir[-1] == "/":
        out_dir = out_dir[:-1]
    sorted_file_name = Path(
        out_dir
        + "/"
        + ".".join(bam_fp.split("/")[-1].split(".")[:-1])
        + "_sorted.bam"
    )

    if force_sort or not sorted_file_name.exists():
        max_read_length, total_reads_out = UMI_utils.sort_cellranger_bam(
            bam_fp,
            str(sorted_file_name),
            show_progress=show_progress,
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
    convertBam2DF(str(collapsed_file_name), str(collapsed_df_file_name))
    logging.info("Collapsed bam directory saved to " + str(collapsed_file_name))
    logging.info("Converted dataframe saved to " + str(collapsed_df_file_name))


def convertBam2DF(
    data_fp: str, out_fp: str, create_pd: bool = False
) -> pd.DataFrame:
    """Converts a BAM file to a dataframe.

    Saves the contents of a BAM file to a tab-delimited table saved to a text
    file. Rows represent alignments with relevant fields such as the CellBC,
    UMI, read count, sequence, and sequence qualities.

    Args:
      data_fp: The input filepath for the BAM file to be converted.
      out_fp: The output filepath specifying where the resulting dataframe is to
        be stored and its name.
      create_pd: Specifies whether to generate and return a pd.Dataframe.

    Returns:
      If create_pd: a pd.Dataframe containing the BAM information.
      Else: None, output saved to file

    """
    f = open(out_fp, "w")
    f.write("cellBC\tUMI\treadCount\tgrpFlag\tseq\tqual\treadName\n")

    als = []

    bam_fh = pysam.AlignmentFile(
        data_fp, ignore_truncation=True, check_sq=False
    )
    for al in bam_fh:
        cellBC, UMI, readCount, grpFlag = al.query_name.split("_")
        seq = al.query_sequence
        qual = al.query_qualities
        # Pysam qualities are represented as an array of unsigned chars,
        # so they are converted to the ASCII-encoded format that are found 
        # in the typical SAM formatting. 
        encode_qual = "".join(map(lambda x: chr(x + 33), qual))
        f.write(
            cellBC
            + "\t"
            + UMI
            + "\t"
            + readCount
            + "\t"
            + grpFlag
            + "\t"
            + seq
            + "\t"
            + encode_qual
            + "\t"
            + al.query_name
            + "\n"
        )

        if create_pd:
            als.append(
                [
                    cellBC,
                    UMI,
                    int(readCount),
                    grpFlag,
                    seq,
                    encode_qual,
                    al.query_name,
                ]
            )

    f.close()

    if create_pd:
        df = pd.DataFrame(als)
        df = df.rename(
            columns={
                0: "cellBC",
                1: "UMI",
                2: "readCount",
                3: "grpFlag",
                4: "seq",
                5: "qual",
                6: "readName",
            }
        )
        return df

def collapseDF2fastq(data_fp, out_fp):
    df = pd.read_csv(data_fp, sep = "\t")
    f = open(out_fp, "w")
    for i, row in df.iterrows():
        cellBC = row[0]
        UMI = row[1]
        readCount = row[2]
        seq = row[4]
        qual = row[5]
        f.write(
            "@" 
            + cellBC
            + "_"
            + UMI
            + "_"
            + str(readCount)
            + "\n"
            + seq
            + "\n"
            + "+"
            + "\n"
            + qual
            + "\n"
        )
    f.close()
