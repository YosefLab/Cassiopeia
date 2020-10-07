"""
This file stores generally important functionality for the Cassiopeia-Preprocess
pipeline.
"""
import logging

import numpy as np
import pandas as pd
import pysam
from tqdm.auto import tqdm


def filter_cells(
    molecule_table: pd.DataFrame,
    min_umi_per_cell: int = 10,
    min_avg_reads_per_umi: float = 2.0,
) -> pd.DataFrame:
    """Filter out cell barcodes that have too few UMIs or too few reads/UMI

    Args:
        molecule_table: MoleculeTable to be filtered.
        min_umi_per_cell: Minimum number of UMIs per cell.
        min_avg_reads_per_umi: Minimum coverage (i.e. average) reads / UMI in a cell

    Returns:
        A filtered MoleculeTable
    """

    tooFewUMI_UMI = []
    cellBC2nM = {}

    # Create a cell-filter dictionary for hash lookup later on when filling
    # in the table
    cell_filter = {}

    for n, group in tqdm(molecule_table.groupby(["cellBC"])):
        umi_per_cellBC_n = group.shape[0]
        reads_per_cellBC_n = group.agg({"readCount": "sum"}).readCount
        avg_reads_per_UMI_n = float(reads_per_cellBC_n) / float(
            umi_per_cellBC_n
        )
        if (umi_per_cellBC_n <= min_umi_per_cell) or (
            avg_reads_per_UMI_n <= min_avg_reads_per_umi
        ):
            cell_filter[n] = True
            tooFewUMI_UMI.append(group.shape[0])
        else:
            cell_filter[n] = False
            cellBC2nM[n] = group.shape[0]

    # apply the filter using the hash table created above
    molecule_table["filter"] = molecule_table["cellBC"].map(cell_filter)

    n_umi_filt = molecule_table[molecule_table["filter"] == True].shape[0]
    n_cells_filt = len(
        molecule_table.loc[molecule_table["filter"] == True, "cellBC"].unique()
    )

    logging.info(f"Filtered out {n_umi_filt} UMIs.")
    logging.info(f"Filtered out {n_cells_filt} cells.")

    filt_molecule_table = molecule_table[
        molecule_table["filter"] == False
    ].copy()
    filt_molecule_table.drop(columns=["filter"], inplace=True)
    return filt_molecule_table


def convert_bam_to_df(
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
