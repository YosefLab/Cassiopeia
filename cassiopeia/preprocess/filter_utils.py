"""
This file contains functions pertaining to filtering alignment tables.
Invoked through pipeline.py and supports the filter_alignments and 
call_lineage_group functions. 
"""

import logging
import os
import sys

import Levenshtein
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from typing import Tuple
from tqdm import tqdm

from . import lineageGroup_utils as lg_utils


def record_stats(
    moleculetable: pd.DataFrame,
) -> Tuple[np.array, np.array, np.array]:
    """
    Simple function to record the number of UMIs.

    Args:
        moleculetable: A DataFrame of alignments

    Returns:
        Read counts for each alignment, number of unique UMIs per intBC, number
            of UMIs per cellBC
    """

    # Count UMI per intBC
    umi_per_ibc = np.array([])
    for n, g in tqdm(moleculetable.groupby(["cellBC"]), desc="Recording stats"):
        x = g.groupby(["intBC"]).agg({"UMI": "nunique"})["UMI"]
        if x.shape[0] > 0:
            umi_per_ibc = np.concatenate([umi_per_ibc, np.array(x)])

    # Count UMI per cellBC
    umi_per_cbc = (
        moleculetable.groupby(["cellBC"])
        .agg({"UMI": "count"})
        .sort_values("UMI", ascending=False)["UMI"]
    )

    return (
        np.array(moleculetable["ReadCount"]),
        umi_per_ibc,
        np.array(umi_per_cbc),
    )


def filter_cellbcs(
    moleculetable: pd.DataFrame, umiCountThresh: int = 10, verbose: bool = False
) -> pd.DataFrame:
    """
    Filter out cell barcodes that have too few UMIs.

    Filters out cell barcodes that have a number of unique UMIs <= umiCountThresh.
    Assumes that UMIs have been resolved, i.e. that each UMI only appears once.

    Args:
        moleculetable: A DataFrame of alignments to be filtered
        umiCountThresh: The minimum number of UMIs needed for a cellBC to be
            included in the filtered DataFrame
        verbose: Indicates whether to log the number of cellBCs and UMIs
            remaining after filtering

    Returns:
        n_moleculetable: A filtered DataFrame of alignments.
    """

    # Create a cell-filter dictionary for hash lookup lmter on when filling
    # in the table
    cell_filter = {}

    for n, group in tqdm(
        moleculetable.groupby(["cellBC"]), desc="Filtering cellBCs for UMIs"
    ):
        if group.shape[0] <= umiCountThresh:
            cell_filter[n] = "bad"
            # tooFewUMI_UMI.append(group.shape[0])
        else:
            cell_filter[n] = "good"

    # apply the filter using the hash table created above
    moleculetable["status"] = moleculetable["cellBC"].map(cell_filter)

    # filter based on status & reindex
    n_moleculetable = moleculetable[(moleculetable["status"] == "good")]
    n_moleculetable.index = [i for i in range(n_moleculetable.shape[0])]

    if verbose:
        lg_utils.generate_log_output(n_moleculetable)

    return n_moleculetable


def filter_umis(
    moleculetable: pd.DataFrame,
    readCountThresh: int = 100,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Filters out UMIs with too few reads.

    Filters out all UMIs with a read count <= readCountThresh.

    Args:
        moleculetable: A DataFrame of alignments to be filtered
        readCountThresh: The minimum read count needed for a UMI to be
            included in the filtered DataFrame
        verbose: Indicates whether to log the number of cellBCs and UMIs
            remaining after filtering

    Returns:
        n_moleculetable: A filtered DataFrame of alignments.
    """

    # filter based on status & reindex
    n_moleculetable = moleculetable[
        moleculetable["ReadCount"] > readCountThresh
    ]
    n_moleculetable.index = [i for i in range(n_moleculetable.shape[0])]

    if verbose:
        lg_utils.generate_log_output(n_moleculetable)

    return n_moleculetable


def error_correct_intbc(
    moleculetable: pd.DataFrame,
    prop: float = 0.5,
    umiCountThresh: int = 10,
    bcDistThresh: int = 1,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Error corrects close intBCs with small enough unique UMI counts.

    Considers each pair of intBCs sharing a cellBC in the DataFrame for
    correction. For a pair of intBCs, changes all instances of one to other if:
        1. They have the same allele.
        2. The Levenshtein distance between their sequences is <= bcDistThresh.
        3. The number of UMIs of the intBC to be changed is <= umiCountThresh.
        4. The proportion of the UMI count in the intBC to be changed out of the
        total UMI count in both intBCs <= prop.
    Note: Prop should be <= 0.5, as this algorithm only corrects intBCs with
    fewer/equal UMIs towards intBCs with more UMIs. Additionally, if multiple
    intBCs are within the distance threshold of an intBC, it corrects the intBC
    towards the intBC with the most UMIs.

    Args:
        moleculetable: A DataFrame of alignments to be filtered
        prop: proportion by which to filter integration barcodes
        umiCountThresh: maximum umi count for which to correct barcodes
        bcDistThresh: barcode distance threshold, to decide what's similar
            enough to error correct
        verbose: Indicates whether to log every cellBC correction and the
            number of cellBCs and UMIs remaining after filtering

    Returns:
        moleculetable: Filtered alignment table with error corrected intBCs
    """

    # create index filter hash map
    index_filter = {}
    for n in moleculetable.index.values:
        index_filter[n] = "good"

    recovered = 0
    numUMI_corrected = 0
    for name, grp in tqdm(
        moleculetable.groupby(["cellBC"]), desc="Error Correcting intBCs"
    ):

        x1 = (
            grp.groupby(["intBC", "allele"])
            .agg({"UMI": "count", "ReadCount": "sum"})
            .sort_values("UMI", ascending=False)
            .reset_index()
        )

        if x1.shape[0] > 1:
            for r1 in range(x1.shape[0]):
                iBC1, allele1 = x1.loc[r1, "intBC"], x1.loc[r1, "allele"]
                for r2 in range(r1 + 1, x1.shape[0]):
                    iBC2, allele2 = x1.loc[r2, "intBC"], x1.loc[r2, "allele"]
                    bclDist = Levenshtein.distance(iBC1, iBC2)
                    if bclDist <= bcDistThresh and allele1 == allele2:
                        totalCount = x1.loc[[r1, r2], "UMI"].sum()
                        umiCounts = x1.loc[[r1, r2], "UMI"]
                        props = umiCounts / totalCount

                        # if the alleles are the same and the proportions are good, then let's error correct
                        if props[r2] < prop and umiCounts[r2] <= umiCountThresh:
                            bad_locs = moleculetable[
                                (moleculetable["cellBC"] == name)
                                & (moleculetable["intBC"] == iBC2)
                                & (moleculetable["allele"] == allele2)
                            ]
                            recovered += 1
                            numUMI_corrected += len(bad_locs.index.values)
                            moleculetable.loc[
                                bad_locs.index.values, "intBC"
                            ] = iBC1

                            if verbose:
                                # f.write(name + "\t" + iBC2 + "\t" + iBC1 + "\t")
                                # f.write(str(x1.loc[r2, "UMI"]) + "\t" + str(x1.loc[r1, "UMI"]) + "\n")
                                logging.info(
                                    f"In cellBC {name}, intBC {iBC2} corrected to {iBC1},"
                                    + "correcting UMI "
                                    + str({x1.loc[r2, "UMI"]})
                                    + "to "
                                    + str({x1.loc[r1, "UMI"]})
                                )

    # return filtered allele table
    # n_moleculetable = moleculetable[(moleculetable["status"] == "good")]
    moleculetable.index = [i for i in range(moleculetable.shape[0])]

    if verbose:
        lg_utils.generate_log_output(moleculetable)

    return moleculetable
