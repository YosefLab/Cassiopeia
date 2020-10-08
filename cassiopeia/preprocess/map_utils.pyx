"""
This file contains functions pertaining to mapping intBCs. 
Invoked through pipeline.py and supports the filter_alignments function.
"""

import logging
import sys

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pylab
from tqdm import tqdm

from cassiopeia.preprocess import utilities

sys.setrecursionlimit(10000)


def map_intbcs(
    moleculetable: pd.DataFrame, verbose: bool = False
) -> pd.DataFrame:
    """Performs a procedure to cleanly assign one allele to each intBC/cellBC
    pairing

    For each intBC/cellBC pairing, selects the most frequent allele and
    removes alignments that don't have that allele.

    Args:
        moleculetable: A molecule table of cellBC-UMI pairs to be filtered
        verbose: Indicates whether to log every correction and the number of
            cellBCs and UMIs remaining after filtering

    Returns
        An allele table with one allele per cellBC-intBC pair
    """

    # Have to drop out all intBCs that are NaN
    moleculetable = moleculetable.dropna(subset=["intBC"])

    # create mappings from intBC/cellBC pairs to alleles
    moleculetable["status"] = "good"
    moleculetable["filter_column"] = moleculetable[["intBC", "cellBC"]].apply(
        lambda x: "_".join(x), axis=1
    )
    moleculetable["filter_column2"] = moleculetable[
        ["intBC", "cellBC", "allele"]
    ].apply(lambda x: "_".join(x), axis=1)
    moleculetable["allele_counter"] = moleculetable["allele"]

    filter_dict = {}

    # For each intBC/cellBC pair, we want only one allele (select majority 
    # allele for now)
    corrected = 0
    numUMI_corrected = 0
    for n, group in tqdm(
        moleculetable.groupby(["filter_column"]), desc="Mapping intBCs"
    ):

        x1 = (
            group.groupby(["filter_column2", "allele"])
            .agg(
                {"readCount": "sum", "allele_counter": "count", "UMI": "count"}
            )
            .sort_values("readCount", ascending=False)
            .reset_index()
        )

        # If we've found an intBC that corresponds to more than one allele in 
        # the same cell, then let's error correct towards the more frequently 
        # occuring allele

        # But, this will ALWAYS be the first allele because we sorted above, so
        # we can generalize and always assign the intBC to the first element in 
        # x1.

        a = x1.iloc[0]["allele"]

        # Let's still keep count of how many times we had to re-assign for 
        # logging purposes
        filter_dict[x1.iloc[0]["filter_column2"]] = "good"
        if x1.shape[0] > 1:

            for i in range(1, x1.shape[0]):
                filter_dict[x1.iloc[i]["filter_column2"]] = "bad"
                corrected += 1
                numUMI_corrected += x1.loc[i, "UMI"]

            if verbose:
                for i in range(1, x1.shape[0]):
                    logging.info(
                        f"In group {n}, re-assigned allele "
                        + str(x1.loc[i, "allele"])
                        + f" to {a},"
                        + " re-assigning UMI "
                        + str(x1.loc[i, "UMI"])
                        + " to "
                        + str(x1.loc[0, "UMI"])
                    )

    moleculetable["status"] = moleculetable["filter_column2"].map(filter_dict)
    moleculetable = moleculetable[(moleculetable["status"] == "good")]
    moleculetable.index = [i for i in range(moleculetable.shape[0])]
    moleculetable = moleculetable.drop(
        columns=["filter_column", "filter_column2", "allele_counter", "status"]
    )

    # log results
    if verbose:
        logging.info("Picking alleles:")
        logging.info(f"# Alleles removed: {corrected}")
        logging.info(
            f"# UMIs affected through removing alleles: {numUMI_corrected}"
        )
        utilities.generate_log_output(moleculetable)

    return moleculetable
