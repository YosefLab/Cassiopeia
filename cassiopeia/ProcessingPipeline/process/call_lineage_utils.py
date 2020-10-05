import logging
import os
import sys

from matplotlib import colors, colorbar
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pylab
from tqdm import tqdm
from typing import Dict, List
import time

sys.setrecursionlimit(10000)
from . import lineageGroup_utils as lg_utils


def assign_lineage_groups(
    pivot_in: pd.DataFrame,
    min_clust_size: int,
    min_intbc_thresh: float = 0.2,
    kinship_thresh: float = 0.2,
    verbose: bool = False,
) -> pd.DataFrame:
    """A wrapper function to find lineage groups and assign cells to them.

    Iteratively finds lineage groups. Runs the algorithm that forms groups
    around the most frequent intBC in the unassigned cells until a group
    of <= min_clust_size is formed.

    Args:
        pivot_in: The input pivot table of UMI counts for each cellBC-intBC pair
        min_clust_size: The minimum number of cells needed for a group
        min_intbc_thresh: A parameter for the grouping algorithm. In order for
            an intBC to be included in the group set, it must have more than 
            this proportion of cells shared with the most frequent intBC
        kinship_thresh: A parameter for the grouping algorithm that determines
            the proportion of intBCs that a cell needs to share with the group
            in order to included in that group
        verbose: Indicates whether to log the size of each group

    Returns:
        piv_assigned: A pivot table of cells labled with lineage group 
            assignments
    """
    ## Run LG Assign function

    # initiate output variables
    piv_assigned = pd.DataFrame()

    # Loop for iteratively assigning LGs
    prev_clust_size = np.inf
    i = 0
    while prev_clust_size > min_clust_size:
        # run function
        piv_lg, piv_nolg = find_top_lg(
            pivot_in,
            i,
            min_intbc_prop=min_intbc_thresh,
            kinship_thresh=kinship_thresh,
            verbose=verbose,
        )

        # append returned objects to output variable
        piv_assigned = piv_assigned.append(piv_lg)

        # update pivot_in by removing assigned alignments
        pivot_in = piv_nolg

        prev_clust_size = piv_lg.shape[0]

        i += 1

    return piv_assigned


def find_top_lg(
    PIVOT_in: pd.DataFrame,
    iteration: int,
    min_intbc_prop: float = 0.2,
    kinship_thresh: float = 0.2,
    verbose: bool = False,
) -> (pd.pivot_table, pd.pivot_table):

    """Algorithm to creates lineage groups from a pivot table of UMI counts 
    for each cellBC-intBC pair. 

    First, identifies the most frequent intBC. Then, selects all intBCs that
    share a proportion of cells >= min_intbc_prop with the most frequent and 
    defines that as the cluster set. Then finds all cells that have >= 
    kinship_thresh intBCs that are in the cluster set and include them in the
    cluster. Finally outputs the cluster as the lineage group.

    Args:
        pivot_in: The input pivot table of UMI counts for each cellBC-intBC pair
        iteration: The cluster number and iteration number of the iterative 
            wrapper function
        min_intbc_thresh: In order for an intBC to be included in the cluster 
            set, it must have more than this proportion of cells shared with 
            the most frequent intBC
        kinship_thresh: Determines the proportion of intBCs that a cell needs 
            to share with the cluster in order to included in that cluster
        verbose: Indicates whether to log the size of each cluster

    Returns:
        PIV_LG: A pivot table of cells labled with lineage group assignments
        PIV_noLG: A pivot table of the remaining unassigned cells
    """

    # calculate sum of observed intBCs, identify top intBC
    intBC_sums = PIVOT_in.sum(0).sort_values(ascending=False)
    intBC_top = intBC_sums.index[0]

    # take subset of PIVOT table that contain cells that have the top intBC
    subPIVOT_in = PIVOT_in[PIVOT_in[intBC_top] > 0]
    subPIVOT_in_sums = subPIVOT_in.sum(0)
    ordered_intBCs2 = subPIVOT_in_sums.sort_values(
        ascending=False
    ).index.tolist()
    subPIVOT_in = subPIVOT_in[ordered_intBCs2]

    # binarize
    subPIVOT_in[subPIVOT_in > 0] = 1

    # Define intBC set
    subPIVOT_in_sums2 = subPIVOT_in.sum(0)
    total = subPIVOT_in_sums2[intBC_top]
    intBC_sums_filt = subPIVOT_in_sums2[
        subPIVOT_in_sums2 >= min_intbc_prop * total
    ]

    # Reduce PIV to only intBCs considered in set
    intBC_set = intBC_sums_filt.index.tolist()
    PIV_set = PIVOT_in.iloc[:, PIVOT_in.columns.isin(intBC_set)]

    # Calculate fraction of UMIs within intBC_set ("kinship") for each cell 
    # in PIV_set
    f_inset = PIV_set.sum(axis=1)

    # define set of cells with good kinship
    f_inset_filt = f_inset[f_inset >= kinship_thresh]
    LG_cells = f_inset_filt.index.tolist()

    # Return updated PIV with LG_cells removed
    PIV_noLG = PIVOT_in.iloc[~PIVOT_in.index.isin(LG_cells), :]

    # Return PIV with LG_cells assigned
    PIV_LG = PIVOT_in.iloc[PIVOT_in.index.isin(LG_cells), :]
    PIV_LG["lineageGrp"] = iteration + 1

    # Print statements

    if verbose:
        logging.info(
            f"LG {iteration+1} Assignment: {PIV_LG.shape[0]} cells assigned"
        )

    # # Plot distribution of kinship scores
    # h4 = plt.figure(figsize=(15,10))
    # ax4 = plt.hist(f_inset, bins=49, alpha=0.5, histtype='step')
    # yax4 = plt.yscale('log', basey=10)
    # plt.savefig(outputdir + "/kinship_scores.png")

    return PIV_LG, PIV_noLG


def filter_intbcs_lg_sets(
    PIV_assigned: pd.DataFrame, min_intbc_thresh: float = 0.2
) -> (List[int], Dict[int, pd.DataFrame]):
    """Filters out lineage group sets for low-proportion intBCs.

    For each lineage group, removes the intBCs that <= min_intbc_thresh
    proportion of cells in that group have. Effectively removes intBCs
    with low cell counts in each group from being considered for lineage
    reconstruction.

    Args:
        PIV_assigned: A pivot table of cells labled with lineage group
            assignments
        min_intbc_thresh: The minimum proportion of cells in a lineage group
            that must have an intBC for the intBC to remain in the lineage
            group set

    Returns:
        master_LGs: A list of the lineage groups
        master_intBCs: A dictionary that has mappings from the lineage group
            number to the set of intBCs being used for reconstruction

    """
    master_intBCs = {}
    master_LGs = []

    for i, PIV_i in PIV_assigned.groupby(["lineageGrp"]):
        PIVi_bin = PIV_i.copy()
        # Drop the lineageGroup column
        PIVi_bin = PIVi_bin.drop(
            ["lineageGrp"], axis=1
        )  
        PIVi_bin[PIVi_bin > 0] = 1

        intBC_sums = PIVi_bin.sum(0)
        intBC_normsums = intBC_sums / max(intBC_sums)

        intBC_normsums_filt_i = intBC_normsums[intBC_normsums >= min_intbc_thresh]
        intBC_set_i = intBC_normsums_filt_i.index.tolist()

        # Update masters
        master_intBCs[i] = intBC_set_i
        master_LGs.append(i)

    return master_LGs, master_intBCs


def score_lineage_kinships(
    PIV: pd.DataFrame, master_LGs: List[int], master_intBCs: Dict[int, pd.DataFrame]
) -> pd.DataFrame:
    """Identifies which lineage group each cell should belong to.

    Given a set of cells and a set of lineage groups with their intBCs sets, 
    identifies which lineage group each cell has the most kinship with. Kinship
    is defined as the total UMI count of intBCs shared between the cell and the
    intBC set of a lineage group.

    Args:
        PIV: A pivot table of cells labled with lineage group assignments
        master_LGs: A list of the lineage groups
        master_intBCs: A dictionary that has mappings from the lineage group
            number to the set of intBCs being used for reconstruction


    Returns:
        max_kinship_LG: A DataFrame that contains the lineage group for each
        cell with the greatest kinship
    """

    dfLG2intBC = pd.DataFrame()

    # Identifies which lineage groups have which intBCs in their set
    for i in range(len(master_LGs)):
        LGi = master_LGs[i]
        intBCsi = master_intBCs[LGi]
        dfi = pd.DataFrame(index=[LGi], columns=intBCsi, data=1)
        dfLG2intBC = dfLG2intBC.append(dfi, "sort=False")

    dfLG2intBC = dfLG2intBC.fillna(0)

    # Reorder
    flat_master_intBCs = []
    intBC_dupl_check = set()
    for key in master_intBCs.keys():
        sublist = master_intBCs[key]
        for item in sublist:
            if item not in intBC_dupl_check:
                flat_master_intBCs.append(item)
                intBC_dupl_check.add(item)

    dfLG2intBC = dfLG2intBC[flat_master_intBCs]

    # Construct matrices for multiplication
    subPIVOT = PIV[flat_master_intBCs]
    subPIVOT = subPIVOT.fillna(0)

    # Matrix math
    dfCellBC2LG = subPIVOT.dot(dfLG2intBC.T)
    max_kinship = dfCellBC2LG.max(axis=1)

    max_kinship_ind = dfCellBC2LG.idxmax(axis=1).to_frame()
    max_kinship_frame = max_kinship.to_frame()

    max_kinship_LG = pd.concat([max_kinship_frame, max_kinship_ind + 1], axis=1)
    max_kinship_LG.columns = ["maxOverlap", "lineageGrp"]

    # max_kinship_LG_filt = max_kinship_LG[max_kinship_LG['maxOverlap'] >= 0.75]

    # with open(outputdir + "/lglog.txt", "a") as f:
    #    f.write(str(max_kinship_LG.shape[0] - max_kinship_LG_filt.shape[0]) + " cells filtered by kinship\n")

    return max_kinship_LG


def annotate_lineage_groups(
    dfMT: pd.DataFrame, max_kinship_LG: pd.DataFrame, master_intBCs: Dict[int, pd.DataFrame]
) -> pd.DataFrame:
    """
    Assign cells in the allele table to a lineage group.

    Takes in a DataFrame of alignments and a DataFrame identifying the chosen
    lineage group for each cell and annotates the lineage groups in the 
    original DataFrame.

    Args:
        dfMT: A DataFrame of alignments
        max_kinship_LG: A DataFrame with the max kinship lienage group for each
            cell, see documentation of score_lineage_kinships
        master_intBCs: A dictionary relating lineage group to its set of intBCs

    Returns:
        Original DataFrame with annotated lineage group assignments for cells
    """

    dfMT["lineageGrp"] = 0

    cellBC2LG = {}
    for n in max_kinship_LG.index:
        cellBC2LG[n] = max_kinship_LG.loc[n, "lineageGrp"]

    dfMT["lineageGrp"] = dfMT["cellBC"].map(cellBC2LG)

    dfMT["lineageGrp"] = dfMT["lineageGrp"].fillna(value=0)

    lg_sizes = {}
    rename_lg = {}

    for n, g in dfMT.groupby(["lineageGrp"]):

        if n != 0:
            lg_sizes[n] = len(g["cellBC"].unique())

    sorted_by_value = sorted(lg_sizes.items(), key=lambda kv: kv[1])[::-1]

    for i, tup in zip(range(1, len(sorted_by_value) + 1), sorted_by_value):
        print(i, tup[0], float(i))
        rename_lg[tup[0]] = float(i)

    rename_lg[0] = 0.0

    dfMT["lineageGrp"] = dfMT.apply(lambda x: rename_lg[x.lineageGrp], axis=1)

    return dfMT


def filter_intBCs_final_lineages(
    at: pd.DataFrame, min_intbc_thresh: float = 0.05
) -> List[pd.DataFrame]:
    """Filters out low-proportion intBCs from the final lineages.

    After the assignments of the final lineage groups have been decided, 
    for each intBC removes it from the intBC set of the group if the 
    proportion of cells in the lineage group that have that intBC is <=
    min_intbc_thresh

    Args:
        at: An alignment DataFrame annotated with final lineage groups
        min_intbc_thresh: The proportion threshold by which to filter intBCs
            from each lineage group
    Returns:
        lgs: A list of alignment DataFrames recording the UMI counts, intBCs, 
            and cellBCs of each lineage group, one table for each lineage group
    """

    lineageGrps = at["lineageGrp"].unique()
    at_piv = pd.pivot_table(
        at, index="cellBC", columns="intBC", values="UMI", aggfunc="count"
    )
    at_piv.fillna(value=0, inplace=True)
    at_piv[at_piv > 0] = 1

    lgs = []

    for i in tqdm(lineageGrps):

        lg = at[at["lineageGrp"] == i]
        cells = lg["cellBC"].unique()

        lg_pivot = at_piv.loc[cells]

        props = (
            lg_pivot.apply(lambda x: pylab.sum(x) / len(x))
            .to_frame()
            .reset_index()
        )
        props.columns = ["iBC", "prop"]

        props = props.sort_values(by="prop", ascending=False)
        props.index = props["iBC"]

        p_bc = props[(props["prop"] > min_intbc_thresh) & (props["iBC"] != "NC")]

        lg_group = lg.loc[np.in1d(lg["intBC"], p_bc["iBC"])]
        lgs.append(lg_group)

    return lgs


def filtered_lineage_group_to_allelle_table(
    filtered_lgs: List[pd.DataFrame],
) -> pd.DataFrame:
    """Produces the final allele table as a DataFrame to be used for 
    reconstruction.

    Takes a list of alignment DataFrames annotated with lineage groups and 
    forms a final DataFrame of indel information. 

    Args:
        filtered_lgs: A DataFrame of alignments annotated with lineage groups

    Returns:
        final_df: A final processed DataFrame with indel information
    """

    final_df = pd.concat(filtered_lgs)

    grouping = []
    for i in final_df.columns:
        if i[0] == "r":
            grouping.append(i)
    grouping = ["cellBC", "intBC", "allele"] + grouping + ["lineageGrp"]

    final_df = final_df.groupby(grouping, as_index=False).agg(
        {"UMI": "count", "readCount": "sum"}
    )
    final_df["Sample"] = final_df.apply(
        lambda x: x.cellBC.split(".")[0], axis=1
    )

    return final_df
