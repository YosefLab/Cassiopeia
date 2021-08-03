"""
This file contains functions pertaining to calling lineage groups.
Invoked through pipeline.py and supports the call_lineage_group function.
"""
import os
import sys
import time

from typing import Dict, List, Tuple

from matplotlib import colors, colorbar
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pylab
import re

from cassiopeia.mixins import logger

sys.setrecursionlimit(10000)


def assign_lineage_groups(
    pivot_in: pd.DataFrame,
    min_clust_size: int,
    min_intbc_thresh: float = 0.2,
    kinship_thresh: float = 0.2,
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

    Returns:
        piv_assigned: A pivot table of cells labled with lineage group
            assignments
    """
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
        )

        # append returned objects to output variable
        piv_assigned = piv_assigned.append(piv_lg, sort=True)

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
) -> Tuple[pd.DataFrame, pd.DataFrame]:

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

    Returns:
        PIV_LG: A pivot table of cells labled with lineage group assignments
        PIV_noLG: A pivot table of the remaining unassigned cells
    """

    # Calculate sum of observed intBCs, identify top intBC
    intBC_sums = PIVOT_in.sum(0).sort_values(ascending=False)
    intBC_top = intBC_sums.index[0]

    # Take subset of PIVOT table that contain cells that have the top intBC
    subPIVOT_in = PIVOT_in[PIVOT_in[intBC_top] > 0]
    subPIVOT_in_sums = subPIVOT_in.sum(0)
    ordered_intBCs2 = subPIVOT_in_sums.sort_values(
        ascending=False
    ).index.tolist()
    subPIVOT_in = subPIVOT_in[ordered_intBCs2]

    # Binarize
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
    PIV_LG = PIVOT_in.iloc[PIVOT_in.index.isin(LG_cells), :].copy()
    PIV_LG["lineageGrp"] = iteration + 1

    # Print statements
    logger.debug(
        f"LG {iteration+1} Assignment: {PIV_LG.shape[0]} cells assigned"
    )

    return PIV_LG, PIV_noLG


def filter_intbcs_lg_sets(
    PIV_assigned: pd.DataFrame, min_intbc_thresh: float = 0.2
) -> Tuple[List[int], Dict[int, pd.DataFrame]]:
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
        # Drop the lineageGrp column
        PIVi_bin = PIVi_bin.drop(["lineageGrp"], axis=1)
        PIVi_bin[PIVi_bin > 0] = 1

        intBC_sums = PIVi_bin.sum(0)
        intBC_normsums = intBC_sums / max(intBC_sums)

        intBC_normsums_filt_i = intBC_normsums[
            intBC_normsums >= min_intbc_thresh
        ]
        intBC_set_i = intBC_normsums_filt_i.index.tolist()

        # Update masters
        master_intBCs[i] = intBC_set_i
        master_LGs.append(i)

    return master_LGs, master_intBCs


def score_lineage_kinships(
    PIV: pd.DataFrame,
    master_LGs: List[int],
    master_intBCs: Dict[int, pd.DataFrame],
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
        dfLG2intBC = dfLG2intBC.append(dfi, sort=True)

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

    max_kinship_LG = pd.concat(
        [max_kinship_frame, max_kinship_ind + 1], axis=1, sort=True
    )
    max_kinship_LG.columns = ["maxOverlap", "lineageGrp"]

    return max_kinship_LG


def annotate_lineage_groups(
    dfMT: pd.DataFrame,
    max_kinship_LG: pd.DataFrame,
    master_intBCs: Dict[int, pd.DataFrame],
) -> pd.DataFrame:
    """
    Assign cells in the allele table to a lineage group.

    Takes in an allele table and a DataFrame identifying the chosen
    lineage group for each cell and annotates the lineage groups in the
    original DataFrame.

    Args:
        dfMT: An allele table of cellBC-UMI-allele groups
        max_kinship_LG: A DataFrame with the max kinship lineage group for each
            cell, see documentation of score_lineage_kinships
        master_intBCs: A dictionary relating lineage group to its set of intBCs

    Returns:
        Original allele table with annotated lineage group assignments for cells
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
        rename_lg[tup[0]] = float(i)

    rename_lg[0] = 0.0

    dfMT["lineageGrp"] = dfMT.apply(lambda x: rename_lg[x.lineageGrp], axis=1)

    return dfMT


def filter_intbcs_final_lineages(
    at: pd.DataFrame, min_intbc_thresh: float = 0.05
) -> List[pd.DataFrame]:
    """Filters out low-proportion intBCs from the final lineages.

    After the assignments of the final lineage groups have been decided,
    for each intBC-lineage group pair, removes all alignments with that
    intBC-lineage pair from the final alignment DataFrame if the proportion
    of cells in that group that have that intBC is <= min_intbc_thresh.

    Args:
        at: An allele table of cellBC-UMI-allele groups annotated with final lineage groups
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

    for i in lineageGrps:

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

        p_bc = props[
            (props["prop"] > min_intbc_thresh) & (props["iBC"] != "NC")
        ]

        lg_group = lg.loc[np.in1d(lg["intBC"], p_bc["iBC"])]
        lgs.append(lg_group)

    return lgs


def filtered_lineage_group_to_allele_table(
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

    final_df = pd.concat(filtered_lgs, sort=True)

    grouping = []
    for i in final_df.columns:
        if bool(re.search(r"r\d", i)):
            grouping.append(i)
    grouping = ["cellBC", "intBC", "allele"] + grouping + ["lineageGrp"]

    final_df = final_df.groupby(grouping, as_index=False).agg(
        {"UMI": "count", "readCount": "sum"}
    )

    final_df["Sample"] = final_df.apply(
        lambda x: x.cellBC.split(".")[0], axis=1
    )

    return final_df


def plot_overlap_heatmap(at, at_pivot_I, output_directory):
    """Generates a plot showing the overlap of intBC sets, indicating clones.

    Generates a plot with cellBCs as the rows and intBCs as the columns. Shows
    which intBCs are contained by which cells, with cells sharing a lot of
    overlap indicating that they might belong to the same clonal population.

    Args:
        at: An allele table of cellBC-UMI-allele groups
        at_pivot_I: A pivot table of indicators indicating which cellBCs have
            which UMIs
        output_directory: The directory in which to store the plot

    Returns:
        None, plot is saved to output directory
    """

    # Close old plots
    plt.close()

    flat_master = []
    for n, lg in at.groupby("lineageGrp"):

        for item in lg["intBC"].unique():
            flat_master.append(item)

    at_pivot_I = at_pivot_I[flat_master]

    h2 = plt.figure(figsize=(20, 20))
    axmat2 = h2.add_axes([0.3, 0.1, 0.6, 0.8])
    im2 = axmat2.matshow(at_pivot_I, aspect="auto", origin="upper")

    plt.savefig(os.path.join(output_directory, "clustered_intbc.png"))
    plt.close()


def plot_overlap_heatmap_lg(at, at_pivot_I, output_directory):
    """Generates a plot of the allele table.

    Generates a plot where the rows are cellBCs and the columns are cutsites
    for the sequence of each intBC. Colors indicate the allele information
    relative to the reference sequence, with red indicating a deletion, blue
    indicating an insertion, gray indicating an uncut site (matches the
    reference), and white indicating that intBC is not found in that cell. The
    bar chart on the left indicates the UMI count of each cellBC, and the bars
    on the bottom indicate the number of unique mutations observed at each
    cutsite.

    Args:
        at: An allele table of cellBC-UMI-allele groups
        at_pivot_I: A pivot table of indicators indicating which cellBCs have
            which UMIs
        output_directory: The directory in which to store the plot

    Returns:
        None, plot is saved to output directory
    """

    if not os.path.exists(
        os.path.join(output_directory, "lineageGrp_piv_heatmaps")
    ):
        os.makedirs(os.path.join(output_directory, "lineageGrp_piv_heatmaps"))

    for n, lg_group in at.groupby("lineageGrp"):

        plt.close()

        lg_group = add_cutsite_encoding(lg_group)

        s_cmap = colors.ListedColormap(["grey", "red", "blue"], N=3)

        values = [i for i in lg_group.columns if re.findall(r"s\d", i)]

        lg_group_pivot = pd.pivot_table(
            lg_group,
            index=["cellBC"],
            columns=["intBC"],
            values=values,
            aggfunc=pylab.mean,
        ).T
        lg_group_pivot2 = pd.pivot_table(
            lg_group,
            index=["cellBC"],
            columns=["intBC"],
            values="UMI",
            aggfunc=pylab.size,
        )

        cell_umi_count = (
            lg_group.groupby(["cellBC"])
            .agg({"UMI": "count"})
            .sort_values(by="UMI")
        )

        agg_dict = {}
        for i in lg_group.columns:
            if re.search(r"r\d", i):
                agg_dict[i] = "nunique"
        n_unique_alleles = lg_group.groupby(["intBC"]).agg(agg_dict)

        col_order = (
            lg_group_pivot2.dropna(axis=1, how="all")
            .sum()
            .sort_values(ascending=False, inplace=False)
            .index
        )

        if len(col_order) < 2:
            continue

        s3 = lg_group_pivot.unstack(level=0).T
        s3 = s3[col_order]
        s3 = s3.T.stack(level=1).T

        s3 = s3.loc[cell_umi_count.index]

        # s3_2 = (
        #     lg_group_pivot2.dropna(axis=1, how="all")
        #     .sum()
        #     .sort_values(ascending=False, inplace=False)[col_order]
        # )

        n_unique_alleles = n_unique_alleles.loc[col_order]
        s3_intBCs = col_order
        s3_cellBCs = s3.index.tolist()

        # Plot heatmap
        h = plt.figure(figsize=(14, 10))

        ax = h.add_axes([0.3, 0.1, 0.6, 0.8], frame_on=True)
        im = ax.matshow(s3, aspect="auto", origin="lower", cmap=s_cmap)

        axx1 = plt.xticks(
            range(1, len(col_order) * len(agg_dict), len(agg_dict)),
            col_order,
            rotation="vertical",
            family="monospace",
        )

        ax3 = h.add_axes([0.2, 0.1, 0.1, 0.8], frame_on=True)
        plt.barh(range(s3.shape[0]), cell_umi_count["UMI"])
        plt.ylim([0, s3.shape[0]])
        ax3.autoscale(tight=True)

        axy0 = ax3.set_yticks(range(len(s3_cellBCs)))
        axy1 = ax3.set_yticklabels(s3_cellBCs, family="monospace")

        w = 1 / len(agg_dict)
        x = np.arange(len(s3_intBCs)) - w * ((len(agg_dict) - 1) // 2)
        ax2 = h.add_axes([0.3, 0, 0.6, 0.1], frame_on=False)
        for i in agg_dict.keys():
            num = int(re.findall(r"\d", i)[0])
            b = ax2.bar(
                x + (num - 1) * w, n_unique_alleles[i], width=w, label=i
            )
        ax2.set_xlim([0, len(s3_intBCs)])
        ax2.set_ylim(
            ymin=0,
            ymax=(
                max([n_unique_alleles[i].max() for i in agg_dict.keys()]) + 10
            ),
        )
        ax2.set_xticks([])
        ax2.yaxis.tick_right()
        ax2.invert_yaxis()
        ax2.autoscale(tight=True)
        plt.legend()

        # plt.gcf().subplots_adjust(bottom=0.15)
        # plt.tight_layout()
        plt.savefig(
            os.path.join(
                output_directory,
                "lineageGrp_piv_heatmaps/lg_"
                + str(int(n))
                + "_piv_heatmap.png",
            )
        )
        plt.close()


def add_cutsite_encoding(lg_group):
    """Adds the encoding for the mutation type at each cutsite for each cellBC.

    Args:
        lg_group: A pivot table representing a lineage group

    Returns:
        A pivot table with cutsite encodings
    """
    cutsites = []

    for i in lg_group.columns:
        digit = re.findall(r"\d", i)
        if digit:
            cutsites.append(i)
            lg_group["s" + digit[0]] = 0

    for i in lg_group.index:
        for r in cutsites:
            if lg_group.loc[i, r] == "['None']":
                lg_group.loc[i, r.replace("r", "s")] = 0.9
            elif "D" in lg_group.loc[i, r]:
                lg_group.loc[i, r.replace("r", "s")] = 1.9
            elif "I" in lg_group.loc[i, r]:
                lg_group.loc[i, r.replace("r", "s")] = 2.9

    return lg_group
