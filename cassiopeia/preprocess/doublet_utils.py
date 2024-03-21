"""
This file contains functions pertaining to filtering cell doublets.
Invoked through pipeline.py and supports the filter_molecule_table and
call_lineage_groups functions.
"""
from typing import Dict, Set, Tuple, List

import pandas as pd

from cassiopeia.mixins import logger
from cassiopeia.preprocess import utilities


@utilities.log_molecule_table
def filter_intra_doublets(
    molecule_table: pd.DataFrame, prop: float = 0.1
) -> pd.DataFrame:
    """Filters cells that present too much conflicting allele information.

    For each cellBC, calculates the most common allele for each intBC by UMI
    count. Also calculates the proportion of UMIs of alleles that conflict
    with the most common. If the proportion across all UMIs is > prop, filters
    out alignments with that cellBC from the DataFrame.

    Args:
        molecule_table: A molecule table of cellBC-UMI pairs to be filtered
        prop: The threshold representing the minimum proportion of conflicting
        UMIs needed to filter out a cellBC from the DataFrame

    Returns
        A filtered molecule table
    """
    umis_per_allele = (
        molecule_table.groupby(["cellBC", "intBC", "allele"])["UMI"]
        .size()
        .reset_index()
        .sort_values("UMI", ascending=False)
    )
    umis_per_allele_unique = umis_per_allele.drop_duplicates(
        ["cellBC", "intBC"]
    )
    umis_per_cellBC = umis_per_allele.groupby("cellBC")["UMI"].sum()
    conflicting_umis_per_cellBC = (
        umis_per_cellBC - umis_per_allele_unique.groupby("cellBC")["UMI"].sum()
    )
    prop_multi_alleles_per_cellBC = (
        conflicting_umis_per_cellBC / umis_per_cellBC
    )
    passing_mask = prop_multi_alleles_per_cellBC <= prop
    passing_cellBCs = set(prop_multi_alleles_per_cellBC.index[passing_mask])
    logger.debug(
        f"Filtered {(~passing_mask).sum()} cellBCs with too much conflicitng "
        "allele information."
    )
    return molecule_table[molecule_table["cellBC"].isin(passing_cellBCs)]


def get_intbc_set(
    lg: pd.DataFrame, thresh: int = None
) -> Tuple[Set[str], Dict[str, float]]:
    """A simple function to return the intBC set of a lineage group.

    Given a lineage groups, returns the intBC set for that lineage
    group, i.e. the set of intBCs that the cells in the lineage group have.
    If thresh is specified, also filters out intBCs with low proportions.

    Args:
        lg: An allele table, representing a single lineage group
        thresh: The threshold representing the minimum proportion of cells that
            have an intBC needed in each lineage group in order for that intBC
            to be included in the intBC set

    Returns:
        A list containing the intBCs in the lineage group, and a dictionary
            with intBCs as keys and the proportion of cells that do not have
            that intBC in each lineage group as values.
    """
    n_cells = lg["cellBC"].nunique()
    intBC_groups = lg.groupby("intBC")["cellBC"]
    cellBC_per_intBC = intBC_groups.nunique()
    dropouts = 1 - (cellBC_per_intBC / n_cells)
    intBCs = (
        dropouts.index if thresh is None else dropouts.index[dropouts < thresh]
    )
    return set(intBCs), dict(dropouts)


def compute_lg_membership(
    cell: pd.DataFrame,
    intbc_sets: Dict[int, Set[str]],
    lg_dropouts: Dict[int, Dict[str, float]],
) -> Dict[int, float]:
    """Calculates the kinship for the given cell for each lineage group.

    Given a cell, for each lineage group, it calculates the intBC intersection
    with that lineage group, weighted by the cell proportions that have each
    intBC in that group.

    Args:
        cell: An allele table subsetted to one cellBC
        intbc_sets: A dictionary of the intBC sets of each lineage group
        lg_dropouts: A dictionary of the cell proportion of cells that do not
            have that intBC for each lineage group

    Returns:
        A kinship score for each lineage group
    """

    lg_mem = {}

    # Get the intBC set for the cell
    intBCs = set(cell["intBC"].unique())
    for lg_key in intbc_sets:
        lg_do = lg_dropouts[lg_key]
        # Calculate the intersect
        intersect = intBCs & intbc_sets[lg_key]
        if intersect:
            # Calculate weighted intersection, normalized by the total cell
            # proportions
            lg_mem[lg_key] = (
                len(intersect) - sum(lg_do[intBC] for intBC in intersect)
            ) / (len(lg_do) - sum(lg_do.values()))
        else:
            lg_mem[lg_key] = 0

    # Normalize the membership values across linaege groups
    factor = 1.0 / sum(lg_mem.values())
    for l in lg_mem:
        lg_mem[l] = lg_mem[l] * factor

    return lg_mem


def filter_inter_doublets(at: pd.DataFrame, rule: float = 0.35) -> pd.DataFrame:
    """Filters out cells whose kinship with their assigned lineage is low.

    Essentially, filters out cells that have ambigious kinship across multiple
    lineage groups. For every cell, calculates the kinship it has with its
    assigned lineage, with kinship defined as the weighted proportion of intBCs
    it shares with the intBC set for a lineage (see compute_lg_membership for
    more details on the weighting). If that kinship is <= rule, then it is
    filtered out.

    Args:
        at: An allele table of cellBC-intBC-allele groups to be filtered
        rule: The minimum kinship threshold which a cell needs to pass in order
            to be included in the final DataFrame

    Returns:
        A filtered allele table
    """
    ibc_sets = {}
    dropouts = {}
    for lg_name, at_lg in at.groupby("lineageGrp"):
        ibc_sets[lg_name], dropouts[lg_name] = get_intbc_set(at_lg)

    # Calculate kinship for each lineage group for each cell
    n_filtered = 0
    passing_cellBCs = []
    for cellBC, at_cellBC in at.groupby("cellBC"):
        lg = int(at_cellBC["lineageGrp"].iloc[0])
        mem = compute_lg_membership(at_cellBC, ibc_sets, dropouts)
        if mem[lg] < rule:
            n_filtered += 1
        else:
            passing_cellBCs.append(cellBC)

    n_cells = at["cellBC"].nunique()
    logger.debug(f"Filtered {n_filtered} inter-doublets of {n_cells} cells")
    return at[at["cellBC"].isin(passing_cellBCs)]


def filter_doublet_lg_sets(
        PIV: pd.DataFrame,
        master_LGs: List[int], 
        master_intBCs: Dict[int, List[str]]
) -> Tuple[List[int], Dict[int, List[str]]]:
    """Filters out lineage groups that are likely doublets.

    Essentially, filters out lineage groups that have a high proportion of
    intBCs that are shared with other lineage groups. For every lineage group,
    calculates the proportion of intBCs that are shared with every pair of two
    other lineage groups. If the proportion is > .8, then the lineage group
    is filtered out.

    Args:
        PIV: A pivot table of cellBC-intBC-allele groups to be filtered
        master_LGs: A list of lineage groups to be filtered
        master_intBCs: A dictionary that has mappings from the lineage group
            number to the set of intBCs being used for reconstruction

    Returns:
        A filtered list of lineage groups and a filtered dictionary of intBCs
            for each lineage group
    """
    lg_sorted = (PIV.value_counts('lineageGrp')
        .reset_index().sort_values('lineageGrp', ascending=False))

    for lg in lg_sorted['lineageGrp']:
        lg = tuple([lg])
        filtered = False
        lg_intBC = set(master_intBCs[lg])
        for lg_i in master_LGs:
            for lg_j in master_LGs:
                if lg == lg_i or lg == lg_j:
                    continue
                pair_intBC = set(master_intBCs[lg_i]).union(set(master_intBCs[lg_j]))
                if len(pair_intBC.intersection(lg_intBC)) > len(lg_intBC) * .8:
                    master_LGs.remove(lg)
                    master_intBCs.pop(lg)
                    logger.debug(f"Filtered lineage group {lg} as a doublet"
                                 f" of {lg_i} and {lg_j}")
                    filtered = True
                    break
            if filtered:
                break

    return master_LGs, master_intBCs