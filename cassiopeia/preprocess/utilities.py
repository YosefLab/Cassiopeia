"""
This file stores generally important functionality for the Cassiopeia-Preprocess
pipeline.
"""
import functools
import itertools
import os
import time
from typing import Callable, Dict, List, Optional, Tuple
import warnings

from collections import defaultdict, OrderedDict
import matplotlib
import matplotlib.pyplot as plt
import ngs_tools as ngs
import numpy as np
import pandas as pd
import pylab
import pysam
import re
from tqdm.auto import tqdm

from cassiopeia.mixins import is_ambiguous_state, logger, PreprocessWarning


def log_molecule_table(wrapped: Callable):
    """Function decorator that logs molecule_table stats.

    Simple decorator that logs the number of total reads, the number of unique
    UMIs, and the number of unique cellBCs in a DataFrame that is returned
    from a function.

    Args:
        wrapped: The wrapped original function. Since this is a function
            decorator, this argument is passed implicitly by Python internals.
    """

    @functools.wraps(wrapped)
    def wrapper(*args, **kwargs):
        df = wrapped(*args, **kwargs)
        umi_count = df["UMI"].dtype != object
        logger.debug(
            f"Resulting {'alleletable' if umi_count else 'molecule_table'} statistics:"
        )
        logger.debug(f"# Reads: {df['readCount'].sum()}")
        logger.debug(f"# UMIs: {df['UMI'].sum() if umi_count else df.shape[0]}")
        logger.debug(f"# Cell BCs: {df['cellBC'].nunique()}")
        return df

    return wrapper


def log_runtime(wrapped: Callable):
    """Function decorator that logs the start, end and runtime of a function.

    Args:
        wrapped: The wrapped original function. Since this is a function
            decorator, this argument is passed implicitly by Python internals.
    """

    @functools.wraps(wrapped)
    def wrapper(*args, **kwargs):
        t0 = time.time()
        logger.info("Starting...")
        try:
            return wrapped(*args, **kwargs)
        finally:
            logger.info(f"Finished in {time.time() - t0} s.")

    return wrapper


def log_kwargs(wrapped: Callable):
    """Function decorator that logs the keyword arguments of a function.

    This function only logs keyword arguments because usually the unnamed
    arguments contain Pandas DataFrames, which are difficult to log cleanly as
    text.

    Args:
        wrapped: The wrapped original function. Since this is a function
            decorator, this argument is passed implicitly by Python internals.
    """

    @functools.wraps(wrapped)
    def wrapper(*args, **kwargs):
        logger.debug(f"Keyword arguments: {kwargs}")
        return wrapped(*args, **kwargs)

    return wrapper


@log_molecule_table
def filter_cells(
    molecule_table: pd.DataFrame,
    min_umi_per_cell: int = 10,
    min_avg_reads_per_umi: float = 2.0,
) -> pd.DataFrame:
    """Filter out cell barcodes that have too few UMIs or too few reads/UMI.

    Args:
        molecule_table: A molecule table of cellBC-UMI pairs to be filtered
        min_umi_per_cell: Minimum number of UMIs per cell for cell to not be
            filtered. Defaults to 10.
        min_avg_reads_per_umi: Minimum coverage (i.e. average) reads per
            UMI in a cell needed in order for that cell not to be filtered.
            Defaults to 2.0.

    Returns:
        A filtered molecule table
    """
    # Detect if the UMI column contains UMI counts or the actual UMI sequence
    umi_count = molecule_table["UMI"].dtype != object

    cell_groups = molecule_table.groupby("cellBC")
    umis_per_cell = (
        cell_groups["UMI"].sum() if umi_count else cell_groups.size()
    )
    umis_per_cell_mask = umis_per_cell >= min_umi_per_cell
    avg_reads_per_umi = cell_groups["readCount"].sum() / umis_per_cell
    avg_read_per_umi_mask = avg_reads_per_umi >= min_avg_reads_per_umi

    umis_per_cell_passing = set(umis_per_cell_mask.index[umis_per_cell_mask])
    avg_read_per_umi_passing = set(
        avg_read_per_umi_mask.index[avg_read_per_umi_mask]
    )
    passing_cells = umis_per_cell_passing & avg_read_per_umi_passing
    passing_mask = molecule_table["cellBC"].isin(passing_cells)
    n_cells = molecule_table["cellBC"].nunique()
    logger.info(
        f"Filtered out {n_cells - len(passing_cells)} cells with too few UMIs "
        "or too few average number of reads per UMI."
    )
    molecule_table_filt = molecule_table[~passing_mask]
    n_umi_filt = (
        molecule_table_filt["UMI"].sum()
        if umi_count
        else molecule_table_filt.shape[0]
    )
    logger.info(f"Filtered out {n_umi_filt} UMIs as a result.")
    return molecule_table[passing_mask].copy()


@log_molecule_table
def filter_umis(
    molecule_table: pd.DataFrame, min_reads_per_umi: int = 100
) -> pd.DataFrame:
    """
    Filters out UMIs with too few reads.

    Filters out all UMIs with a read count <= min_reads_per_umi.

    Args:
        molecule_table: A molecule table of cellBC-UMI pairs to be filtered
        min_reads_per_umi: The minimum read count needed for a UMI to not be
            filtered. Defaults to 100.

    Returns:
        A filtered molecule table
    """
    return molecule_table[molecule_table["readCount"] >= min_reads_per_umi]


@log_molecule_table
def error_correct_intbc(
    molecule_table: pd.DataFrame,
    prop: float = 0.5,
    umi_count_thresh: int = 10,
    dist_thresh: int = 1,
) -> pd.DataFrame:
    """
    Error corrects close intBCs with small enough unique UMI counts.

    Considers each pair of intBCs sharing a cellBC in the DataFrame for
    correction. For a pair of intBCs, changes all instances of one to other if:
        1. They have the same allele.
        2. The Levenshtein distance between their sequences is <= dist_thresh.
        3. The number of UMIs of the intBC to be changed is <= umi_count_thresh.
        4. The proportion of the UMI count in the intBC to be changed out of the
        total UMI count in both intBCs <= prop.

    Note:
        Prop should be <= 0.5, as this algorithm only corrects intBCs with
            fewer/equal UMIs towards intBCs with more UMIs. Additionally, if
            multiple intBCs are within the distance threshold of an intBC, it
            corrects the intBC towards the intBC with the most UMIs.

    Args:
        molecule_table: A molecule table of cellBC-UMI pairs to be filtered
        prop: proportion by which to filter integration barcodes
        umi_count_thresh: maximum umi count for which to correct barcodes
        dist_thresh: barcode distance threshold, to decide what's similar
            enough to error correct

    Returns:
        Filtered molecule table with error corrected intBCs
    """
    if prop > 0.5:
        warnings.warn(
            "No intBC correction was done because `prop` is greater than 0.5.",
            PreprocessWarning,
        )
        return molecule_table

    cellBC_intBC_allele_groups = molecule_table.groupby(
        ["cellBC", "intBC", "allele"], sort=False
    )
    cellBC_intBC_allele_indices = cellBC_intBC_allele_groups.groups
    molecule_table_agg = (
        cellBC_intBC_allele_groups.agg({"UMI": "count", "readCount": "sum"})
        .sort_values("UMI", ascending=False)
        .reset_index()
    )
    for (cellBC, allele), intBC_table in tqdm(
        molecule_table_agg.groupby(["cellBC", "allele"], sort=False),
        desc="Error Correcting intBCs",
    ):
        # NOTE: row1 UMIs >= row2 UMIs because groupby operations preserve
        # row orders
        for i1 in range(intBC_table.shape[0]):
            row1 = intBC_table.iloc[i1]
            intBC1 = row1["intBC"]
            UMI1 = row1["UMI"]
            for i2 in range(i1 + 1, intBC_table.shape[0]):
                row2 = intBC_table.iloc[i2]
                intBC2 = row2["intBC"]
                UMI2 = row2["UMI"]
                total_count = UMI1 + UMI2
                proportion = UMI2 / total_count
                distance = ngs.sequence.levenshtein_distance(intBC1, intBC2)

                # Correct
                if (
                    distance <= dist_thresh
                    and proportion < prop
                    and UMI2 <= umi_count_thresh
                ):
                    key_to_correct = (cellBC, intBC2, allele)
                    molecule_table.loc[
                        cellBC_intBC_allele_indices[key_to_correct], "intBC"
                    ] = intBC1

                    logger.info(
                        f"In cellBC {cellBC}, intBC {intBC2} corrected to "
                        f"{intBC1}, correcting {UMI2} UMIs to {UMI1} UMIs."
                    )
    return molecule_table


def record_stats(
    molecule_table: pd.DataFrame,
) -> Tuple[np.array, np.array, np.array]:
    """
    Simple function to record the number of UMIs.

    Args:
        molecule_table: A DataFrame of alignments

    Returns:
        Read counts for each alignment, number of unique UMIs per intBC, number
            of UMIs per cellBC
    """
    umis_per_intBC = (
        molecule_table.groupby(["cellBC", "intBC"], sort=False).size().values
    )
    umis_per_cellBC = molecule_table.groupby("cellBC", sort=False).size().values

    return (
        molecule_table["readCount"].values,
        umis_per_intBC,
        umis_per_cellBC,
    )


def convert_bam_to_df(data_fp: str) -> pd.DataFrame:
    """Converts a BAM file to a Pandas dataframe.

    Args:
        data_fp: The input filepath for the BAM file to be converted.

    Returns:
        A Pandas dataframe containing the BAM information.
    """
    als = []
    with pysam.AlignmentFile(
        data_fp, ignore_truncation=True, check_sq=False
    ) as bam_fh:
        for al in bam_fh:
            cellBC, UMI, readCount, grpFlag = al.query_name.split("_")
            seq = al.query_sequence
            qual = al.query_qualities
            encode_qual = pysam.array_to_qualitystring(qual)
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
    return pd.DataFrame(
        als,
        columns=[
            "cellBC",
            "UMI",
            "readCount",
            "grpFlag",
            "seq",
            "qual",
            "readName",
        ],
    )


def convert_alleletable_to_character_matrix(
    alleletable: pd.DataFrame,
    ignore_intbcs: List[str] = [],
    allele_rep_thresh: float = 1.0,
    missing_data_allele: Optional[str] = None,
    missing_data_state: int = -1,
    mutation_priors: Optional[pd.DataFrame] = None,
    cut_sites: Optional[List[str]] = None,
    collapse_duplicates: bool = True,
) -> Tuple[
    pd.DataFrame, Dict[int, Dict[int, float]], Dict[int, Dict[int, str]]
]:
    """Converts an AlleleTable into a character matrix.

    Given an AlleleTable storing the observed mutations for each intBC / cellBC
    combination, create a character matrix for input into a CassiopeiaSolver
    object. By default, we codify uncut mutations as '0' and missing data items
    as '-1'. The function also have the ability to ignore certain intBC sets as
    well as cut sites with too little diversity.

    Args:
        alleletable: Allele Table to be converted into a character matrix
        ignore_intbcs: A set of intBCs to ignore
        allele_rep_thresh: A threshold for removing target sites that have an
            allele represented by this proportion
        missing_data_allele: Value in the allele table that indicates that the
            cut-site is missing. This will be converted into
            ``missing_data_state``
        missing_data_state: A state to use for missing data.
        mutation_priors: A table storing the prior probability of a mutation
            occurring. This table is used to create a character matrix-specific
            probability dictionary for reconstruction.
        cut_sites: Columns in the AlleleTable to treat as cut sites. If None,
            we assume that the cut-sites are denoted by columns of the form
            "r{int}" (e.g. "r1")
        collapse_duplicates: Whether or not to collapse duplicate character
            states present for a single cellBC-intBC pair. This option has no
            effect if there are no allele conflicts. Defaults to True.

    Returns:
        A character matrix, a probability dictionary, and a dictionary mapping
            states to the original mutation.
    """
    if cut_sites is None:
        cut_sites = get_default_cut_site_columns(alleletable)

    filtered_samples = defaultdict(OrderedDict)
    for sample in alleletable.index:
        cell = alleletable.loc[sample, "cellBC"]
        intBC = alleletable.loc[sample, "intBC"]

        if intBC in ignore_intbcs:
            continue

        for i, c in enumerate(cut_sites):
            if intBC not in ignore_intbcs:
                filtered_samples[cell].setdefault(f"{intBC}{c}", []).append(
                    alleletable.loc[sample, c]
                )

    character_strings = defaultdict(list)
    allele_counter = defaultdict(OrderedDict)

    _intbc_uniq = set()
    allele_dist = defaultdict(list)
    for s in filtered_samples:
        for key in filtered_samples[s]:
            _intbc_uniq.add(key)
            allele_dist[key].extend(list(set(filtered_samples[s][key])))

    # remove intBCs that are not diverse enough
    intbc_uniq = []
    dropped = []
    for key in allele_dist.keys():
        props = np.unique(allele_dist[key], return_counts=True)[1]
        props = props / len(allele_dist[key])
        if np.any(props > allele_rep_thresh):
            dropped.append(key)
        else:
            intbc_uniq.append(key)

    print(
        "Dropping the following intBCs due to lack of diversity with threshold "
        + str(allele_rep_thresh)
        + ": "
        + str(dropped)
    )

    prior_probs = defaultdict(dict)
    indel_to_charstate = defaultdict(dict)
    # for all characters
    for i in tqdm(range(len(list(intbc_uniq))), desc="Processing characters"):

        c = list(intbc_uniq)[i]
        indel_to_charstate[i] = {}

        # for all samples, construct a character string
        for sample in filtered_samples.keys():

            if c in filtered_samples[sample]:

                # This is a list of states
                states = filtered_samples[sample][c]
                transformed_states = []

                for state in states:

                    if type(state) != str and np.isnan(state):
                        transformed_states.append(missing_data_state)
                        continue

                    if state == "NONE" or "None" in state:
                        transformed_states.append(0)
                    elif (
                        missing_data_allele is not None
                        and state == missing_data_allele
                    ):
                        transformed_states.append(missing_data_state)
                    else:
                        if state in allele_counter[c]:
                            transformed_states.append(allele_counter[c][state])
                        else:
                            # if this is the first time we're seeing the state for this character,
                            # add a new entry to the allele_counter
                            allele_counter[c][state] = (
                                len(allele_counter[c]) + 1
                            )
                            transformed_states.append(allele_counter[c][state])

                            indel_to_charstate[i][
                                len(allele_counter[c])
                            ] = state

                            # add a new entry to the character's probability map
                            if mutation_priors is not None:
                                prob = np.mean(
                                    mutation_priors.loc[state, "freq"]
                                )
                                prior_probs[i][len(allele_counter[c])] = float(
                                    prob
                                )

                if collapse_duplicates:
                    # Sort for testing
                    transformed_states = sorted(set(transformed_states))
                transformed_states = tuple(transformed_states)
                if len(transformed_states) == 1:
                    transformed_states = transformed_states[0]
                character_strings[sample].append(transformed_states)

            else:
                character_strings[sample].append(missing_data_state)

    character_matrix = pd.DataFrame.from_dict(
        character_strings,
        orient="index",
        columns=[f"r{i}" for i in range(1, len(intbc_uniq) + 1)],
    )

    return character_matrix, prior_probs, indel_to_charstate


def convert_alleletable_to_lineage_profile(
    allele_table,
    cut_sites: Optional[List[str]] = None,
    collapse_duplicates: bool = True,
) -> pd.DataFrame:
    """Converts an AlleleTable to a lineage profile.

    Takes in an allele table that summarizes the indels observed at individual
    cellBC-intBC pairs and produces a lineage profile, which essentially is a
    pivot table over the cellBC / intBCs. Conceptually, these lineage profiles
    are identical to character matrices, only the values in the matrix are the
    actual indel identities.

    Args:
        allele_table: AlleleTable.
        cut_sites: Columns in the AlleleTable to treat as cut sites. If None,
            we assume that the cut-sites are denoted by columns of the form
            "r{int}" (e.g. "r1")
        collapse_duplicates: Whether or not to collapse duplicate character
            states present for a single cellBC-intBC pair. This option has no
            effect if there are no allele conflicts. Defaults to True.

    Returns:
        An NxM lineage profile.
    """

    if cut_sites is None:
        cut_sites = get_default_cut_site_columns(allele_table)

    agg_recipe = dict(
        zip([cutsite for cutsite in cut_sites], [list] * len(cut_sites))
    )
    g = allele_table.groupby(["cellBC", "intBC"]).agg(agg_recipe)
    intbcs = allele_table["intBC"].unique()

    # create mutltindex df by hand
    i1 = []
    for i in intbcs:
        i1 += [i] * len(cut_sites)
        i2 = list(cut_sites) * len(intbcs)

    indices = [i1, i2]

    allele_piv = pd.DataFrame(index=g.index.levels[0], columns=indices)
    for j in tqdm(g.index, desc="filling in multiindex table"):
        for val, cutsite in zip(g.loc[j], cut_sites):
            if collapse_duplicates:
                # Sort for testing
                val = sorted(set(val))
            val = tuple(val)
            if len(val) == 1:
                val = val[0]
            allele_piv.loc[j[0]][j[1], cutsite] = val

    allele_piv2 = pd.pivot_table(
        allele_table,
        index=["cellBC"],
        columns=["intBC"],
        values="UMI",
        aggfunc=pylab.size,
    )
    col_order = (
        allele_piv2.dropna(axis=1, how="all")
        .sum()
        .sort_values(ascending=False, inplace=False)
        .index
    )

    lineage_profile = allele_piv[col_order]

    # collapse column names here
    lineage_profile.columns = [
        "_".join(tup).rstrip("_") for tup in lineage_profile.columns.values
    ]

    return lineage_profile


def convert_lineage_profile_to_character_matrix(
    lineage_profile: pd.DataFrame,
    indel_priors: Optional[pd.DataFrame] = None,
    missing_allele_indicator: Optional[str] = None,
    missing_state_indicator: int = -1,
) -> Tuple[
    pd.DataFrame, Dict[int, Dict[int, float]], Dict[int, Dict[int, str]]
]:
    """Converts a lineage profile to a character matrix.

    Takes in a lineage profile summarizing the explicit indel identities
    observed at each cut site in a cell and converts this into a character
    matrix where the indels are abstracted into integers.

    Note:
        The lineage profile is converted directly into a character matrix,
        without performing any collapsing of duplicate states. Instead, this
        should have been done in the previous step, when calling
        :func:`convert_alleletable_to_lineage_profile`.

    Args:
        lineage_profile: Lineage profile
        indel_priors: Dataframe mapping indels to prior probabilities
        missing_allele_indicator: An allele that is being used to represent
            missing data.
        missing_state_indicator: State to indicate missing data

    Returns:
        A character matrix, prior probability dictionary, and mapping from
            character/state pairs to indel identities.
    """

    prior_probs = defaultdict(dict)
    indel_to_charstate = defaultdict(dict)

    lineage_profile = lineage_profile.copy()

    lineage_profile = lineage_profile.fillna("Missing").copy()
    if missing_allele_indicator:
        lineage_profile.replace(
            {missing_allele_indicator: "Missing"}, inplace=True
        )

    samples = []

    lineage_profile.columns = [f"r{i}" for i in range(lineage_profile.shape[1])]
    column_to_unique_values = dict(
        zip(
            lineage_profile.columns,
            [
                lineage_profile[x].factorize()[1].values
                for x in lineage_profile.columns
            ],
        )
    )

    column_to_number = dict(
        zip(lineage_profile.columns, range(lineage_profile.shape[1]))
    )

    mutation_counter = dict(
        zip(lineage_profile.columns, [0] * lineage_profile.shape[1])
    )
    mutation_to_state = defaultdict(dict)

    for col in column_to_unique_values.keys():

        c = column_to_number[col]
        indel_to_charstate[c] = {}

        for indels in column_to_unique_values[col]:
            if not is_ambiguous_state(indels):
                indels = (indels,)

            for indel in indels:
                if indel == "Missing" or indel == "NC":
                    mutation_to_state[col][indel] = -1

                elif "none" in indel.lower():
                    mutation_to_state[col][indel] = 0

                elif indel not in mutation_to_state[col]:
                    mutation_to_state[col][indel] = mutation_counter[col] + 1
                    mutation_counter[col] += 1

                    indel_to_charstate[c][mutation_to_state[col][indel]] = indel

                    if indel_priors is not None:
                        prob = np.mean(indel_priors.loc[indel]["freq"])
                        prior_probs[c][mutation_to_state[col][indel]] = float(
                            prob
                        )

    # Helper function to apply to lineage profile
    def apply_mutation_to_state(x):
        column = []
        for v in x.values:
            if is_ambiguous_state(v):
                column.append(tuple(mutation_to_state[x.name][_v] for _v in v))
            else:
                column.append(mutation_to_state[x.name][v])
        return column

    character_matrix = lineage_profile.apply(apply_mutation_to_state, axis=0)

    character_matrix.index = lineage_profile.index
    character_matrix.columns = [
        f"r{i}" for i in range(lineage_profile.shape[1])
    ]

    return character_matrix, prior_probs, indel_to_charstate


def compute_empirical_indel_priors(
    allele_table: pd.DataFrame,
    grouping_variables: List[str] = ["intBC"],
    cut_sites: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Computes indel prior probabilities.

    Generates indel prior probabilities from the input allele table. The general
    idea behind this procedure is to count the number of times an indel
    independently occur. By default, we treat each intBC as an independent,
    which is true if the input allele table is a clonal population. Here, the
    procedure will count the number of intBCs that contain a particular indel
    and divide by the number of intBCs in the allele table. However, a user can
    be more nuanced in their analysis and group intBC by other variables, such
    as lineage group (this is especially useful if intBCs might occur several
    clonal populations). Then, the procedure will count the number of times an
    indel occurs in a unique lineage-intBC combination.

    Args:
        allele_table: AlleleTable
        grouping_variables: Variables to stratify data by, to treat as
            independent groups in counting indel occurrences. These must be
            columns in the allele table
        cut_sites: Columns in the AlleleTable to treat as cut sites. If None,
            we assume that the cut-sites are denoted by columns of the form
            "r{int}" (e.g. "r1")

    Returns:
        A DataFrame mapping indel identities to the probability.
    """

    if cut_sites is None:
        cut_sites = get_default_cut_site_columns(allele_table)

    agg_recipe = dict(
        zip([cut_site for cut_site in cut_sites], ["unique"] * len(cut_sites))
    )
    groups = allele_table.groupby(grouping_variables).agg(agg_recipe)

    indel_count = defaultdict(int)

    for g in groups.index:

        alleles = np.unique(np.concatenate(groups.loc[g].values))
        for a in alleles:
            if "none" not in a.lower():
                indel_count[a] += 1

    tot = len(groups.index)

    indel_freqs = dict(
        zip(list(indel_count.keys()), [v / tot for v in indel_count.values()])
    )

    indel_priors = pd.DataFrame([indel_count, indel_freqs]).T
    indel_priors.columns = ["count", "freq"]
    indel_priors.index.name = "indel"

    return indel_priors


def get_default_cut_site_columns(allele_table: pd.DataFrame) -> List[str]:
    """Retrieves the default cut-sites columns of an AlleleTable.

    A basic helper function that will retrieve the cut-sites from an AlleleTable
    if the AlleleTable was created using the Cassiopeia pipeline. In this case,
    each cut-site is denoted by an integer preceded by the character "r", for
    example "r1" or "r2".

    Args:
        allele_table: AlleleTable

    Return:
        Columns in the AlleleTable corresponding to the cut sites.
    """
    cut_sites = [
        column
        for column in allele_table.columns
        if bool(re.search(r"r\d", column))
    ]

    return cut_sites
