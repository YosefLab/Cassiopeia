"""Conversions between allele tables, lineage profiles and character matrices."""

import re
from collections import OrderedDict, defaultdict

import numpy as np
import pandas as pd
import pylab
from tqdm.auto import tqdm

from cassiopeia.mixins import is_ambiguous_state


def convert_alleletable_to_character_matrix(
    alleletable: pd.DataFrame,
    ignore_intbcs: list[str] = None,
    allele_rep_thresh: float = 1.0,
    missing_data_allele: str | None = None,
    missing_data_state: int = -1,
    mutation_priors: pd.DataFrame | None = None,
    cut_sites: list[str] | None = None,
    collapse_duplicates: bool = True,
) -> tuple[pd.DataFrame, dict[int, dict[int, float]], dict[int, dict[int, str]]]:
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
    if ignore_intbcs is None:
        ignore_intbcs = []
    if cut_sites is None:
        cut_sites = get_default_cut_site_columns(alleletable)

    filtered_samples = defaultdict(OrderedDict)
    for sample in alleletable.index:
        cell = alleletable.loc[sample, "cellBC"]
        intBC = alleletable.loc[sample, "intBC"]

        if intBC in ignore_intbcs:
            continue

        for _, c in enumerate(cut_sites):
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
                    if not isinstance(state, str) and np.isnan(state):
                        transformed_states.append(missing_data_state)
                        continue

                    if state == "NONE" or "None" in state:
                        transformed_states.append(0)
                    elif missing_data_allele is not None and state == missing_data_allele:
                        transformed_states.append(missing_data_state)
                    else:
                        if state in allele_counter[c]:
                            transformed_states.append(allele_counter[c][state])
                        else:
                            # if this is the first time we're seeing the state for this character,
                            # add a new entry to the allele_counter
                            allele_counter[c][state] = len(allele_counter[c]) + 1
                            transformed_states.append(allele_counter[c][state])

                            indel_to_charstate[i][len(allele_counter[c])] = state

                            # add a new entry to the character's probability map
                            if mutation_priors is not None:
                                prob = np.mean(mutation_priors.loc[state, "freq"])
                                prior_probs[i][len(allele_counter[c])] = float(prob)

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
    cut_sites: list[str] | None = None,
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

    agg_recipe = dict(zip(list(cut_sites), [list] * len(cut_sites), strict=False))
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
        for val, cutsite in zip(g.loc[j], cut_sites, strict=False):
            if collapse_duplicates:
                # Sort for testing
                val = sorted(set(val))
            val = tuple(val)
            if len(val) == 1:
                val = val[0]
            allele_piv.at[j[0], (j[1], cutsite)] = val

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
    lineage_profile.columns = ["_".join(tup).rstrip("_") for tup in lineage_profile.columns.values]

    return lineage_profile


def convert_lineage_profile_to_character_matrix(
    lineage_profile: pd.DataFrame,
    indel_priors: pd.DataFrame | None = None,
    missing_allele_indicator: str | None = None,
    missing_state_indicator: int = -1,
) -> tuple[pd.DataFrame, dict[int, dict[int, float]], dict[int, dict[int, str]]]:
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
        lineage_profile.replace({missing_allele_indicator: "Missing"}, inplace=True)

    lineage_profile.columns = [f"r{i}" for i in range(lineage_profile.shape[1])]
    column_to_unique_values = dict(
        zip(
            lineage_profile.columns,
            [lineage_profile[x].factorize()[1].values for x in lineage_profile.columns],
            strict=False,
        )
    )

    column_to_number = dict(
        zip(lineage_profile.columns, range(lineage_profile.shape[1]), strict=False)
    )

    mutation_counter = dict(
        zip(lineage_profile.columns, [0] * lineage_profile.shape[1], strict=False)
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
                        prior_probs[c][mutation_to_state[col][indel]] = float(prob)

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
    character_matrix.columns = [f"r{i}" for i in range(lineage_profile.shape[1])]

    return character_matrix, prior_probs, indel_to_charstate


def convert_character_matrix_to_allele_table(
    character_matrix: pd.DataFrame,
    state_to_indel: dict[int, dict[int, str]] | None = None,
    keep_ambiguous: bool = False,
):
    """Converts a character matrix back into an allele table.

    Args:
        character_matrix: A dataframe storing the character states for each
            spot.
        state_to_indel: A mapping of numerical states into indel strings.
        keep_ambiguous: A boolean whether to keep ambiguous states for a given
            intBC

    Returns:
            An allele table.
    """
    allele_table = pd.DataFrame(columns=["cellBC", "intBC", "allele", "r1", "UMI"])

    def disambiguate_allele(char, allele):
        if isinstance(allele, tuple):
            if keep_ambiguous:
                all_alleles = [state_to_indel[int(char)][a] if a != 0 else "None" for a in allele]
                return all_alleles
            else:
                allele = allele[0]
        if allele == 0:
            return ["None"]
        if state_to_indel:
            return [state_to_indel[int(char)][allele]]
        return [allele]

    for cell in tqdm(character_matrix.index):
        alleles = character_matrix.loc[cell].values
        non_missing_iid = np.where(alleles != -1)[0]
        intbcs = []
        all_alleles = []
        for iid in non_missing_iid:
            _alleles = disambiguate_allele(iid, alleles[iid])
            intbcs += [f"intbc-{iid}" for _ in range(len(_alleles))]
            all_alleles += _alleles
        cellbcs = [cell] * len(intbcs)
        umis = [1] * len(intbcs)
        new_rows = pd.DataFrame(
            [cellbcs, intbcs, all_alleles, all_alleles, umis],
            index=allele_table.columns,
        ).T
        allele_table = pd.concat([allele_table, new_rows])
    return allele_table


def compute_empirical_indel_priors(
    allele_table: pd.DataFrame,
    grouping_variables: list[str] = None,
    cut_sites: list[str] | None = None,
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
    if grouping_variables is None:
        grouping_variables = ["intBC"]
    if cut_sites is None:
        cut_sites = get_default_cut_site_columns(allele_table)

    agg_recipe = dict(zip(list(cut_sites), ["unique"] * len(cut_sites), strict=False))
    groups = allele_table.groupby(grouping_variables).agg(agg_recipe)

    indel_count = defaultdict(int)

    for g in groups.index:
        alleles = np.unique(np.concatenate(groups.loc[g].values))
        for a in alleles:
            if "none" not in a.lower():
                indel_count[a] += 1

    tot = len(groups.index)

    indel_freqs = dict(
        zip(list(indel_count.keys()), [v / tot for v in indel_count.values()], strict=False)
    )

    indel_priors = pd.DataFrame([indel_count, indel_freqs]).T
    indel_priors.columns = ["count", "freq"]
    indel_priors.index.name = "indel"

    return indel_priors


def get_default_cut_site_columns(allele_table: pd.DataFrame) -> list[str]:
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
    cut_sites = [column for column in allele_table.columns if bool(re.search(r"r\d", column))]

    return cut_sites
