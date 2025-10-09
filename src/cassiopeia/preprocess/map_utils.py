"""Module containing functions pertaining to mapping intBCs.

Invoked through pipeline.py and supports the filter_molecule_table function.
"""

import pandas as pd

from cassiopeia.mixins import logger
from cassiopeia.preprocess import utilities


@utilities.log_molecule_table
def map_intbcs(molecule_table: pd.DataFrame) -> pd.DataFrame:
    """Assign one allele to each intBC/cellBC pair.

    For each intBC/cellBC pairing, selects the most frequent allele (by read
    count, and then by UMI) and removes alignments that don't have that allele.

    Args:
        molecule_table: A molecule table of cellBC-UMI pairs to be filtered

    Returns
    -------
        An allele table with one allele per cellBC-intBC pair
    """
    # Have to drop out all intBCs that are NaN
    molecule_table = molecule_table.dropna(subset=["intBC"])

    # For each cellBC-intBC pair, select the allele that has the highest
    # readCount; on ties, use UMI count
    allele_table = (
        molecule_table.groupby(["cellBC", "intBC", "allele"])
        .agg({"readCount": "sum", "UMI": "count"})
        .reset_index()
        .sort_values(["UMI", "readCount"], ascending=False)
    )
    duplicated_mask = allele_table.duplicated(["cellBC", "intBC"])
    mapped_alleles = set(
        allele_table[~duplicated_mask][["cellBC", "intBC", "allele"]].itertuples(index=False, name=None)
    )

    # True for rows that contain the mapped allele; False for ones to filter out
    selection_mask = molecule_table[["cellBC", "intBC", "allele"]].apply(tuple, axis=1).isin(mapped_alleles)

    mapped_table = molecule_table[selection_mask]
    logger.debug(f"Alleles removed: {duplicated_mask.sum()}")
    logger.debug(f"UMIs removed: {(~selection_mask).sum()}")
    return mapped_table
