"""
This file contains all high-level functionality for preprocessing sequencing
data into character matrices ready for phylogenetic inference. This file
is mainly invoked by cassiopeia_preprocess.py.
"""
import os

import logging
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from cassiopeia.ProcessingPipeline.process import utilities


def resolve_UMI_sequence(
    molecule_table: pd.DataFrame,
    output_directory: str,
    min_avg_reads_per_umi: float = 2.0,
    min_umi_per_cell: int = 10,
    plot: bool = True,
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

  if plot:
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

  if plot:
    # ---------------- Plot Diagnositics after Resolving ---------------- #
    h = plt.figure(figsize=(14, 10))
    plt.plot(top_reads.values(), total_numReads.values(), "r.")
    plt.ylabel("Total Reads")
    plt.xlabel("Number Reads for Picked Sequence")
    plt.title("Total vs. Top Reads for Picked Sequence")
    plt.savefig(os.path.join(outputdir, "/total_vs_top_reads_pickSeq.png"))
    plt.close()

    h = plt.figure(figsize=(14, 10))
    plt.plot(first_reads.values(), second_reads.values(), "r.")
    plt.ylabel("Number Reads for Second Best Sequence")
    plt.xlabel("Number Reads for Picked Sequence")
    plt.title("Second Best vs. Top Reads for Picked Sequence")
    plt.savefig(
        os.path.join(outputdir + "/second_vs_top_reads_pickSeq.png")
    )
    plt.close()

  filt_molecule_table = utilities.filter_cells(
      filt_molecule_table, min_umi_per_cell, min_avg_reads_per_umi
  )
  return filt_molecule_table
