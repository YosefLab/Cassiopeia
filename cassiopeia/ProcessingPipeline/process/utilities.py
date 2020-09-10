"""
This file stores generally important functionality for the Cassiopeia-Preprocess
pipeline.
"""
import logging

import numpy as np
import pandas as pd
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

  for n, group in tqdm(molecule_table.groupby(['cellBC'])):
      umi_per_cellBC_n = group.shape[0]
      reads_per_cellBC_n = group.agg({'readCount': 'sum'}).readCount
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
  molecule_table['filter'] = molecule_table['cellBC'].map(cell_filter)
  
  n_umi_filt = molecule_table[molecule_table['filter'] == True].shape[0]
  n_cells_filt = len(molecule_table.loc[molecule_table['filter'] == True, 'cellBC'].unique())

  logging.info(f'Filtered out {n_umi_filt} UMIs.')
  logging.info(f'Filtered out {n_cells_filt} cells.')

  filt_molecule_table = molecule_table[molecule_table['filter'] == False].copy()
  filt_molecule_table.drop(columns = ['filter'], inplace=True)
  return filt_molecule_table
