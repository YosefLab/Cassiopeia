"""
Abstract class Cassiopeia-Solver, or the phylogenetic inference module.

All algorithms are derived classes of this abstract class, and at a minimum
store an input character matrix and implement a method called `solve`. Each
derived class stores more information around the parameters necessary for
inferring a phylogenetic tree.
"""
import abc
import pandas as pd


class CassiopeiaSolver(abc.ABC):

  def __init__(self, character_matrix: pd.DataFrame, meta_data: pd.DataFrame):

    self.character_matrix = character_matrix
    self.meta_data = meta_data
    self.tree = None

  @abc.abstractmethod
  def solve(self):
    pass