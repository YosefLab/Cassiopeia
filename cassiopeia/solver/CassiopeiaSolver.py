"""
Abstract class CassiopeiaSolver, for the phylogenetic inference module.

All algorithms are derived classes of this abstract class, and at a minimum
store an input character matrix and implement a method called `solve`. Each
derived class stores more information around the parameters necessary for
inferring a phylogenetic tree.
"""
import abc
import pandas as pd
from typing import Dict, Optional


class CassiopeiaSolver(abc.ABC):
    """
  CassiopeiaSolver is an abstract class that all inference algorithms derive
  from. At minimum, all CassiopeiaSolver subclasses will store a character
  matrix and implement a solver procedure.
  """

    def __init__(
        self,
        character_matrix: pd.DataFrame,
        meta_data: Optional[pd.DataFrame] = None,
        priors: Optional[Dict] = None,
    ):

        self.character_matrix = character_matrix
        self.meta_data = meta_data
        self.priors = priors
        self.tree = None

    @abc.abstractmethod
    def solve(self):
        """Solve the inference problem
    """
        pass
