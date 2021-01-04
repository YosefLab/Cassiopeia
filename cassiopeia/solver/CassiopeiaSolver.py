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

    Args:
        character_matrix: A character matrix of observed character states for
            all samples
        missing_char: The character representing missing values
        meta_data: Any meta data associated with the samples
        priors: Prior probabilities of observing a transition from 0 to any
            character state

    Attributes:
        character_matrix: The character matrix describing the samples
        missing_char: The character representing missing values
        meta_data: Data table storing meta data for each sample
        priors: Prior probabilities of character state transitions
        tree: The tree built by `self.solve()`. None if `solve` has not been
            called yet
    """

    def __init__(
        self,
        character_matrix: pd.DataFrame,
        missing_char: int,
        meta_data: Optional[pd.DataFrame] = None,
        priors: Optional[Dict] = None,
    ):

        self.character_matrix = character_matrix
        self.missing_char = missing_char
        self.meta_data = meta_data
        self.tree = None
        self.priors = priors

    @abc.abstractmethod
    def solve(self):
        """Solves the inference problem."""
        pass
